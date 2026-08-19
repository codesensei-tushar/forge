"""Context assembly and context-window management.

Two jobs, both about what the model can see:

**Assembly** — :func:`gather_environment` collects the facts an engineer would
want before touching an unfamiliar repo (where am I, what kind of project is
this, what's the git state) and folds them into the system prompt. Without it
the agent wastes its first two or three iterations rediscovering the obvious.

**Management** — :class:`ContextManager` keeps the conversation under the token
ceiling by compacting old tool output. A long agent run accumulates enormous
tool results; without compaction the run dies partway through with "context
full", which is the worst possible time to stop. Compaction always preserves the
``tool_use``/``tool_result`` pairing the API requires.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from forge.agent.state import AgentState
from forge.providers.types import Message, TextBlock, ToolResultBlock
from forge.tools.context import ToolContext

# Files whose presence identifies the project's toolchain, in priority order.
_PROJECT_MARKERS: tuple[tuple[str, str], ...] = (
    ("pyproject.toml", "Python (pyproject)"),
    ("setup.py", "Python (setuptools)"),
    ("requirements.txt", "Python (requirements.txt)"),
    ("Cargo.toml", "Rust (cargo)"),
    ("go.mod", "Go (modules)"),
    ("package.json", "Node/JavaScript"),
    ("pom.xml", "Java (Maven)"),
    ("build.gradle", "Java/Kotlin (Gradle)"),
    ("Gemfile", "Ruby (Bundler)"),
    ("composer.json", "PHP (Composer)"),
    ("CMakeLists.txt", "C/C++ (CMake)"),
    ("Makefile", "Make"),
)

_MAX_TOP_LEVEL_ENTRIES = 40

# When compacting, how much of a large tool result to keep at each end.
_COMPACT_HEAD = 400
_COMPACT_TAIL = 400
_COMPACT_THRESHOLD = 1_200


@dataclass
class Environment:
    """Static facts about the workspace, rendered into the system prompt."""

    workspace: str
    project_kind: str = "unknown"
    top_level: list[str] = field(default_factory=list)
    git_summary: str = ""
    sandbox: str = ""

    def render(self) -> str:
        lines = [f"Workspace root: {self.workspace}", f"Project type: {self.project_kind}"]
        if self.sandbox:
            lines.append(f"Shell sandbox: {self.sandbox}")
        if self.top_level:
            lines.append("Top-level contents: " + ", ".join(self.top_level))
        if self.git_summary:
            lines.append(f"Git status:\n{self.git_summary}")
        return "\n".join(lines)


def _detect_project_kind(root: Path) -> str:
    kinds = [label for marker, label in _PROJECT_MARKERS if (root / marker).is_file()]
    return ", ".join(kinds) if kinds else "unknown"


def _top_level(root: Path) -> list[str]:
    from forge.tools.filesystem import IGNORE_DIRS

    try:
        children = sorted(root.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    except OSError:
        return []
    names: list[str] = []
    for child in children:
        if child.name.startswith(".") and child.name not in {".github", ".gitignore"}:
            continue
        if child.is_dir():
            if child.name in IGNORE_DIRS:
                continue
            names.append(f"{child.name}/")
        else:
            names.append(child.name)
        if len(names) >= _MAX_TOP_LEVEL_ENTRIES:
            names.append("...")
            break
    return names


async def gather_environment(ctx: ToolContext) -> Environment:
    """Collect workspace facts. Never raises — a partial picture beats none."""
    root = ctx.workspace_root
    env = Environment(
        workspace=str(root),
        project_kind=_detect_project_kind(root),
        top_level=_top_level(root),
        sandbox=ctx.sandbox.describe(),
    )
    try:
        from forge.tools.git import collect_repo_summary

        env.git_summary = await collect_repo_summary(ctx)
    except Exception:  # noqa: BLE001 - context gathering must never fail a run
        env.git_summary = ""
    return env


class ContextManager:
    """Keeps a conversation inside its token budget by compacting old output."""

    def __init__(self, *, max_tokens: int, keep_recent: int = 6) -> None:
        self.max_tokens = max_tokens
        # Recent turns are what the model is actively reasoning over; only
        # compact things older than this many messages.
        self.keep_recent = keep_recent

    def over_budget(self, state: AgentState) -> bool:
        return state.estimate_tokens() > self.max_tokens

    def compact(self, state: AgentState) -> int:
        """Shrink history to fit the budget. Returns tokens reclaimed.

        Two passes, least-destructive first: truncate the bodies of old tool
        results, then drop whole old tool-call exchanges. Both preserve API
        validity — a ``tool_use`` block always keeps its matching
        ``tool_result``, because dropping one without the other is rejected.
        """
        before = state.estimate_tokens()
        if before <= self.max_tokens:
            return 0

        self._truncate_old_tool_results(state)
        if state.estimate_tokens() > self.max_tokens:
            self._drop_old_exchanges(state)

        reclaimed = before - state.estimate_tokens()
        state.compactions += 1
        return reclaimed

    def _mutable_range(self, state: AgentState) -> range:
        """Indices eligible for compaction: past the task, before recent turns."""
        end = max(1, len(state.messages) - self.keep_recent)
        return range(1, end)

    def _truncate_old_tool_results(self, state: AgentState) -> None:
        for index in self._mutable_range(state):
            message = state.messages[index]
            for block in message.content:
                if not isinstance(block, ToolResultBlock):
                    continue
                if len(block.content) <= _COMPACT_THRESHOLD:
                    continue
                omitted = len(block.content) - _COMPACT_HEAD - _COMPACT_TAIL
                block.content = (
                    f"{block.content[:_COMPACT_HEAD]}\n"
                    f"[... {omitted} chars of earlier output dropped to free context ...]\n"
                    f"{block.content[-_COMPACT_TAIL:]}"
                )
            if state.estimate_tokens() <= self.max_tokens:
                return

    def _drop_old_exchanges(self, state: AgentState) -> None:
        """Remove the oldest assistant-tool-call/tool-result pairs entirely."""
        while state.estimate_tokens() > self.max_tokens:
            index = self._find_droppable_pair(state)
            if index is None:
                return
            del state.messages[index : index + 2]
            state.messages.insert(
                index,
                Message.user("[Earlier tool calls in this session were dropped to free context.]"),
            )

    def _find_droppable_pair(self, state: AgentState) -> int | None:
        # Keep index 0 (the original task) and the most recent turns.
        limit = len(state.messages) - self.keep_recent
        for index in range(1, max(1, limit) - 1):
            current, following = state.messages[index], state.messages[index + 1]
            if current.role != "assistant" or not current.tool_uses():
                continue
            if following.role == "user" and any(
                isinstance(b, ToolResultBlock) for b in following.content
            ):
                return index
        return None


def summarize_history(state: AgentState, *, limit: int = 400) -> str:
    """A one-paragraph digest of the conversation, for logs and JSON output."""
    texts = [
        block.text
        for message in state.messages
        for block in message.content
        if isinstance(block, TextBlock) and block.text.strip()
    ]
    joined = " ".join(texts)
    return joined if len(joined) <= limit else joined[: limit - 3] + "..."
