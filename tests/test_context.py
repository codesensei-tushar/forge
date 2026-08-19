"""Tests for context assembly and context-window management.

Compaction is the difference between a long run degrading and a long run dying,
so the properties that matter are: it frees real tokens, it never touches the
recent turns the model is reasoning over, and it never breaks the
``tool_use``/``tool_result`` pairing the API requires.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from forge.agent.context import (
    ContextManager,
    Environment,
    _detect_project_kind,
    _top_level,
    gather_environment,
    summarize_history,
)
from forge.agent.state import AgentState
from forge.providers.types import Message, TextBlock, ToolResultBlock, ToolUseBlock
from forge.tools.context import ToolContext


# --------------------------------------------------------------------------- #
# History builders
# --------------------------------------------------------------------------- #
def tool_call(index: int) -> Message:
    return Message(
        role="assistant",
        content=[ToolUseBlock(id=f"t{index}", name="shell", input={"command": "ls"})],
    )


def tool_output(index: int, size: int) -> Message:
    return Message.tool_results([ToolResultBlock(tool_use_id=f"t{index}", content="X" * size)])


def history(pairs: int, *, size: int) -> AgentState:
    """A task, ``pairs`` tool exchanges, and a final assistant reply."""
    state = AgentState(system_prompt="")
    state.add_user("task")
    for index in range(1, pairs + 1):
        state.add_message(tool_call(index))
        state.add_message(tool_output(index, size))
    state.add_message(Message.assistant("done"))
    return state


def pairing_is_valid(state: AgentState) -> bool:
    """Every tool_result must follow its tool_use, and no call may be orphaned."""
    pending: set[str] = set()
    for message in state.messages:
        results = {b.tool_use_id for b in message.content if isinstance(b, ToolResultBlock)}
        if results - pending:
            return False
        pending -= results
        pending |= {block.id for block in message.tool_uses()}
    return not pending


# --------------------------------------------------------------------------- #
# Budget
# --------------------------------------------------------------------------- #
def test_over_budget_uses_the_estimate() -> None:
    state = AgentState(system_prompt="x" * 4_000)  # ~1000 tokens
    assert ContextManager(max_tokens=500).over_budget(state)
    assert not ContextManager(max_tokens=5_000).over_budget(state)


def test_compact_is_a_noop_under_budget() -> None:
    state = history(2, size=100)
    assert ContextManager(max_tokens=1_000_000).compact(state) == 0
    assert state.compactions == 0


# --------------------------------------------------------------------------- #
# Pass 1: truncate old tool output
# --------------------------------------------------------------------------- #
def test_old_tool_output_is_truncated_in_place() -> None:
    state = AgentState(system_prompt="")
    state.add_user("task")
    state.add_message(tool_call(1))
    state.add_message(tool_output(1, 5_000))
    state.add_message(Message.assistant("ok"))
    state.add_user("next")
    state.add_message(Message.assistant("done"))

    reclaimed = ContextManager(max_tokens=300, keep_recent=2).compact(state)

    assert reclaimed > 0
    assert state.compactions == 1
    assert len(state.messages) == 6, "truncation alone must not drop messages"
    block = state.messages[2].content[0]
    assert isinstance(block, ToolResultBlock)
    assert "chars of earlier output dropped to free context" in block.content
    assert len(block.content) < 1_000
    assert block.content.startswith("X" * 400)
    assert block.content.endswith("X" * 400)


def test_recent_turns_are_never_compacted() -> None:
    """Whatever the model is currently reasoning over stays byte-for-byte."""
    state = history(1, size=5_000)
    original = state.messages[2].content[0].content  # type: ignore[union-attr]

    ContextManager(max_tokens=1, keep_recent=6).compact(state)

    assert state.messages[2].content[0].content == original  # type: ignore[union-attr]


def test_small_tool_output_is_left_alone() -> None:
    state = history(3, size=50)
    ContextManager(max_tokens=1, keep_recent=2).compact(state)
    for message in state.messages:
        for block in message.content:
            if isinstance(block, ToolResultBlock):
                assert "dropped to free context" not in block.content


# --------------------------------------------------------------------------- #
# Pass 2: drop whole exchanges
# --------------------------------------------------------------------------- #
def test_exchanges_are_dropped_when_truncation_is_not_enough() -> None:
    state = history(4, size=3_000)
    before = len(state.messages)

    reclaimed = ContextManager(max_tokens=200, keep_recent=2).compact(state)

    assert reclaimed > 0
    assert len(state.messages) < before
    assert "dropped to free context" in state.messages[1].text()


def test_dropping_preserves_tool_call_pairing() -> None:
    """An unpaired tool_use or tool_result is rejected by the API outright."""
    state = history(4, size=3_000)
    assert pairing_is_valid(state)

    ContextManager(max_tokens=200, keep_recent=2).compact(state)

    assert pairing_is_valid(state)


def test_the_original_task_is_never_dropped() -> None:
    state = history(4, size=3_000)
    ContextManager(max_tokens=200, keep_recent=2).compact(state)
    assert state.messages[0].text() == "task"


def test_compaction_terminates_when_nothing_is_droppable() -> None:
    """The loop must give up rather than spin on an incompressible history."""
    state = AgentState(system_prompt="x" * 100_000)
    state.add_user("task")
    state.add_message(Message.assistant("reply"))

    reclaimed = ContextManager(max_tokens=1, keep_recent=1).compact(state)

    assert reclaimed == 0
    assert len(state.messages) == 2


def test_the_most_recent_exchange_survives() -> None:
    state = history(4, size=3_000)
    ContextManager(max_tokens=200, keep_recent=2).compact(state)

    remaining = {block.id for m in state.messages for block in m.tool_uses()}
    assert "t4" in remaining, "the newest exchange is the one still in play"


# --------------------------------------------------------------------------- #
# Environment assembly
# --------------------------------------------------------------------------- #
def test_environment_renders_only_what_it_has() -> None:
    bare = Environment(workspace="/w").render()
    assert bare == "Workspace root: /w\nProject type: unknown"

    full = Environment(
        workspace="/w",
        project_kind="Python (pyproject)",
        top_level=["src/", "README.md"],
        git_summary="## main",
        sandbox="local host",
    ).render()
    assert "Shell sandbox: local host" in full
    assert "Top-level contents: src/, README.md" in full
    assert "Git status:\n## main" in full


@pytest.mark.parametrize(
    ("marker", "expected"),
    [
        ("pyproject.toml", "Python (pyproject)"),
        ("Cargo.toml", "Rust (cargo)"),
        ("go.mod", "Go (modules)"),
        ("package.json", "Node/JavaScript"),
        ("Makefile", "Make"),
    ],
)
def test_project_kind_is_detected_from_markers(workspace: Path, marker: str, expected: str) -> None:
    (workspace / marker).write_text("")
    assert _detect_project_kind(workspace) == expected


def test_project_kind_lists_every_match(workspace: Path) -> None:
    (workspace / "pyproject.toml").write_text("")
    (workspace / "Makefile").write_text("")
    assert _detect_project_kind(workspace) == "Python (pyproject), Make"


def test_unknown_project_kind(workspace: Path) -> None:
    assert _detect_project_kind(workspace) == "unknown"


def test_top_level_hides_noise(workspace: Path) -> None:
    for name in ("src", "node_modules", ".venv", ".github"):
        (workspace / name).mkdir()
    for name in ("README.md", ".hidden", ".gitignore"):
        (workspace / name).write_text("")

    listing = _top_level(workspace)

    assert ".github/" in listing and "src/" in listing
    assert "README.md" in listing and ".gitignore" in listing
    assert "node_modules/" not in listing
    assert ".venv/" not in listing
    assert ".hidden" not in listing


def test_top_level_is_capped(workspace: Path) -> None:
    for i in range(60):
        (workspace / f"f{i:03d}.txt").write_text("")
    listing = _top_level(workspace)
    assert listing[-1] == "..."
    assert len(listing) == 41


def test_top_level_survives_an_unreadable_root(workspace: Path) -> None:
    assert _top_level(workspace / "does-not-exist") == []


async def test_gather_environment_describes_the_workspace(
    workspace: Path, ctx: ToolContext
) -> None:
    (workspace / "pyproject.toml").write_text("[project]\nname='x'\n")

    env = await gather_environment(ctx)

    assert env.workspace == str(workspace)
    assert env.project_kind == "Python (pyproject)"
    assert "pyproject.toml" in env.top_level
    assert "local host" in env.sandbox
    assert env.git_summary == "", "not a repository"


async def test_gather_environment_never_fails_a_run(
    ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A partial picture beats aborting the task before it starts."""

    async def exploding(_: ToolContext) -> str:
        raise RuntimeError("git blew up")

    monkeypatch.setattr("forge.tools.git.collect_repo_summary", exploding)
    env = await gather_environment(ctx)
    assert env.git_summary == ""
    assert env.workspace == str(ctx.workspace_root)


# --------------------------------------------------------------------------- #
# History digest
# --------------------------------------------------------------------------- #
def test_summarize_history_joins_text_blocks() -> None:
    state = AgentState(system_prompt="")
    state.add_user("first")
    state.add_message(tool_call(1))
    state.add_message(Message.assistant("second"))
    assert summarize_history(state) == "first second"


def test_summarize_history_truncates() -> None:
    state = AgentState(system_prompt="")
    state.add_user("y" * 500)
    digest = summarize_history(state, limit=50)
    assert len(digest) == 50
    assert digest.endswith("...")


def test_summarize_empty_history() -> None:
    assert summarize_history(AgentState(system_prompt="")) == ""


def test_summarize_skips_tool_output() -> None:
    state = AgentState(system_prompt="")
    state.add_user("task")
    state.add_message(tool_output(1, 50))
    assert summarize_history(state) == "task"


def test_estimate_counts_every_block_type() -> None:
    state = AgentState(system_prompt="s" * 4)
    state.add_message(Message(role="assistant", content=[TextBlock(text="t" * 8)]))
    state.add_message(tool_output(1, 12))
    assert state.estimate_tokens() == (4 + 8 + 12) // 4
