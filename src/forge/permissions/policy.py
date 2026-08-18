"""Permission policy: decide whether a tool call runs, is denied, or needs a human.

Rules, in order:

1. Read-only tools always ``ALLOW``.
2. Any ``deny`` pattern found as a substring of the target → ``DENY`` (wins over
   everything, including allow-lists and ``--yes``).
3. A tool granted "always allow" for this session → ``ALLOW``.
4. Any ``allow`` pattern that is a prefix of the target → ``ALLOW``.
5. ``auto_approve`` (``--yes``) → ``ALLOW``.
6. Otherwise → ``ASK`` (the UI prompts the human).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from forge.config import Settings
from forge.tools.base import Tool


class Decision(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"


@dataclass
class PermissionResult:
    decision: Decision
    reason: str = ""


def _match_target(tool: Tool[Any], args: dict[str, Any]) -> str:
    """A single string representing the effect of this call, for matching."""
    if tool.name == "shell":
        return str(args.get("command", "")).strip()
    path = args.get("path")
    return f"{tool.name} {path}".strip() if path else tool.name


class PermissionPolicy:
    def __init__(self, settings: Settings) -> None:
        self._allow = list(settings.allow)
        self._deny = list(settings.deny)
        self._auto_approve = settings.auto_approve
        self._session_allow_tools: set[str] = set()

    def always_allow_tool(self, name: str) -> None:
        """Grant a tool auto-approval for the remainder of the session."""
        self._session_allow_tools.add(name)

    def decide(self, tool: Tool[Any], args: dict[str, Any]) -> PermissionResult:
        if tool.read_only:
            return PermissionResult(Decision.ALLOW, "read-only tool")

        target = _match_target(tool, args)

        for pattern in self._deny:
            if pattern and pattern in target:
                return PermissionResult(Decision.DENY, f"matches deny rule: {pattern!r}")

        if tool.name in self._session_allow_tools:
            return PermissionResult(Decision.ALLOW, "session grant")

        for pattern in self._allow:
            if pattern and target.startswith(pattern):
                return PermissionResult(Decision.ALLOW, f"matches allow rule: {pattern!r}")

        if self._auto_approve:
            return PermissionResult(Decision.ALLOW, "auto-approve")

        return PermissionResult(Decision.ASK, "requires approval")
