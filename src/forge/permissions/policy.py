"""Permission policy: decide whether a tool call runs, is denied, or needs a human.

The decision is a pure function of (tool risk, call target, configured rules),
which makes it unit-testable in isolation from the agent and the terminal.

Rules, in order:

1. Any ``deny`` pattern matching the target → ``DENY``. This wins over
   everything, including allow-lists and ``--yolo``.
2. ``READ`` risk → ``ALLOW``. Observation is always free.
3. A tool granted "always allow" for this session → ``ALLOW``.
4. Any ``allow`` pattern that prefixes the target → ``ALLOW``.
5. ``yolo`` mode → ``ALLOW``.
6. ``WRITE`` risk in ``auto`` mode → ``ALLOW``.
7. Otherwise → ``ASK`` (the UI prompts the human).

A call's risk is the higher of the tool's own assessment and whatever the
configured ``destructive`` patterns say about the target, so ``rm -rf build``
still reaches a human even though ``shell`` is nominally a write-risk tool.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from forge.config import ApprovalMode, Settings
from forge.permissions.risk import Risk

if TYPE_CHECKING:
    # Annotations only: importing forge.tools at runtime would be a cycle, since
    # the tool executor depends on this module.
    from forge.tools.base import Tool


class Decision(StrEnum):
    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"


class Approval(StrEnum):
    """A human's answer to an :attr:`Decision.ASK` prompt."""

    ALLOW = "allow"  # this call only
    ALWAYS = "always"  # this call and every later call to the same tool
    DENY = "deny"


@dataclass(frozen=True)
class PermissionResult:
    decision: Decision
    reason: str = ""
    risk: Risk = Risk.WRITE


def describe_target(tool: Tool[Any], args: dict[str, Any]) -> str:
    """A single string representing the effect of this call, for rule matching."""
    for key in ("command", "path", "pattern"):
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            # Shell commands match bare so `deny = ["git push"]` reads naturally;
            # everything else is namespaced by tool to avoid cross-tool collisions.
            return value.strip() if key == "command" else f"{tool.name} {value.strip()}"
    return tool.name


class PermissionPolicy:
    def __init__(self, settings: Settings) -> None:
        self._allow = [p for p in settings.allow if p]
        self._deny = [p for p in settings.deny if p]
        self._destructive = [p for p in settings.destructive if p]
        self._mode = settings.approval_mode
        self._session_allow_tools: set[str] = set()

    @property
    def mode(self) -> ApprovalMode:
        return self._mode

    def always_allow_tool(self, name: str) -> None:
        """Grant a tool auto-approval for the remainder of the session."""
        self._session_allow_tools.add(name)

    def effective_risk(self, tool: Tool[Any], args: dict[str, Any]) -> Risk:
        """The risk of this specific call, escalated by configured patterns."""
        risk = tool.risk_for(args)
        if risk is Risk.DESTRUCTIVE:
            return risk
        target = describe_target(tool, args)
        if any(pattern in target for pattern in self._destructive):
            return Risk.DESTRUCTIVE
        return risk

    def decide(self, tool: Tool[Any], args: dict[str, Any]) -> PermissionResult:
        target = describe_target(tool, args)
        risk = self.effective_risk(tool, args)

        for pattern in self._deny:
            if pattern in target:
                return PermissionResult(Decision.DENY, f"matches deny rule: {pattern!r}", risk)

        if risk is Risk.READ:
            return PermissionResult(Decision.ALLOW, "read-only", risk)

        if tool.name in self._session_allow_tools:
            return PermissionResult(Decision.ALLOW, "session grant", risk)

        for pattern in self._allow:
            if target.startswith(pattern):
                return PermissionResult(Decision.ALLOW, f"matches allow rule: {pattern!r}", risk)

        if self._mode is ApprovalMode.YOLO:
            return PermissionResult(Decision.ALLOW, "yolo mode", risk)

        if risk is Risk.WRITE and self._mode is ApprovalMode.AUTO:
            return PermissionResult(Decision.ALLOW, "write allowed in auto mode", risk)

        why = "destructive action" if risk is Risk.DESTRUCTIVE else "write in cautious mode"
        return PermissionResult(Decision.ASK, why, risk)
