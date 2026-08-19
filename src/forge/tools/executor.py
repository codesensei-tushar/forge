"""The tool-call pipeline: the one path from a model's request to a result.

::

    ToolUseBlock -> lookup -> validate args -> permission check -> [approval]
                 -> execute (timeout-guarded) -> ToolResult -> ToolResultBlock

The agent loop hands a :class:`~forge.providers.types.ToolUseBlock` to
:meth:`ToolExecutor.execute` and gets back a block it can append to the
conversation. Nothing here raises for an expected failure: an unknown tool, bad
arguments, a denial, a timeout, and a crashing tool all become an error *result*
the model can read and recover from. That property is what keeps a long agent
run from dying on one bad call.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import ValidationError

from forge.config import Settings
from forge.logging import get_logger
from forge.permissions.policy import (
    Approval,
    Decision,
    PermissionPolicy,
    PermissionResult,
    describe_target,
)
from forge.providers.types import ToolResultBlock, ToolUseBlock
from forge.tools.base import Risk, Tool, ToolResult
from forge.tools.context import ToolContext
from forge.tools.registry import ToolRegistry

# Guidance appended to refusals so the model changes course instead of retrying
# the identical call and burning the iteration budget.
_DENY_HINT = "Do not retry this call. Either take a different approach or explain what you need."


class Approver(Protocol):
    """Asks a human to approve a gated call. Must never raise."""

    async def __call__(self, tool: Tool[Any], target: str, perm: PermissionResult) -> Approval: ...


async def deny_all(tool: Tool[Any], target: str, perm: PermissionResult) -> Approval:
    """Default approver for non-interactive runs: refuse anything needing a human."""
    return Approval.DENY


@dataclass
class ToolOutcome:
    """Everything the loop and the trace need to know about one tool call."""

    name: str
    block: ToolResultBlock
    decision: str
    risk: Risk
    duration_s: float
    result: ToolResult | None = None

    @property
    def is_error(self) -> bool:
        return self.block.is_error


class ToolExecutor:
    """Runs model-requested tool calls under validation, permission, and timeout."""

    def __init__(
        self,
        *,
        registry: ToolRegistry,
        policy: PermissionPolicy,
        ctx: ToolContext,
        settings: Settings,
        approver: Approver | None = None,
    ) -> None:
        self.registry = registry
        self.policy = policy
        self.ctx = ctx
        self.settings = settings
        self.approver: Approver = approver or deny_all
        self.log = get_logger("forge.tools.executor")

    def _refuse(
        self, tool_use: ToolUseBlock, message: str, *, risk: Risk, decision: str
    ) -> ToolOutcome:
        return ToolOutcome(
            name=tool_use.name,
            block=ToolResultBlock(tool_use_id=tool_use.id, content=message, is_error=True),
            decision=decision,
            risk=risk,
            duration_s=0.0,
        )

    def timeout_for(self, tool: Tool[Any], args: dict[str, Any]) -> int:
        """Wall-clock ceiling for this call, as a backstop above any inner timeout."""
        if tool.timeout is not None:
            return tool.timeout
        requested = args.get("timeout")
        if isinstance(requested, int) and requested > 0:
            # Leave the tool room to report its own timeout more helpfully.
            return requested + 15
        return self.settings.tool_timeout

    async def execute(self, tool_use: ToolUseBlock) -> ToolOutcome:
        tool = self.registry.get(tool_use.name)
        if tool is None:
            available = ", ".join(self.registry.names())
            return self._refuse(
                tool_use,
                f"Unknown tool {tool_use.name!r}. Available tools: {available}.",
                risk=Risk.READ,
                decision="unknown",
            )

        # --- permission ---
        perm = self.policy.decide(tool, tool_use.input)
        if perm.decision is Decision.DENY:
            return self._refuse(
                tool_use,
                f"Denied by policy ({perm.reason}). {_DENY_HINT}",
                risk=perm.risk,
                decision="deny",
            )

        decision_label = perm.decision.value
        if perm.decision is Decision.ASK:
            target = describe_target(tool, tool_use.input)
            approval = await self.approver(tool, target, perm)
            if approval is Approval.DENY:
                return self._refuse(
                    tool_use,
                    f"The user declined this action. {_DENY_HINT}",
                    risk=perm.risk,
                    decision="deny",
                )
            if approval is Approval.ALWAYS:
                self.policy.always_allow_tool(tool.name)
            decision_label = "ask->allow"

        # --- validate ---
        try:
            args = tool.parse_args(tool_use.input)
        except ValidationError as exc:
            return self._refuse(
                tool_use,
                f"Invalid arguments for {tool.name}: {_format_validation_error(exc)}",
                risk=perm.risk,
                decision=decision_label,
            )

        # --- execute ---
        started = time.perf_counter()
        timeout = self.timeout_for(tool, tool_use.input)
        try:
            result = await asyncio.wait_for(tool.run(args, self.ctx), timeout)
        except TimeoutError:
            result = ToolResult.error(f"{tool.name} exceeded its {timeout}s time limit.")
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - never let a tool crash the loop
            self.log.warning("tool_crashed", tool=tool.name, error=repr(exc))
            result = ToolResult.error(f"{tool.name} raised an unexpected error: {exc!r}")
        duration = time.perf_counter() - started

        self.log.info(
            "tool_call",
            tool=tool.name,
            decision=decision_label,
            risk=perm.risk.value,
            duration_s=round(duration, 3),
            is_error=result.is_error,
        )
        return ToolOutcome(
            name=tool.name,
            block=ToolResultBlock(
                tool_use_id=tool_use.id, content=result.content, is_error=result.is_error
            ),
            decision=decision_label,
            risk=perm.risk,
            duration_s=duration,
            result=result,
        )


def _format_validation_error(exc: ValidationError) -> str:
    """Render a pydantic error compactly enough to be useful in a tool result."""
    parts = []
    for err in exc.errors():
        location = ".".join(str(p) for p in err["loc"]) or "(root)"
        parts.append(f"{location}: {err['msg']}")
    return "; ".join(parts)
