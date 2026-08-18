"""The agent runtime: a robust observe -> decide -> approve -> execute loop.

Deliberately more than ``while True: llm()``:
- model input is validated against each tool's schema before execution;
- every tool runs inside try/except so a failure feeds an error back to the
  model instead of crashing the run;
- permission decisions gate mutating calls (allow / deny / ask-a-human);
- hard guards on iteration count and estimated context size;
- graceful Ctrl-C and provider-error handling;
- every step is recorded to a RunTrace for the end-of-run summary.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from forge.agent.prompts import build_system_prompt
from forge.agent.state import AgentState
from forge.config import Settings
from forge.logging import get_logger
from forge.models.base import ModelProvider, ProviderError
from forge.models.registry import create_provider
from forge.models.types import ModelResponse, ToolResultBlock, ToolUseBlock
from forge.observability.trace import RunTrace
from forge.permissions.policy import Decision, PermissionPolicy
from forge.tools import ToolContext, ToolRegistry, default_registry
from forge.tools.base import ToolResult
from forge.ui.console import Approval, Console


@dataclass
class AgentResult:
    status: str  # completed | max_iterations | max_context | error | aborted
    final_text: str
    trace: RunTrace
    state: AgentState


class AgentRuntime:
    def __init__(
        self,
        *,
        provider: ModelProvider,
        registry: ToolRegistry,
        policy: PermissionPolicy,
        ctx: ToolContext,
        console: Console,
        settings: Settings,
    ) -> None:
        self.provider = provider
        self.registry = registry
        self.policy = policy
        self.ctx = ctx
        self.console = console
        self.settings = settings
        self.log = get_logger("forge.agent")

    async def run_task(self, task: str, state: AgentState | None = None) -> AgentResult:
        if state is None:
            state = AgentState(
                system_prompt=build_system_prompt(
                    self.registry, workspace=str(self.ctx.workspace_root)
                )
            )
        state.add_user(task)
        trace = RunTrace(task=task)
        tools_schema = self.registry.to_provider_schema()
        budget = self.settings.max_iterations
        start_iter = state.iterations
        status = "running"
        final_text = ""

        try:
            while state.iterations - start_iter < budget:
                state.iterations += 1

                if state.estimate_tokens() > self.settings.max_context_tokens:
                    self.console.warning(
                        "Context ceiling reached; stopping. (Compaction arrives in Phase 4.)"
                    )
                    status = "max_context"
                    break

                response = await self._complete(state, tools_schema, trace)
                state.add_message(response.to_message())

                text = response.text()
                if text:
                    self.console.assistant_text(text)
                    final_text = text

                tool_uses = response.tool_uses()
                if not tool_uses:
                    status = "completed"
                    break

                results = [await self._execute_tool_use(tu, trace) for tu in tool_uses]
                state.add_tool_results(results)
            else:
                status = "max_iterations"
                self.console.warning(f"Reached max iterations ({budget}).")
        except KeyboardInterrupt:
            status = "aborted"
            self.console.warning("Aborted by user.")
        except ProviderError as exc:
            status = "error"
            final_text = str(exc)
            self.console.error(str(exc))

        trace.finish(status)
        self.console.run_summary(trace)
        self.log.info("run_finished", status=status, iterations=state.iterations - start_iter)
        return AgentResult(status=status, final_text=final_text, trace=trace, state=state)

    async def _complete(
        self, state: AgentState, tools_schema: list[dict[str, Any]], trace: RunTrace
    ) -> ModelResponse:
        t0 = time.perf_counter()
        with self.console.status("thinking..."):
            response = await self.provider.complete(
                system=state.system_prompt,
                messages=state.messages,
                tools=tools_schema,
                max_tokens=self.settings.max_tokens,
                temperature=self.settings.temperature,
            )
        latency = time.perf_counter() - t0
        trace.record_model_call(
            model=response.model or self.provider.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_s=latency,
            stop_reason=response.stop_reason,
        )
        self.log.info(
            "model_call",
            latency_s=round(latency, 2),
            stop_reason=response.stop_reason,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
        return response

    async def _execute_tool_use(self, tu: ToolUseBlock, trace: RunTrace) -> ToolResultBlock:
        self.console.tool_call(tu.name, tu.input)
        tool = self.registry.get(tu.name)
        if tool is None:
            msg = f"Unknown tool: {tu.name}"
            self.console.error(msg)
            trace.record_tool_call(name=tu.name, decision="deny", duration_s=0.0, is_error=True)
            return ToolResultBlock(tool_use_id=tu.id, content=msg, is_error=True)

        perm = self.policy.decide(tool, tu.input)
        if perm.decision is Decision.DENY:
            self.console.denied(tu.name, perm.reason)
            trace.record_tool_call(name=tu.name, decision="deny", duration_s=0.0, is_error=True)
            return ToolResultBlock(
                tool_use_id=tu.id,
                content=f"Denied by policy ({perm.reason}). Do not retry; try another approach.",
                is_error=True,
            )

        decision_label = "allow"
        if perm.decision is Decision.ASK:
            target = str(tu.input.get("command") or tu.input.get("path") or tu.name)
            approval = self.console.ask_approval(tu.name, target, perm)
            if approval is Approval.DENY:
                self.console.denied(tu.name, "denied by user")
                trace.record_tool_call(
                    name=tu.name, decision="deny", duration_s=0.0, is_error=True
                )
                return ToolResultBlock(
                    tool_use_id=tu.id,
                    content="Denied by the user. Do not retry the same action; "
                    "consider an alternative or ask what to do instead.",
                    is_error=True,
                )
            if approval is Approval.ALWAYS:
                self.policy.always_allow_tool(tu.name)
            decision_label = "ask->allow"

        t0 = time.perf_counter()
        try:
            args = tool.parse_args(tu.input)
        except ValidationError as exc:
            result = ToolResult.error(f"Invalid arguments for {tu.name}: {exc}")
        else:
            try:
                result = await tool.run(args, self.ctx)
            except Exception as exc:  # noqa: BLE001 - never let a tool crash the loop
                self.log.warning("tool_crashed", tool=tu.name, error=repr(exc))
                result = ToolResult.error(f"Tool {tu.name} raised an exception: {exc!r}")
        duration = time.perf_counter() - t0

        self.console.tool_result(tu.name, result)
        trace.record_tool_call(
            name=tu.name, decision=decision_label, duration_s=duration, is_error=result.is_error
        )
        return ToolResultBlock(
            tool_use_id=tu.id, content=result.content, is_error=result.is_error
        )


def build_runtime(settings: Settings, console: Console) -> AgentRuntime:
    """Assemble a runtime from settings (provider, tools, policy, context)."""
    return AgentRuntime(
        provider=create_provider(settings),
        registry=default_registry(),
        policy=PermissionPolicy(settings),
        ctx=ToolContext(settings),
        console=console,
        settings=settings,
    )
