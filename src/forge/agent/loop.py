"""The agent runtime: a robust build -> call -> execute -> observe loop.

Deliberately more than ``while True: llm()``. The loop's job is to keep making
forward progress in the presence of every ordinary failure:

- **Bad model input** — arguments are validated against each tool's schema
  before execution, and a schema violation returns an error the model can fix.
- **Failing tools** — a crash, timeout, or non-zero exit becomes a structured
  error *result*, so the model sees what happened and can recover.
- **Refused actions** — the permission policy gates writes and destructive
  actions; a denial tells the model to change approach rather than retry.
- **Flaky providers** — transient 429s/5xx/timeouts are retried with backoff
  instead of ending the run.
- **A full context window** — old tool output is compacted away rather than
  aborting the task at the worst possible moment.
- **Runaway loops** — a hard iteration ceiling, always.

Everything the run did is recorded on a :class:`RunTrace` for the summary and
for ``--json``.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

from forge.agent.context import ContextManager, gather_environment
from forge.agent.prompts import build_system_prompt
from forge.agent.state import AgentState
from forge.config import Settings
from forge.logging import get_logger
from forge.observability.trace import RunTrace
from forge.permissions.policy import PermissionPolicy
from forge.providers.base import ModelProvider, ProviderError
from forge.providers.registry import create_provider
from forge.providers.types import ModelResponse, ToolResultBlock, ToolUseBlock
from forge.sandbox import create_sandbox
from forge.tools import ToolContext, ToolExecutor, ToolRegistry, default_registry
from forge.ui.console import Console, console_approver

# How many times the model may be asked to continue after being cut off by the
# output-token limit, before we accept the truncated answer and stop.
_MAX_CONTINUATIONS = 3

_CONTINUE_NUDGE = (
    "Your previous message was cut off by the output token limit. "
    "Continue from exactly where you stopped. Do not repeat what you already said."
)


class RunStatus:
    """Terminal states for a task. Only ``COMPLETED`` means the model finished."""

    COMPLETED = "completed"
    MAX_ITERATIONS = "max_iterations"
    REFUSED = "refused"
    ERROR = "error"
    ABORTED = "aborted"


@dataclass
class AgentResult:
    status: str
    final_text: str
    trace: RunTrace
    state: AgentState

    @property
    def ok(self) -> bool:
        return self.status == RunStatus.COMPLETED

    @property
    def exit_code(self) -> int:
        """Process exit status: 0 success, 130 interrupted, 1 anything else."""
        if self.ok:
            return 0
        return 130 if self.status == RunStatus.ABORTED else 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "ok": self.ok,
            "response": self.final_text,
            "iterations": self.state.iterations,
            "trace": self.trace.to_dict(),
        }


class AgentRuntime:
    def __init__(
        self,
        *,
        provider: ModelProvider,
        registry: ToolRegistry,
        executor: ToolExecutor,
        ctx: ToolContext,
        console: Console,
        settings: Settings,
    ) -> None:
        self.provider = provider
        self.registry = registry
        self.executor = executor
        self.ctx = ctx
        self.console = console
        self.settings = settings
        self.context = ContextManager(max_tokens=settings.max_context_tokens)
        self.log = get_logger("forge.agent")
        self._tools_schema: list[dict[str, Any]] | None = None
        self._system_prompt: str | None = None

    # ----------------------------------------------------------------- setup
    async def system_prompt(self) -> str:
        """Build the system prompt once per runtime, including workspace facts."""
        if self._system_prompt is None:
            environment = await gather_environment(self.ctx)
            self._system_prompt = build_system_prompt(self.registry, environment=environment)
        return self._system_prompt

    def tools_schema(self) -> list[dict[str, Any]]:
        if self._tools_schema is None:
            self._tools_schema = self.registry.to_provider_schema()
        return self._tools_schema

    async def new_state(self) -> AgentState:
        return AgentState(system_prompt=await self.system_prompt())

    # ------------------------------------------------------------------ loop
    async def run_task(self, task: str, state: AgentState | None = None) -> AgentResult:
        """Run one task to completion, reusing ``state`` to continue a session."""
        if state is None:
            state = await self.new_state()
        state.add_user(task)

        trace = RunTrace(task=task)
        budget = self.settings.max_iterations
        start_iteration = state.iterations
        continuations = 0
        status = RunStatus.MAX_ITERATIONS
        final_text = ""

        try:
            while state.iterations - start_iteration < budget:
                state.iterations += 1

                if self.context.over_budget(state):
                    reclaimed = self.context.compact(state)
                    trace.compactions += 1
                    self.console.info(f"Compacted context (~{reclaimed:,} tokens reclaimed).")
                    self.log.info("context_compacted", reclaimed_tokens=reclaimed)

                response = await self._complete(state, trace)
                state.add_message(response.to_message())
                state.record_usage(response.usage)

                if text := response.text().strip():
                    self.console.assistant_text(text)
                    final_text = text

                if response.stop_reason == "refusal":
                    status = RunStatus.REFUSED
                    final_text = final_text or "The model declined to continue with this task."
                    self.console.warning(final_text)
                    break

                tool_uses = response.tool_uses()
                if tool_uses:
                    results = await self._run_tools(tool_uses, trace)
                    state.add_tool_results(results)
                    continue

                # No tools requested. Either the model is done, or it ran out of
                # output tokens mid-sentence and deserves a chance to finish.
                if response.truncated and continuations < _MAX_CONTINUATIONS:
                    continuations += 1
                    self.console.info("Response hit the output limit; asking it to continue.")
                    state.add_user(_CONTINUE_NUDGE)
                    continue

                status = RunStatus.COMPLETED
                break
            else:
                self.console.warning(
                    f"Stopped after {budget} iterations without finishing. "
                    "Re-run with --max-iterations to allow more, or narrow the task."
                )
        except (KeyboardInterrupt, asyncio.CancelledError):
            status = RunStatus.ABORTED
            self.console.warning("Aborted.")
        except ProviderError as exc:
            status = RunStatus.ERROR
            final_text = str(exc)
            self.console.error(str(exc))

        trace.finish(status)
        self.console.run_summary(trace)
        self.log.info(
            "run_finished",
            status=status,
            iterations=state.iterations - start_iteration,
            tool_calls=trace.num_tool_calls,
        )
        return AgentResult(status=status, final_text=final_text, trace=trace, state=state)

    # --------------------------------------------------------------- helpers
    async def _complete(self, state: AgentState, trace: RunTrace) -> ModelResponse:
        """One model call, retrying transient provider failures with backoff."""
        attempts = max(1, self.settings.max_provider_retries)
        last_error: ProviderError | None = None

        for attempt in range(attempts):
            started = time.perf_counter()
            try:
                with self.console.status("thinking..."):
                    response = await self.provider.complete(
                        system=state.system_prompt,
                        messages=state.messages,
                        tools=self.tools_schema(),
                        max_tokens=self.settings.max_tokens,
                        temperature=self.settings.temperature,
                    )
            except ProviderError as exc:
                last_error = exc
                if not exc.retryable or attempt == attempts - 1:
                    raise
                delay = self.settings.retry_base_delay * (2**attempt)
                self.console.warning(
                    f"{exc} — retrying in {delay:.0f}s (attempt {attempt + 2}/{attempts})."
                )
                self.log.warning("provider_retry", attempt=attempt + 1, delay_s=delay)
                await asyncio.sleep(delay)
                continue

            latency = time.perf_counter() - started
            trace.record_model_call(
                model=response.model or self.provider.model,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                latency_s=latency,
                stop_reason=response.stop_reason,
                retries=attempt,
            )
            self.log.info(
                "model_call",
                latency_s=round(latency, 2),
                stop_reason=response.stop_reason,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            )
            return response

        # Unreachable: the final attempt either returns or re-raises.
        raise last_error or ProviderError("Model call failed with no response")

    async def _run_tools(
        self, tool_uses: list[ToolUseBlock], trace: RunTrace
    ) -> list[ToolResultBlock]:
        """Execute requested tools in order.

        Order is preserved deliberately: a batch is usually "edit, then run the
        tests", and running those concurrently would race.
        """
        blocks: list[ToolResultBlock] = []
        for tool_use in tool_uses:
            self.console.tool_call(tool_use.name, tool_use.input)
            outcome = await self.executor.execute(tool_use)
            self.console.tool_outcome(outcome)
            trace.record_tool_call(
                name=outcome.name,
                decision=outcome.decision,
                duration_s=outcome.duration_s,
                is_error=outcome.is_error,
                risk=outcome.risk.value,
            )
            blocks.append(outcome.block)
        return blocks

    async def aclose(self) -> None:
        await self.provider.aclose()
        await self.ctx.aclose()


def build_runtime(settings: Settings, console: Console) -> AgentRuntime:
    """Assemble a runtime from settings: provider, sandbox, tools, policy, UI."""
    registry = default_registry(settings)
    policy = PermissionPolicy(settings)
    ctx = ToolContext(settings, sandbox=create_sandbox(settings))
    executor = ToolExecutor(
        registry=registry,
        policy=policy,
        ctx=ctx,
        settings=settings,
        approver=console_approver(console),
    )
    return AgentRuntime(
        provider=create_provider(settings),
        registry=registry,
        executor=executor,
        ctx=ctx,
        console=console,
        settings=settings,
    )
