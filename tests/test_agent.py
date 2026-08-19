"""Tests for the agent runtime loop, driven entirely by the offline FakeProvider.

These are the tests that matter most: they pin down the loop's robustness
contract — it terminates, it never crashes on a bad tool call, it feeds failures
back to the model, and it recovers from transient provider errors.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from forge.agent.loop import AgentRuntime, RunStatus
from forge.config import ApprovalMode, Settings
from forge.permissions.policy import PermissionPolicy
from forge.providers.base import FakeProvider, ProviderError
from forge.providers.types import (
    Message,
    ModelResponse,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    Usage,
)
from forge.tools import ToolExecutor, default_registry
from forge.tools.context import ToolContext
from forge.ui.console import Console


def make_runtime(
    settings: Settings, provider: FakeProvider, *, approver: object = None
) -> AgentRuntime:
    registry = default_registry(settings)
    ctx = ToolContext(settings)
    executor = ToolExecutor(
        registry=registry,
        policy=PermissionPolicy(settings),
        ctx=ctx,
        settings=settings,
        approver=approver,  # type: ignore[arg-type]
    )
    return AgentRuntime(
        provider=provider,
        registry=registry,
        executor=executor,
        ctx=ctx,
        console=Console(quiet=True),
        settings=settings,
    )


def tool_use(name: str, _id: str = "t1", **inp: object) -> ModelResponse:
    return ModelResponse(
        blocks=[ToolUseBlock(id=_id, name=name, input=dict(inp))],
        stop_reason="tool_use",
        usage=Usage(input_tokens=1, output_tokens=1),
    )


def final(text: str, *, stop_reason: str = "end_turn") -> ModelResponse:
    return ModelResponse(
        blocks=[TextBlock(text=text)],
        stop_reason=stop_reason,  # type: ignore[arg-type]
        usage=Usage(input_tokens=1, output_tokens=1),
    )


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #
async def test_loop_executes_tool_then_finishes(settings: Settings, workspace: Path) -> None:
    (workspace / "hello.txt").write_text("hi there")
    provider = FakeProvider(
        script=[
            ModelResponse(
                blocks=[ToolUseBlock(id="t1", name="read_file", input={"path": "hello.txt"})],
                stop_reason="tool_use",
                usage=Usage(input_tokens=10, output_tokens=5),
            ),
            ModelResponse(
                blocks=[TextBlock(text="The file says hi there.")],
                stop_reason="end_turn",
                usage=Usage(input_tokens=8, output_tokens=6),
            ),
        ]
    )
    result = await make_runtime(settings, provider).run_task("read hello.txt")

    assert result.status == RunStatus.COMPLETED
    assert result.ok and result.exit_code == 0
    assert "hi there" in result.final_text
    assert result.trace.num_model_calls == 2
    assert result.trace.num_tool_calls == 1
    assert result.trace.num_tool_errors == 0
    assert result.trace.total_tokens == 29


async def test_tool_result_is_fed_back_to_the_model(settings: Settings, workspace: Path) -> None:
    (workspace / "data.txt").write_text("the-secret-value")
    provider = FakeProvider(script=[tool_use("read_file", path="data.txt"), final("done")])
    await make_runtime(settings, provider).run_task("read it")

    # The second model call must have seen the tool's output in its history.
    second_call = provider.calls[1]
    results = [b for m in second_call for b in m.content if isinstance(b, ToolResultBlock)]
    assert any("the-secret-value" in b.content for b in results)


async def test_multiple_tool_uses_run_in_order(settings: Settings, workspace: Path) -> None:
    provider = FakeProvider(
        script=[
            ModelResponse(
                blocks=[
                    ToolUseBlock(
                        id="a", name="write_file", input={"path": "x.txt", "content": "1"}
                    ),
                    ToolUseBlock(id="b", name="read_file", input={"path": "x.txt"}),
                ],
                stop_reason="tool_use",
            ),
            final("both done"),
        ]
    )
    result = await make_runtime(settings, provider).run_task("write then read")

    assert result.trace.num_tool_calls == 2
    assert [c.name for c in result.trace.tool_calls] == ["write_file", "read_file"]
    # The read observed the write, which only holds if execution is ordered.
    assert (workspace / "x.txt").read_text() == "1"


# --------------------------------------------------------------------------- #
# Guard rails
# --------------------------------------------------------------------------- #
async def test_loop_respects_max_iterations(settings: Settings) -> None:
    s = settings.model_copy(update={"max_iterations": 3})
    provider = FakeProvider(responder=lambda _m: tool_use("list_directory", path="."))
    result = await make_runtime(s, provider).run_task("keep going forever")

    assert result.status == RunStatus.MAX_ITERATIONS
    assert result.exit_code == 1
    assert result.trace.num_model_calls == 3
    assert result.trace.num_tool_calls == 3


async def test_denied_tool_feeds_error_and_model_recovers(settings: Settings) -> None:
    provider = FakeProvider(
        script=[tool_use("shell", command="rm -rf /"), final("Understood, I won't.")]
    )
    result = await make_runtime(settings, provider).run_task("delete everything")

    assert result.status == RunStatus.COMPLETED
    call = result.trace.tool_calls[0]
    assert call.decision == "deny" and call.is_error
    assert "Understood" in result.final_text


async def test_gated_call_is_refused_without_an_approver(settings: Settings) -> None:
    """Non-interactive runs must refuse rather than hang waiting for input."""
    s = settings.model_copy(update={"approval_mode": ApprovalMode.CAUTIOUS})
    provider = FakeProvider(
        script=[tool_use("write_file", path="new.txt", content="x"), final("refused, ok")]
    )
    result = await make_runtime(s, provider).run_task("write a file")

    assert result.trace.tool_calls[0].is_error
    assert result.status == RunStatus.COMPLETED


async def test_gated_call_proceeds_when_approved(
    settings: Settings, workspace: Path, approve_all: object
) -> None:
    s = settings.model_copy(update={"approval_mode": ApprovalMode.CAUTIOUS})
    provider = FakeProvider(
        script=[tool_use("write_file", path="ok.txt", content="yes"), final("written")]
    )
    result = await make_runtime(s, provider, approver=approve_all).run_task("write a file")

    assert result.trace.tool_calls[0].decision == "ask->allow"
    assert (workspace / "ok.txt").read_text() == "yes"


async def test_unknown_tool_becomes_a_recoverable_error(settings: Settings) -> None:
    provider = FakeProvider(script=[tool_use("does_not_exist"), final("done")])
    result = await make_runtime(settings, provider).run_task("x")

    assert result.status == RunStatus.COMPLETED
    assert result.trace.tool_calls[0].is_error
    assert result.trace.tool_calls[0].decision == "unknown"


async def test_invalid_tool_arguments_become_a_recoverable_error(settings: Settings) -> None:
    provider = FakeProvider(script=[tool_use("read_file"), final("I'll add the path")])
    result = await make_runtime(settings, provider).run_task("read something")

    assert result.status == RunStatus.COMPLETED
    assert result.trace.tool_calls[0].is_error


async def test_crashing_tool_does_not_kill_the_run(settings: Settings) -> None:
    from pydantic import BaseModel

    from forge.tools.base import Risk, Tool, ToolResult

    class BoomArgs(BaseModel):
        pass

    class Boom(Tool[BoomArgs]):
        name = "boom"
        description = "always explodes"
        risk = Risk.READ
        Args = BoomArgs

        async def run(self, args: BoomArgs, ctx: object) -> ToolResult:
            raise RuntimeError("kaboom")

    registry = default_registry(settings)
    registry.register(Boom())
    ctx = ToolContext(settings)
    runtime = AgentRuntime(
        provider=FakeProvider(script=[tool_use("boom"), final("recovered")]),
        registry=registry,
        executor=ToolExecutor(
            registry=registry, policy=PermissionPolicy(settings), ctx=ctx, settings=settings
        ),
        ctx=ctx,
        console=Console(quiet=True),
        settings=settings,
    )
    result = await runtime.run_task("explode")

    assert result.status == RunStatus.COMPLETED
    assert result.trace.tool_calls[0].is_error


# --------------------------------------------------------------------------- #
# Provider failures
# --------------------------------------------------------------------------- #
async def test_fatal_provider_error_ends_the_run(settings: Settings) -> None:
    def boom(_msgs: object) -> ModelResponse:
        raise ProviderError("bad credentials", retryable=False)

    result = await make_runtime(settings, FakeProvider(responder=boom)).run_task("x")

    assert result.status == RunStatus.ERROR
    assert "bad credentials" in result.final_text
    assert result.exit_code == 1


async def test_transient_provider_error_is_retried(settings: Settings) -> None:
    s = settings.model_copy(update={"max_provider_retries": 3, "retry_base_delay": 0.0})
    attempts = {"n": 0}

    def flaky(_msgs: object) -> ModelResponse:
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise ProviderError("overloaded", retryable=True, status=529)
        return final("recovered after retries")

    result = await make_runtime(s, FakeProvider(responder=flaky)).run_task("x")

    assert result.status == RunStatus.COMPLETED
    assert attempts["n"] == 3
    assert result.trace.model_calls[0].retries == 2


async def test_retries_give_up_after_the_limit(settings: Settings) -> None:
    s = settings.model_copy(update={"max_provider_retries": 2, "retry_base_delay": 0.0})

    def always_flaky(_msgs: object) -> ModelResponse:
        raise ProviderError("still overloaded", retryable=True, status=503)

    result = await make_runtime(s, FakeProvider(responder=always_flaky)).run_task("x")

    assert result.status == RunStatus.ERROR


async def test_refusal_stops_the_loop(settings: Settings) -> None:
    provider = FakeProvider(script=[final("I won't do that.", stop_reason="refusal")])
    result = await make_runtime(settings, provider).run_task("something disallowed")

    assert result.status == RunStatus.REFUSED
    assert result.exit_code == 1


async def test_truncated_response_is_continued(settings: Settings) -> None:
    provider = FakeProvider(
        script=[final("Here is the first ha", stop_reason="max_tokens"), final("...lf. Done.")]
    )
    result = await make_runtime(settings, provider).run_task("write a long answer")

    assert result.status == RunStatus.COMPLETED
    assert result.trace.num_model_calls == 2


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
async def test_state_persists_across_tasks(settings: Settings) -> None:
    provider = FakeProvider(script=[final("first"), final("second")])
    runtime = make_runtime(settings, provider)

    r1 = await runtime.run_task("task one")
    r2 = await runtime.run_task("task two", r1.state)

    assert r2.state.messages[0].text() == "task one"
    assert [m.role for m in r2.state.messages].count("user") >= 2
    assert r2.status == RunStatus.COMPLETED


async def test_usage_accumulates_on_state(settings: Settings) -> None:
    provider = FakeProvider(script=[final("a"), final("b")])
    runtime = make_runtime(settings, provider)
    r1 = await runtime.run_task("one")
    r2 = await runtime.run_task("two", r1.state)

    assert r2.state.usage.total_tokens == 4


async def test_system_prompt_includes_workspace_and_tools(settings: Settings) -> None:
    runtime = make_runtime(settings, FakeProvider())
    prompt = await runtime.system_prompt()

    assert str(settings.workspace_root) in prompt
    assert "read_file" in prompt and "shell" in prompt


async def test_result_serializes_to_json(settings: Settings) -> None:
    import json

    provider = FakeProvider(script=[final("all done")])
    result = await make_runtime(settings, provider).run_task("do it")
    payload = json.loads(json.dumps(result.to_dict()))

    assert payload["status"] == "completed"
    assert payload["ok"] is True
    assert payload["response"] == "all done"
    assert payload["trace"]["counts"]["model_calls"] == 1


@pytest.mark.parametrize(
    ("status", "code"),
    [
        (RunStatus.COMPLETED, 0),
        (RunStatus.ABORTED, 130),
        (RunStatus.ERROR, 1),
        (RunStatus.MAX_ITERATIONS, 1),
    ],
)
def test_exit_codes(status: str, code: int) -> None:
    from forge.agent.loop import AgentResult
    from forge.agent.state import AgentState
    from forge.observability.trace import RunTrace

    result = AgentResult(
        status=status, final_text="", trace=RunTrace(), state=AgentState(system_prompt="")
    )
    assert result.exit_code == code


def test_message_helpers_roundtrip() -> None:
    response = ModelResponse(
        blocks=[TextBlock(text="hi"), ToolUseBlock(id="1", name="read_file")],
        stop_reason="tool_use",
    )
    message = response.to_message()

    assert isinstance(message, Message)
    assert message.role == "assistant"
    assert message.text() == "hi"
    assert [b.name for b in message.tool_uses()] == ["read_file"]
