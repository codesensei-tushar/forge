"""Tests for the agent runtime loop using an offline FakeProvider."""

from __future__ import annotations

from pathlib import Path

from forge.agent.loop import AgentRuntime
from forge.config import Settings
from forge.models.base import FakeProvider
from forge.models.types import ModelResponse, TextBlock, ToolUseBlock, Usage
from forge.permissions.policy import PermissionPolicy
from forge.tools import default_registry
from forge.tools.context import ToolContext
from forge.ui.console import Console


def make_runtime(settings: Settings, provider: FakeProvider) -> AgentRuntime:
    return AgentRuntime(
        provider=provider,
        registry=default_registry(),
        policy=PermissionPolicy(settings),
        ctx=ToolContext(settings),
        console=Console(),
        settings=settings,
    )


def _tool_use(name: str, **inp: object) -> ModelResponse:
    return ModelResponse(
        blocks=[ToolUseBlock(id="t1", name=name, input=dict(inp))],
        stop_reason="tool_use",
        usage=Usage(input_tokens=1, output_tokens=1),
    )


def _final(text: str) -> ModelResponse:
    return ModelResponse(
        blocks=[TextBlock(text=text)],
        stop_reason="end_turn",
        usage=Usage(input_tokens=1, output_tokens=1),
    )


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
    assert result.status == "completed"
    assert "hi there" in result.final_text
    assert result.trace.num_model_calls == 2
    assert result.trace.num_tool_calls == 1
    assert result.trace.num_tool_errors == 0
    assert result.trace.total_tokens == 29


async def test_loop_respects_max_iterations(settings: Settings) -> None:
    s = settings.model_copy(update={"max_iterations": 3})
    provider = FakeProvider(responder=lambda _msgs: _tool_use("list_directory", path="."))
    result = await make_runtime(s, provider).run_task("keep going")
    assert result.status == "max_iterations"
    assert result.trace.num_model_calls == 3
    assert result.trace.num_tool_calls == 3


async def test_loop_denied_tool_feeds_error_and_recovers(settings: Settings) -> None:
    provider = FakeProvider(
        script=[_tool_use("shell", command="rm -rf /"), _final("Understood, I won't.")]
    )
    result = await make_runtime(settings, provider).run_task("delete everything")
    assert result.status == "completed"
    assert result.trace.num_tool_calls == 1
    tc = result.trace.tool_calls[0]
    assert tc.decision == "deny"
    assert tc.is_error


async def test_loop_handles_unknown_tool(settings: Settings) -> None:
    provider = FakeProvider(script=[_tool_use("does_not_exist"), _final("done")])
    result = await make_runtime(settings, provider).run_task("x")
    assert result.status == "completed"
    assert result.trace.tool_calls[0].is_error


async def test_loop_reports_provider_error(settings: Settings) -> None:
    from forge.models.base import ProviderError

    def boom(_msgs: object) -> ModelResponse:
        raise ProviderError("gateway exploded")

    provider = FakeProvider(responder=boom)
    result = await make_runtime(settings, provider).run_task("x")
    assert result.status == "error"
    assert "exploded" in result.final_text


async def test_state_persists_across_tasks(settings: Settings) -> None:
    provider = FakeProvider(script=[_final("first"), _final("second")])
    runtime = make_runtime(settings, provider)
    r1 = await runtime.run_task("task one")
    r2 = await runtime.run_task("task two", r1.state)
    roles = [m.role for m in r2.state.messages]
    assert roles.count("user") >= 2
    assert r2.state.messages[0].text() == "task one"
    assert r2.status == "completed"
