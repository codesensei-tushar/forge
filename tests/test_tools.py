"""Tests for the tool registry and the execution pipeline.

The pipeline is the funnel every model-requested action passes through::

    ToolUseBlock -> lookup -> permission -> [approval] -> validate -> execute

Its central contract is that *nothing* raises: every failure mode becomes an
error result the model can read and recover from.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, Field

from forge.config import ApprovalMode, Settings
from forge.permissions.policy import Approval, Decision, PermissionPolicy, PermissionResult
from forge.providers.types import ToolUseBlock
from forge.tools import ToolContext, ToolExecutor, default_registry, default_tools
from forge.tools.base import Risk, Tool, ToolResult, _strip_titles
from forge.tools.executor import deny_all
from forge.tools.registry import ToolNotFoundError, ToolRegistry


class EchoArgs(BaseModel):
    text: str = Field(description="What to echo back.")


class Echo(Tool[EchoArgs]):
    name = "echo"
    description = "Echo the given text."
    risk = Risk.READ
    Args = EchoArgs

    async def run(self, args: EchoArgs, ctx: ToolContext) -> ToolResult:
        return ToolResult.ok(args.text)


class Boom(Tool[EchoArgs]):
    name = "boom"
    description = "Raise an unexpected exception."
    risk = Risk.READ
    Args = EchoArgs

    async def run(self, args: EchoArgs, ctx: ToolContext) -> ToolResult:
        raise RuntimeError("internal explosion")


class SlowArgs(BaseModel):
    pass


class Slow(Tool[SlowArgs]):
    name = "slow"
    description = "Never finishes."
    risk = Risk.READ
    # A zero ceiling exercises the timeout path without making the test wait on a
    # real clock.
    timeout = 0
    Args = SlowArgs

    async def run(self, args: SlowArgs, ctx: ToolContext) -> ToolResult:
        await asyncio.sleep(60)
        return ToolResult.ok("unreachable")


def use(name: str, **args: Any) -> ToolUseBlock:
    return ToolUseBlock(id="call-1", name=name, input=args)


def make_executor(
    settings: Settings,
    ctx: ToolContext,
    *,
    tools: list[Tool[Any]] | None = None,
    approver: Any = None,
) -> ToolExecutor:
    registry = ToolRegistry(tools) if tools is not None else default_registry(settings)
    return ToolExecutor(
        registry=registry,
        policy=PermissionPolicy(settings),
        ctx=ctx,
        settings=settings,
        approver=approver,
    )


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
def test_register_and_lookup() -> None:
    registry = ToolRegistry([Echo()])
    assert "echo" in registry
    assert len(registry) == 1
    assert registry.get("echo") is not None
    assert registry.get("nope") is None
    assert [t.name for t in registry] == ["echo"]


def test_duplicate_registration_is_rejected() -> None:
    registry = ToolRegistry([Echo()])
    with pytest.raises(ValueError, match="already registered"):
        registry.register(Echo())


def test_replace_allows_overriding_a_tool() -> None:
    registry = ToolRegistry([Echo()])
    replacement = Echo()
    registry.register(replacement, replace=True)
    assert registry.get("echo") is replacement


def test_nameless_tool_is_rejected() -> None:
    class Nameless(Echo):
        name = ""

    with pytest.raises(ValueError, match="has no name"):
        ToolRegistry([Nameless()])


def test_unregister_returns_the_tool_then_none() -> None:
    registry = ToolRegistry([Echo()])
    assert registry.unregister("echo") is not None
    assert registry.unregister("echo") is None
    assert len(registry) == 0


def test_require_raises_with_the_available_names() -> None:
    registry = ToolRegistry([Echo()])
    assert registry.require("echo").name == "echo"
    with pytest.raises(ToolNotFoundError, match="echo"):
        registry.require("missing")


def test_names_are_sorted(registry: ToolRegistry) -> None:
    assert registry.names() == sorted(registry.names())


def test_by_risk_partitions_the_tool_set(registry: ToolRegistry) -> None:
    reads = {t.name for t in registry.by_risk(Risk.READ)}
    destructive = {t.name for t in registry.by_risk(Risk.DESTRUCTIVE)}
    assert {"read_file", "list_directory", "search_files", "git_status", "git_diff"} <= reads
    assert {"git_reset", "git_revert"} <= destructive
    assert reads.isdisjoint(destructive)


def test_provider_schema_shape(registry: ToolRegistry) -> None:
    schemas = registry.to_provider_schema()
    assert len(schemas) == len(registry)
    for schema in schemas:
        assert set(schema) == {"name", "description", "input_schema"}
        assert schema["description"]
        assert schema["input_schema"]["type"] == "object"


def test_schemas_carry_no_pydantic_titles(registry: ToolRegistry) -> None:
    """Titles are re-sent on every model call and say nothing new."""
    for schema in registry.to_provider_schema():
        body = schema["input_schema"]
        assert "title" not in body
        for prop in body.get("properties", {}).values():
            assert "title" not in prop


def test_strip_titles_recurses_into_unions() -> None:
    cleaned = _strip_titles(
        {
            "title": "Outer",
            "properties": {"a": {"title": "A", "anyOf": [{"title": "T", "type": "string"}]}},
        }
    )
    assert "title" not in cleaned
    assert "title" not in cleaned["properties"]["a"]
    assert "title" not in cleaned["properties"]["a"]["anyOf"][0]


def test_describe_lists_every_tool(registry: ToolRegistry) -> None:
    described = registry.describe()
    for name in registry.names():
        assert name in described


def test_git_tools_can_be_disabled() -> None:
    with_git = {t.name for t in default_tools(Settings(enable_git_tools=True))}
    without_git = {t.name for t in default_tools(Settings(enable_git_tools=False))}
    assert "git_status" in with_git
    assert not any(n.startswith("git_") for n in without_git)
    assert {"read_file", "shell"} <= without_git


def test_push_is_not_reachable_through_any_tool(registry: ToolRegistry) -> None:
    """Publishing is the operator's decision; no tool may perform it."""
    assert "git_push" not in registry
    assert not any("push" in name for name in registry.names())


def test_tool_repr_and_read_only_flag() -> None:
    assert Echo().read_only is True
    assert "echo" in repr(Echo())


# --------------------------------------------------------------------------- #
# Executor: happy path
# --------------------------------------------------------------------------- #
async def test_executor_runs_a_tool(settings: Settings, ctx: ToolContext) -> None:
    executor = make_executor(settings, ctx, tools=[Echo()])
    outcome = await executor.execute(use("echo", text="hi"))

    assert not outcome.is_error
    assert outcome.block.content == "hi"
    assert outcome.block.tool_use_id == "call-1"
    assert outcome.decision == "allow"
    assert outcome.risk is Risk.READ
    assert outcome.result is not None and not outcome.result.is_error


# --------------------------------------------------------------------------- #
# Executor: every failure becomes a recoverable result
# --------------------------------------------------------------------------- #
async def test_unknown_tool_lists_what_is_available(settings: Settings, ctx: ToolContext) -> None:
    executor = make_executor(settings, ctx, tools=[Echo()])
    outcome = await executor.execute(use("nonexistent"))

    assert outcome.is_error and outcome.decision == "unknown"
    assert "Unknown tool" in outcome.block.content
    assert "echo" in outcome.block.content


async def test_invalid_arguments_are_reported_field_by_field(
    settings: Settings, ctx: ToolContext
) -> None:
    executor = make_executor(settings, ctx, tools=[Echo()])
    outcome = await executor.execute(use("echo"))  # missing required `text`

    assert outcome.is_error
    assert "Invalid arguments for echo" in outcome.block.content
    assert "text" in outcome.block.content


async def test_a_crashing_tool_returns_an_error_result(
    settings: Settings, ctx: ToolContext
) -> None:
    executor = make_executor(settings, ctx, tools=[Boom()])
    outcome = await executor.execute(use("boom", text="x"))

    assert outcome.is_error
    assert "unexpected error" in outcome.block.content
    assert "internal explosion" in outcome.block.content


async def test_a_hanging_tool_is_timed_out(settings: Settings, ctx: ToolContext) -> None:
    executor = make_executor(settings, ctx, tools=[Slow()])
    outcome = await executor.execute(use("slow"))

    assert outcome.is_error
    assert "time limit" in outcome.block.content


async def test_cancellation_propagates(settings: Settings, ctx: ToolContext) -> None:
    """Ctrl-C must abort the run, not be swallowed as a tool error."""

    class Cancelling(Tool[SlowArgs]):
        name = "cancelling"
        description = "raises CancelledError"
        risk = Risk.READ
        Args = SlowArgs

        async def run(self, args: SlowArgs, ctx: ToolContext) -> ToolResult:
            raise asyncio.CancelledError

    executor = make_executor(settings, ctx, tools=[Cancelling()])
    with pytest.raises(asyncio.CancelledError):
        await executor.execute(use("cancelling"))


# --------------------------------------------------------------------------- #
# Executor: permission integration
# --------------------------------------------------------------------------- #
async def test_denied_call_tells_the_model_not_to_retry(
    settings: Settings, ctx: ToolContext
) -> None:
    outcome = await make_executor(settings, ctx).execute(use("shell", command="sudo rm -rf /"))

    assert outcome.is_error and outcome.decision == "deny"
    assert "Denied by policy" in outcome.block.content
    assert "Do not retry" in outcome.block.content


async def test_default_approver_denies_gated_calls(workspace: Path, ctx: ToolContext) -> None:
    """A non-interactive run must refuse rather than block on a human."""
    cautious = Settings(workspace_root=workspace, approval_mode=ApprovalMode.CAUTIOUS)
    outcome = await make_executor(cautious, ctx).execute(
        use("write_file", path="a.txt", content="x")
    )

    assert outcome.is_error and outcome.decision == "deny"
    assert "declined" in outcome.block.content


async def test_approval_lets_the_call_through(workspace: Path, ctx: ToolContext) -> None:
    cautious = Settings(workspace_root=workspace, approval_mode=ApprovalMode.CAUTIOUS)
    asked: list[str] = []

    async def approve(tool: Tool[Any], target: str, perm: PermissionResult) -> Approval:
        asked.append(target)
        return Approval.ALLOW

    outcome = await make_executor(cautious, ctx, approver=approve).execute(
        use("write_file", path="a.txt", content="x")
    )

    assert not outcome.is_error and outcome.decision == "ask->allow"
    assert asked == ["write_file a.txt"]
    assert (workspace / "a.txt").read_text() == "x"


async def test_always_grants_the_tool_for_the_session(workspace: Path, ctx: ToolContext) -> None:
    cautious = Settings(workspace_root=workspace, approval_mode=ApprovalMode.CAUTIOUS)
    prompts = {"n": 0}

    async def approve_always(tool: Tool[Any], target: str, perm: PermissionResult) -> Approval:
        prompts["n"] += 1
        return Approval.ALWAYS

    executor = make_executor(cautious, ctx, approver=approve_always)
    first = await executor.execute(use("write_file", path="a.txt", content="1"))
    second = await executor.execute(use("write_file", path="b.txt", content="2"))

    assert not first.is_error and not second.is_error
    assert prompts["n"] == 1, "the second call must not re-prompt"
    assert second.decision == "allow"


async def test_permission_is_checked_before_arguments_are_validated(
    settings: Settings, ctx: ToolContext
) -> None:
    """A denied call is refused on its own terms, not on a schema technicality."""
    outcome = await make_executor(settings, ctx).execute(
        use("shell", command="git push origin", timeout="not-an-int")
    )
    assert outcome.decision == "deny"
    assert "Denied by policy" in outcome.block.content
    assert "Invalid arguments" not in outcome.block.content


async def test_deny_all_is_the_default_approver(settings: Settings, ctx: ToolContext) -> None:
    executor = ToolExecutor(
        registry=default_registry(settings),
        policy=PermissionPolicy(settings),
        ctx=ctx,
        settings=settings,
    )
    assert executor.approver is deny_all
    refusal = PermissionResult(decision=Decision.ASK, reason="test", risk=Risk.DESTRUCTIVE)
    assert await deny_all(Echo(), "echo", refusal) is Approval.DENY


# --------------------------------------------------------------------------- #
# Timeout resolution
# --------------------------------------------------------------------------- #
def test_timeout_prefers_the_tools_own_ceiling(settings: Settings, ctx: ToolContext) -> None:
    executor = make_executor(settings, ctx)
    assert executor.timeout_for(Slow(), {}) == 0


def test_timeout_pads_a_model_requested_shell_timeout(settings: Settings, ctx: ToolContext) -> None:
    """The outer guard sits above the tool's own timeout so the tool reports first."""
    executor = make_executor(settings, ctx)
    shell = executor.registry.require("shell")
    assert executor.timeout_for(shell, {"timeout": 30}) == 45


def test_timeout_falls_back_to_the_global_setting(settings: Settings, ctx: ToolContext) -> None:
    executor = make_executor(settings, ctx)
    shell = executor.registry.require("shell")
    assert executor.timeout_for(shell, {}) == settings.tool_timeout
    assert executor.timeout_for(shell, {"timeout": -5}) == settings.tool_timeout


# --------------------------------------------------------------------------- #
# ToolResult
# --------------------------------------------------------------------------- #
def test_tool_result_constructors_carry_metadata() -> None:
    ok = ToolResult.ok("fine", exit_code=0)
    bad = ToolResult.error("broken", exit_code=1)
    assert not ok.is_error and ok.metadata == {"exit_code": 0}
    assert bad.is_error and bad.metadata == {"exit_code": 1}
