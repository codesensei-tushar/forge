"""Tests for the shell tool."""

from __future__ import annotations

from forge.tools.context import ToolContext
from forge.tools.shell import Shell, ShellArgs


async def test_shell_echo(ctx: ToolContext) -> None:
    res = await Shell().run(ShellArgs(command="echo hello"), ctx)
    assert not res.is_error
    assert "hello" in res.content
    assert "[exit code: 0]" in res.content


async def test_shell_nonzero_exit_is_error(ctx: ToolContext) -> None:
    res = await Shell().run(ShellArgs(command="exit 3"), ctx)
    assert res.is_error
    assert res.metadata["exit_code"] == 3


async def test_shell_timeout(ctx: ToolContext) -> None:
    res = await Shell().run(ShellArgs(command="sleep 3", timeout=1), ctx)
    assert res.is_error
    assert "timed out" in res.content


async def test_shell_runs_in_workspace(ctx: ToolContext) -> None:
    res = await Shell().run(ShellArgs(command="pwd"), ctx)
    assert not res.is_error
    assert ctx.workspace_root.name in res.content
