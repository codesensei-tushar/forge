"""Tests for the shell tool.

The shell is the riskiest tool in the set, so its contract is narrow: a non-zero
exit is a *result*, a hang is killed, and secrets never reach the child process.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from forge.tools.context import ToolContext
from forge.tools.shell import Shell, ShellArgs, format_exec_result, shell_tools, truncate_middle


async def shell(ctx: ToolContext, command: str, **kwargs: object) -> object:
    return await Shell().run(ShellArgs(command=command, **kwargs), ctx)  # type: ignore[arg-type]


async def test_command_output_is_captured(ctx: ToolContext) -> None:
    result = await shell(ctx, "echo hello")
    assert not result.is_error  # type: ignore[attr-defined]
    assert "hello" in result.content  # type: ignore[attr-defined]
    assert "[exit code: 0]" in result.content  # type: ignore[attr-defined]


async def test_stderr_is_labelled(ctx: ToolContext) -> None:
    result = await shell(ctx, "echo oops >&2")
    assert "--- stderr ---" in result.content  # type: ignore[attr-defined]
    assert "oops" in result.content  # type: ignore[attr-defined]


async def test_non_zero_exit_is_an_error_result_not_an_exception(ctx: ToolContext) -> None:
    """The model has to be able to see and react to a failing test run."""
    result = await shell(ctx, "exit 3")
    assert result.is_error  # type: ignore[attr-defined]
    assert "[exit code: 3]" in result.content  # type: ignore[attr-defined]
    assert result.metadata["exit_code"] == 3  # type: ignore[attr-defined]


async def test_silent_command_says_so(ctx: ToolContext) -> None:
    result = await shell(ctx, "true")
    assert "(no output)" in result.content  # type: ignore[attr-defined]


async def test_empty_command_is_rejected(ctx: ToolContext) -> None:
    result = await shell(ctx, "   ")
    assert result.is_error and "Empty command" in result.content  # type: ignore[attr-defined]


async def test_runs_in_the_workspace_by_default(workspace: Path, ctx: ToolContext) -> None:
    (workspace / "marker.txt").write_text("x")
    result = await shell(ctx, "ls")
    assert "marker.txt" in result.content  # type: ignore[attr-defined]


async def test_workdir_narrows_the_cwd(workspace: Path, ctx: ToolContext) -> None:
    (workspace / "sub").mkdir()
    (workspace / "sub" / "inner.txt").write_text("x")
    result = await shell(ctx, "ls", workdir="sub")
    assert "inner.txt" in result.content  # type: ignore[attr-defined]


async def test_workdir_cannot_escape_the_workspace(ctx: ToolContext) -> None:
    result = await shell(ctx, "ls", workdir="..")
    assert result.is_error and "outside the workspace" in result.content  # type: ignore[attr-defined]


async def test_workdir_must_be_a_directory(workspace: Path, ctx: ToolContext) -> None:
    (workspace / "file.txt").write_text("x")
    result = await shell(ctx, "ls", workdir="file.txt")
    assert result.is_error and "not a directory" in result.content  # type: ignore[attr-defined]


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")
async def test_a_hanging_command_is_killed(ctx: ToolContext) -> None:
    result = await shell(ctx, "sleep 30", timeout=1)
    assert result.is_error  # type: ignore[attr-defined]
    assert "timed out after 1s" in result.content  # type: ignore[attr-defined]
    assert result.metadata["timed_out"] is True  # type: ignore[attr-defined]


async def test_startup_failure_is_reported(
    ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    from forge.sandbox import base as sandbox_base

    async def failing_exec(request: sandbox_base.ExecRequest) -> sandbox_base.ExecResult:
        return sandbox_base.ExecResult(
            exit_code=-1, stdout="", stderr="", startup_error="Failed to start command: boom"
        )

    monkeypatch.setattr(ctx.sandbox, "exec", failing_exec)
    result = await shell(ctx, "anything")
    assert result.is_error and "Failed to start command" in result.content  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- #
# Credential hygiene
# --------------------------------------------------------------------------- #
async def test_secrets_are_stripped_from_the_child_environment(
    ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A prompt injection that shells out must not be able to read the API key."""
    monkeypatch.setenv("MY_SECRET_TOKEN", "leak-me")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "leak-me-too")
    monkeypatch.setenv("FORGE_TEST_PLAIN", "harmless")

    result = await shell(ctx, "env")
    body = result.content  # type: ignore[attr-defined]

    assert "MY_SECRET_TOKEN" not in body
    assert "ANTHROPIC_API_KEY" not in body
    assert "FORGE_TEST_PLAIN" in body, "ordinary variables must still be inherited"


async def test_output_is_machine_readable(ctx: ToolContext) -> None:
    """No pager, no colour: the model reads this, not a terminal."""
    result = await shell(ctx, "echo $TERM $NO_COLOR")
    assert "dumb 1" in result.content  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #
def test_truncate_middle_keeps_both_ends() -> None:
    text = "A" * 100 + "B" * 100
    out = truncate_middle(text, limit=40)
    assert out.startswith("A" * 20)
    assert out.endswith("B" * 20)
    assert "chars omitted" in out


def test_truncate_middle_leaves_short_output_alone() -> None:
    assert truncate_middle("short", limit=100) == "short"


def test_format_exec_result_reads_like_a_terminal() -> None:
    rendered = format_exec_result("pytest -q", "1 passed\n", "", 0)
    assert rendered.splitlines() == ["$ pytest -q", "1 passed", "[exit code: 0]"]


def test_shell_tools_exposes_only_the_shell() -> None:
    assert [t.name for t in shell_tools()] == ["shell"]
