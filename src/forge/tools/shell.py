"""Shell execution tool.

Commands run through the context's :class:`~forge.sandbox.base.Sandbox` rather
than via a direct subprocess call, so switching to container isolation is a
configuration change and not a rewrite of this file. Output is captured, middle-
truncated to keep the context window usable, and a non-zero exit is reported as
an error *result* — never an exception — so the model can read the failure and
decide what to do about it.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from forge.tools.base import Risk, Tool, ToolResult
from forge.tools.context import PathOutsideWorkspaceError, ToolContext

_OUTPUT_LIMIT = 30_000


def truncate_middle(text: str, limit: int = _OUTPUT_LIMIT) -> str:
    """Keep the head and tail of long output; the middle is rarely the signal."""
    if len(text) <= limit:
        return text
    half = limit // 2
    head, tail = text[:half], text[-half:]
    omitted = len(text) - len(head) - len(tail)
    return f"{head}\n[... {omitted} chars omitted ...]\n{tail}"


def format_exec_result(command: str, stdout: str, stderr: str, exit_code: int) -> str:
    """Render a command's outcome the way a human would read it in a terminal."""
    parts = [f"$ {command}"]
    if stdout.strip():
        parts.append(stdout.rstrip("\n"))
    if stderr.strip():
        parts.append("--- stderr ---\n" + stderr.rstrip("\n"))
    if not stdout.strip() and not stderr.strip():
        parts.append("(no output)")
    parts.append(f"[exit code: {exit_code}]")
    return truncate_middle("\n".join(parts))


class ShellArgs(BaseModel):
    command: str = Field(description="Shell command to run (executed via /bin/sh).")
    timeout: int | None = Field(
        default=None,
        description="Timeout in seconds (defaults to the configured shell timeout).",
    )
    workdir: str | None = Field(
        default=None, description="Working directory relative to the workspace root."
    )


class Shell(Tool[ShellArgs]):
    name = "shell"
    description = (
        "Run a shell command in the workspace and return its stdout, stderr, and "
        "exit code. Use for tests, builds, linters, and other CLI tools."
    )
    risk = Risk.WRITE
    Args = ShellArgs

    async def run(self, args: ShellArgs, ctx: ToolContext) -> ToolResult:
        command = args.command.strip()
        if not command:
            return ToolResult.error("Empty command")

        cwd = ctx.workspace_root
        if args.workdir:
            try:
                cwd = ctx.resolve_path(args.workdir)
            except PathOutsideWorkspaceError as exc:
                return ToolResult.error(str(exc))
            if not cwd.is_dir():
                return ToolResult.error(f"workdir is not a directory: {args.workdir}")

        timeout = args.timeout or ctx.settings.shell_timeout
        outcome = await ctx.exec(command, cwd=cwd, timeout=timeout)

        if outcome.startup_error:
            return ToolResult.error(outcome.startup_error, exit_code=outcome.exit_code)
        if outcome.timed_out:
            return ToolResult.error(
                f"Command timed out after {timeout}s and was killed: {command}\n"
                "Consider a narrower command or a longer timeout.",
                exit_code=outcome.exit_code,
                timed_out=True,
            )

        return ToolResult(
            content=format_exec_result(command, outcome.stdout, outcome.stderr, outcome.exit_code),
            is_error=outcome.exit_code != 0,
            metadata={"exit_code": outcome.exit_code, "duration_s": outcome.duration_s},
        )


def shell_tools() -> list[Tool[Any]]:
    return [Shell()]
