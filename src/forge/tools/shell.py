"""Shell execution tool.

Runs a command via the system shell inside the workspace with a hard timeout
and captured, truncated output. In Phase 1 this executes directly on the host
(gated by the permission policy); Phase 3 will route it through a sandbox.
"""

from __future__ import annotations

import asyncio
from typing import Any

from pydantic import BaseModel, Field

from forge.tools.base import Tool, ToolResult
from forge.tools.context import PathOutsideWorkspaceError, ToolContext

_OUTPUT_LIMIT = 30_000


def _truncate_middle(text: str, limit: int = _OUTPUT_LIMIT) -> str:
    if len(text) <= limit:
        return text
    head = text[: limit // 2]
    tail = text[-limit // 2 :]
    omitted = len(text) - len(head) - len(tail)
    return f"{head}\n[... {omitted} chars omitted ...]\n{tail}"


class ShellArgs(BaseModel):
    command: str = Field(description="Shell command to run (executed via /bin/sh).")
    timeout: int | None = Field(
        default=None, description="Timeout in seconds (defaults to the configured shell timeout)."
    )
    workdir: str | None = Field(
        default=None, description="Working directory relative to the workspace root."
    )


class Shell(Tool[ShellArgs]):
    name = "shell"
    description = (
        "Run a shell command in the workspace and return its stdout, stderr, and "
        "exit code. Use for tests, builds, git, and other CLI tools."
    )
    read_only = False
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

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(cwd),
            )
        except OSError as exc:
            return ToolResult.error(f"Failed to start command: {exc}")

        try:
            stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return ToolResult.error(f"Command timed out after {timeout}s: {command}")

        rc = proc.returncode or 0
        stdout = stdout_b.decode("utf-8", errors="replace")
        stderr = stderr_b.decode("utf-8", errors="replace")

        parts = [f"$ {command}"]
        if stdout:
            parts.append(stdout.rstrip("\n"))
        if stderr:
            parts.append("--- stderr ---\n" + stderr.rstrip("\n"))
        parts.append(f"[exit code: {rc}]")
        body = _truncate_middle("\n".join(parts))

        return ToolResult(content=body, is_error=rc != 0, metadata={"exit_code": rc})


def shell_tools() -> list[Tool[Any]]:
    return [Shell()]
