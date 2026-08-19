"""Host-local sandbox: a plain subprocess with a timeout and a scrubbed env.

This is the default backend. It provides no CPU/memory isolation — the guard
rails at this level are the workspace path boundary, the permission policy, the
command timeout, and the fact that credentials are stripped from the child
environment. Use the Docker backend when the agent's code is untrusted.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import time

from forge.sandbox.base import ExecRequest, ExecResult, Sandbox, scrub_env

# Forced into every child environment so captured output is parseable rather than
# terminal-shaped.
_MACHINE_READABLE = {
    "TERM": "dumb",
    "GIT_PAGER": "cat",
    "PAGER": "cat",
    "NO_COLOR": "1",
}


class LocalSandbox(Sandbox):
    name = "local"

    def __init__(self, *, scrub_secrets: bool = True) -> None:
        self.scrub_secrets = scrub_secrets

    def describe(self) -> str:
        detail = "credentials stripped" if self.scrub_secrets else "full environment"
        return f"local host ({detail}, no resource limits)"

    def _env_for(self, request: ExecRequest) -> dict[str, str]:
        base = scrub_env() if self.scrub_secrets else dict(os.environ)
        # Keep tool output machine-readable: no pagers, no ANSI colour codes.
        # These override the host's own values (an interactive TERM would defeat
        # the point), but an explicit request env still wins over both.
        base.update(_MACHINE_READABLE)
        base.update(request.env)
        return base

    async def exec(self, request: ExecRequest) -> ExecResult:
        started = time.perf_counter()
        try:
            proc = await asyncio.create_subprocess_shell(
                request.command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                stdin=asyncio.subprocess.DEVNULL,
                cwd=str(request.cwd),
                env=self._env_for(request),
                start_new_session=True,
            )
        except OSError as exc:
            return ExecResult(
                exit_code=-1,
                stdout="",
                stderr="",
                startup_error=f"Failed to start command: {exc}",
                duration_s=time.perf_counter() - started,
            )

        try:
            stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), request.timeout)
        except TimeoutError:
            await terminate_process_group(proc)
            return ExecResult(
                exit_code=124,
                stdout="",
                stderr="",
                timed_out=True,
                duration_s=time.perf_counter() - started,
            )

        return ExecResult(
            exit_code=proc.returncode or 0,
            stdout=stdout_b.decode("utf-8", errors="replace"),
            stderr=stderr_b.decode("utf-8", errors="replace"),
            duration_s=time.perf_counter() - started,
        )


async def terminate_process_group(proc: asyncio.subprocess.Process) -> None:
    """Kill the whole process group, then reap it.

    ``start_new_session=True`` puts the child in its own group, so a shell that
    spawned background children does not leave orphans behind on timeout.
    """
    with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    with contextlib.suppress(ProcessLookupError):
        proc.kill()
    await proc.wait()
