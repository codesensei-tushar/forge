"""Docker sandbox: run each command in a disposable, resource-capped container.

The workspace is bind-mounted at ``/workspace`` so edits are visible on the
host, but CPU, memory, and process count are capped and networking is off by
default. Each command gets its own ``docker run --rm`` container: there is no
state to leak between commands, and a wedged command cannot outlive its timeout.

The trade-off is startup latency (~0.3-1s per command) and that anything the
command installs does not persist. Pass a prepared image via ``sandbox_image``
when the task needs a toolchain.
"""

from __future__ import annotations

import asyncio
import shutil
import time
from pathlib import Path

from forge.sandbox.base import ExecRequest, ExecResult, Sandbox, SandboxError

_CONTAINER_WORKSPACE = "/workspace"


def docker_available() -> bool:
    """True when a usable ``docker`` binary is on PATH."""
    return shutil.which("docker") is not None


class DockerSandbox(Sandbox):
    name = "docker"

    def __init__(
        self,
        *,
        workspace_root: Path,
        image: str = "python:3.12-slim",
        cpus: float = 2.0,
        memory: str = "2g",
        network: bool = False,
        pids_limit: int = 512,
    ) -> None:
        if not docker_available():
            raise SandboxError(
                "Docker sandbox requested but the 'docker' binary was not found on PATH. "
                'Install Docker or set sandbox = "none".'
            )
        self.workspace_root = workspace_root
        self.image = image
        self.cpus = cpus
        self.memory = memory
        self.network = network
        self.pids_limit = pids_limit

    def describe(self) -> str:
        net = "enabled" if self.network else "disabled"
        return (
            f"docker {self.image} (cpus={self.cpus}, memory={self.memory}, "
            f"pids={self.pids_limit}, network={net})"
        )

    def _container_cwd(self, cwd: Path) -> str:
        """Map a host path inside the workspace to its container path."""
        try:
            relative = cwd.resolve().relative_to(self.workspace_root)
        except ValueError:
            return _CONTAINER_WORKSPACE
        return (
            _CONTAINER_WORKSPACE if str(relative) == "." else f"{_CONTAINER_WORKSPACE}/{relative}"
        )

    def _docker_argv(self, request: ExecRequest) -> list[str]:
        argv = [
            "docker",
            "run",
            "--rm",
            "--interactive=false",
            f"--volume={self.workspace_root}:{_CONTAINER_WORKSPACE}",
            f"--workdir={self._container_cwd(request.cwd)}",
            f"--cpus={self.cpus}",
            f"--memory={self.memory}",
            f"--pids-limit={self.pids_limit}",
            # Dropping privileges keeps a container escape from being a root escape.
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
        ]
        if not self.network:
            argv.append("--network=none")
        for key, value in request.env.items():
            argv.append(f"--env={key}={value}")
        argv += [
            "--env=TERM=dumb",
            "--env=NO_COLOR=1",
            "--env=GIT_PAGER=cat",
            self.image,
            "/bin/sh",
            "-c",
            request.command,
        ]
        return argv

    async def exec(self, request: ExecRequest) -> ExecResult:
        started = time.perf_counter()
        try:
            proc = await asyncio.create_subprocess_exec(
                *self._docker_argv(request),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                stdin=asyncio.subprocess.DEVNULL,
            )
        except OSError as exc:
            return ExecResult(
                exit_code=-1,
                stdout="",
                stderr="",
                startup_error=f"Failed to start docker: {exc}",
                duration_s=time.perf_counter() - started,
            )

        try:
            stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), request.timeout)
        except TimeoutError:
            proc.kill()
            await proc.wait()
            return ExecResult(
                exit_code=124,
                stdout="",
                stderr="",
                timed_out=True,
                duration_s=time.perf_counter() - started,
            )

        stderr = stderr_b.decode("utf-8", errors="replace")
        exit_code = proc.returncode or 0
        # Docker uses 125 for "the daemon/CLI itself failed", which is a sandbox
        # fault rather than a failure of the user's command. Surface it as such.
        if exit_code == 125:
            return ExecResult(
                exit_code=exit_code,
                stdout=stdout_b.decode("utf-8", errors="replace"),
                stderr=stderr,
                startup_error=f"docker could not run the container: {stderr.strip()[:400]}",
                duration_s=time.perf_counter() - started,
            )

        return ExecResult(
            exit_code=exit_code,
            stdout=stdout_b.decode("utf-8", errors="replace"),
            stderr=stderr,
            duration_s=time.perf_counter() - started,
        )
