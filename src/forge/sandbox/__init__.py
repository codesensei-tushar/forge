"""Sandbox selection: turn settings into a live :class:`Sandbox`."""

from __future__ import annotations

from forge.config import SandboxMode, Settings
from forge.sandbox.base import ExecRequest, ExecResult, Sandbox, SandboxError, scrub_env
from forge.sandbox.docker import DockerSandbox, docker_available
from forge.sandbox.local import LocalSandbox

__all__ = [
    "DockerSandbox",
    "ExecRequest",
    "ExecResult",
    "Sandbox",
    "SandboxError",
    "create_sandbox",
    "docker_available",
    "scrub_env",
]


def create_sandbox(settings: Settings) -> Sandbox:
    """Instantiate the configured sandbox backend.

    Raises :class:`SandboxError` if the requested backend is unavailable — an
    explicit failure is better than silently downgrading isolation.
    """
    if settings.sandbox is SandboxMode.DOCKER:
        return DockerSandbox(
            workspace_root=settings.workspace_root,
            image=settings.sandbox_image,
            cpus=settings.sandbox_cpus,
            memory=settings.sandbox_memory,
            network=settings.sandbox_network,
            pids_limit=settings.sandbox_pids_limit,
        )
    return LocalSandbox()
