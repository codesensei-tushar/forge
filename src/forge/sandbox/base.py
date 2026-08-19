"""Sandbox abstraction for command execution.

Every shell command the agent runs goes through a :class:`Sandbox`, so the
isolation policy is one swappable object rather than a flag threaded through
tool code. :class:`~forge.sandbox.local.LocalSandbox` runs on the host (fast,
what you want for your own repo); :class:`~forge.sandbox.docker.DockerSandbox`
adds CPU/memory/PID caps and disables networking by default.
"""

from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

# Environment variables never forwarded into a sandboxed command. The agent has
# no legitimate need for the operator's credentials, and shelling out is the
# easiest way for a prompt injection to exfiltrate them.
_SECRET_PATTERN = re.compile(
    r"(KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|SESSION|COOKIE|AUTH)",
    re.IGNORECASE,
)
_SECRET_PREFIXES: tuple[str, ...] = (
    "ANTHROPIC_",
    "OPENAI_",
    "AWS_",
    "GOOGLE_",
    "AZURE_",
    "GH_",
    "GITHUB_",
    "NPM_",
    "PYPI_",
    "DOCKER_",
    "SLACK_",
    "STRIPE_",
)


def scrub_env(env: dict[str, str] | None = None) -> dict[str, str]:
    """Return ``env`` (default: the process environment) minus secret-looking keys."""
    source = dict(os.environ if env is None else env)
    return {
        key: value
        for key, value in source.items()
        if not _SECRET_PATTERN.search(key) and not key.startswith(_SECRET_PREFIXES)
    }


@dataclass(frozen=True)
class ExecRequest:
    """One command to execute inside the sandbox."""

    command: str
    cwd: Path
    timeout: int
    env: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ExecResult:
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False
    duration_s: float = 0.0
    # Set when the sandbox itself failed to start the command.
    startup_error: str | None = None

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out and self.startup_error is None


class SandboxError(RuntimeError):
    """Raised when a sandbox cannot be created or prepared."""


class Sandbox(ABC):
    """A place to run commands with a defined isolation boundary."""

    name: str = "base"

    @abstractmethod
    async def exec(self, request: ExecRequest) -> ExecResult:
        """Run one command and return its captured result. Must not raise for
        ordinary failures (non-zero exit, timeout) — encode them in the result."""
        raise NotImplementedError

    @abstractmethod
    def describe(self) -> str:
        """A short human-readable summary of the active limits."""
        raise NotImplementedError

    async def aclose(self) -> None:
        """Release any resources held by the sandbox."""
        return None
