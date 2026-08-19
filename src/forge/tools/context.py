"""Execution context shared by all tools.

Centralizes the three things every tool needs and none should implement itself:
workspace path resolution (with escape rejection), command execution (via the
configured :class:`~forge.sandbox.base.Sandbox`), and logging. Tools call
through the context rather than reaching for the OS directly, which is what
makes swapping in a container sandbox a one-line configuration change.
"""

from __future__ import annotations

from pathlib import Path

import structlog

from forge.config import Settings
from forge.logging import get_logger
from forge.sandbox.base import ExecRequest, ExecResult, Sandbox
from forge.sandbox.local import LocalSandbox


class PathOutsideWorkspaceError(ValueError):
    """Raised when a tool tries to touch a path outside the workspace root."""


class ToolContext:
    def __init__(
        self,
        settings: Settings,
        logger: structlog.stdlib.BoundLogger | None = None,
        sandbox: Sandbox | None = None,
    ) -> None:
        self.settings = settings
        self.workspace_root: Path = settings.workspace_root
        self.log = logger or get_logger("forge.tools")
        self.sandbox: Sandbox = sandbox or LocalSandbox()

    def resolve_path(self, path: str) -> Path:
        """Resolve ``path`` (relative to the workspace) and reject escapes.

        Symlinks are resolved before the containment check, so a link that
        points outside the workspace is rejected too.
        """
        candidate = Path(path).expanduser()
        resolved = (
            candidate.resolve()
            if candidate.is_absolute()
            else (self.workspace_root / candidate).resolve()
        )
        if resolved != self.workspace_root and self.workspace_root not in resolved.parents:
            raise PathOutsideWorkspaceError(
                f"{path!r} resolves outside the workspace ({self.workspace_root})"
            )
        return resolved

    def relative(self, path: Path) -> str:
        """Format an absolute path relative to the workspace for display."""
        try:
            return str(path.relative_to(self.workspace_root))
        except ValueError:
            return str(path)

    async def exec(
        self,
        command: str,
        *,
        cwd: Path | None = None,
        timeout: int | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecResult:
        """Run a shell command through the active sandbox."""
        return await self.sandbox.exec(
            ExecRequest(
                command=command,
                cwd=cwd or self.workspace_root,
                timeout=timeout or self.settings.shell_timeout,
                env=dict(env or {}),
            )
        )

    async def aclose(self) -> None:
        await self.sandbox.aclose()
