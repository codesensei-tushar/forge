"""Execution context shared by all tools.

Centralizes workspace resolution and path-safety so every filesystem-touching
tool enforces the same boundary. In Phase 3 this is where a sandbox executor
will be injected; tools call through the context rather than the OS directly.
"""

from __future__ import annotations

from pathlib import Path

import structlog

from forge.config import Settings
from forge.logging import get_logger


class PathOutsideWorkspaceError(ValueError):
    """Raised when a tool tries to touch a path outside the workspace root."""


class ToolContext:
    def __init__(self, settings: Settings, logger: structlog.stdlib.BoundLogger | None = None) -> None:
        self.settings = settings
        self.workspace_root: Path = settings.workspace_root
        self.log = logger or get_logger("forge.tools")

    def resolve_path(self, path: str) -> Path:
        """Resolve ``path`` (relative to the workspace) and reject escapes.

        Symlinks are resolved before the containment check, so a link that
        points outside the workspace is rejected too.
        """
        candidate = Path(path)
        resolved = (
            candidate.resolve()
            if candidate.is_absolute()
            else (self.workspace_root / candidate).resolve()
        )
        try:
            resolved.relative_to(self.workspace_root)
        except ValueError as exc:
            raise PathOutsideWorkspaceError(
                f"{path!r} resolves outside the workspace ({self.workspace_root})"
            ) from exc
        return resolved

    def relative(self, path: Path) -> str:
        """Format an absolute path relative to the workspace for display."""
        try:
            return str(path.relative_to(self.workspace_root))
        except ValueError:
            return str(path)
