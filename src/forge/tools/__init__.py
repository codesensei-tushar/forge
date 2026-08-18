"""Tools package: registry, context, and the built-in tool set."""

from __future__ import annotations

from typing import Any

from forge.tools.base import Tool, ToolResult
from forge.tools.context import PathOutsideWorkspaceError, ToolContext
from forge.tools.filesystem import filesystem_tools
from forge.tools.registry import ToolRegistry
from forge.tools.shell import shell_tools


def default_tools() -> list[Tool[Any]]:
    """The built-in Phase 1 tool set (filesystem + shell)."""
    return [*filesystem_tools(), *shell_tools()]


def default_registry() -> ToolRegistry:
    return ToolRegistry(default_tools())


__all__ = [
    "PathOutsideWorkspaceError",
    "Tool",
    "ToolContext",
    "ToolRegistry",
    "ToolResult",
    "default_registry",
    "default_tools",
]
