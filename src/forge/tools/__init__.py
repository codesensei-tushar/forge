"""Tools package: registry, executor, context, and the built-in tool set."""

from __future__ import annotations

from typing import Any

from forge.config import Settings
from forge.tools.base import Risk, Tool, ToolResult
from forge.tools.context import PathOutsideWorkspaceError, ToolContext
from forge.tools.executor import Approver, ToolExecutor, ToolOutcome
from forge.tools.filesystem import filesystem_tools
from forge.tools.git import git_tools
from forge.tools.registry import ToolNotFoundError, ToolRegistry
from forge.tools.shell import shell_tools


def default_tools(settings: Settings | None = None) -> list[Tool[Any]]:
    """The built-in tool set: filesystem, shell, and (optionally) git."""
    tools: list[Tool[Any]] = [*filesystem_tools(), *shell_tools()]
    if settings is None or settings.enable_git_tools:
        tools += git_tools()
    return tools


def default_registry(settings: Settings | None = None) -> ToolRegistry:
    return ToolRegistry(default_tools(settings))


__all__ = [
    "Approver",
    "PathOutsideWorkspaceError",
    "Risk",
    "Tool",
    "ToolContext",
    "ToolExecutor",
    "ToolNotFoundError",
    "ToolOutcome",
    "ToolRegistry",
    "ToolResult",
    "default_registry",
    "default_tools",
    "filesystem_tools",
    "git_tools",
    "shell_tools",
]
