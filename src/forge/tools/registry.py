"""The tool registry: the single source of truth for what the model can call.

Everything the model is allowed to do is registered here, and every call it
makes is looked up here. Keeping that mapping in one object is what lets later
capabilities — MCP servers, per-tool metrics, dynamic enable/disable — be added
without touching the agent loop.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

from forge.tools.base import Risk, Tool


class ToolNotFoundError(KeyError):
    """Raised by :meth:`ToolRegistry.require` for an unregistered name."""


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool[Any]] | None = None) -> None:
        self._tools: dict[str, Tool[Any]] = {}
        self.register_all(tools or ())

    def register(self, tool: Tool[Any], *, replace: bool = False) -> None:
        """Add a tool. Duplicate names are an error unless ``replace`` is set."""
        if not getattr(tool, "name", ""):
            raise ValueError(f"{type(tool).__name__} has no name")
        if tool.name in self._tools and not replace:
            raise ValueError(f"Tool {tool.name!r} is already registered")
        self._tools[tool.name] = tool

    def register_all(self, tools: Iterable[Tool[Any]], *, replace: bool = False) -> None:
        for tool in tools:
            self.register(tool, replace=replace)

    def unregister(self, name: str) -> Tool[Any] | None:
        """Remove a tool by name, returning it if it was present."""
        return self._tools.pop(name, None)

    def get(self, name: str) -> Tool[Any] | None:
        return self._tools.get(name)

    def require(self, name: str) -> Tool[Any]:
        tool = self._tools.get(name)
        if tool is None:
            raise ToolNotFoundError(f"Unknown tool {name!r}. Available: {', '.join(self.names())}")
        return tool

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __iter__(self) -> Iterator[Tool[Any]]:
        return iter(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def by_risk(self, risk: Risk) -> list[Tool[Any]]:
        return [t for t in self._tools.values() if t.risk is risk]

    def to_provider_schema(self) -> list[dict[str, Any]]:
        """The tool list advertised to the model provider."""
        return [tool.to_provider_schema() for tool in self._tools.values()]

    def describe(self) -> str:
        """A compact ``name (risk): description`` listing for the system prompt."""
        return "\n".join(f"- {tool.name}: {tool.description}" for tool in self._tools.values())
