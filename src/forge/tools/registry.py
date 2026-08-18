"""A registry of available tools."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

from forge.tools.base import Tool


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool[Any]] | None = None) -> None:
        self._tools: dict[str, Tool[Any]] = {}
        for tool in tools or ():
            self.register(tool)

    def register(self, tool: Tool[Any]) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool {tool.name!r} is already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool[Any] | None:
        return self._tools.get(name)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __iter__(self) -> Iterator[Tool[Any]]:
        return iter(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)

    def names(self) -> list[str]:
        return list(self._tools)

    def to_provider_schema(self) -> list[dict[str, Any]]:
        """The tool list advertised to the model provider."""
        return [tool.to_provider_schema() for tool in self._tools.values()]
