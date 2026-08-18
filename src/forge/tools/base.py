"""The Tool abstraction and its result type."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

from forge.tools.context import ToolContext

ArgsT = TypeVar("ArgsT", bound=BaseModel)


class ToolResult(BaseModel):
    """The outcome of a tool invocation, as fed back to the model."""

    content: str
    is_error: bool = False
    # Free-form data for observability; never sent to the model.
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def ok(cls, content: str, **metadata: Any) -> ToolResult:
        return cls(content=content, is_error=False, metadata=metadata)

    @classmethod
    def error(cls, content: str, **metadata: Any) -> ToolResult:
        return cls(content=content, is_error=True, metadata=metadata)


class Tool(ABC, Generic[ArgsT]):
    """Base class for a model-callable tool.

    Subclasses set ``name``/``description``/``read_only`` and an ``Args`` pydantic
    model. The JSON schema advertised to the model is derived from ``Args``, and
    the loop validates raw model input against it before ``run`` is ever called.
    """

    name: str
    description: str
    read_only: bool = False
    Args: type[ArgsT]

    def input_schema(self) -> dict[str, Any]:
        return self.Args.model_json_schema()

    def to_provider_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema(),
        }

    def parse_args(self, raw: dict[str, Any]) -> ArgsT:
        """Validate raw model input into a typed args model (may raise)."""
        return self.Args.model_validate(raw)

    @abstractmethod
    async def run(self, args: ArgsT, ctx: ToolContext) -> ToolResult:
        """Execute the tool. Should not raise for expected failures — return
        ``ToolResult.error(...)`` so the model can read and recover from it."""
        raise NotImplementedError
