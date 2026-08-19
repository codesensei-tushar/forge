"""The Tool abstraction, its risk classification, and its result type."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

# Re-exported: Risk is defined below both packages so tools and permissions can
# share it without importing each other. See forge/permissions/risk.py.
from forge.permissions.risk import Risk
from forge.tools.context import ToolContext

__all__ = ["ArgsT", "Risk", "Tool", "ToolResult"]

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


def _strip_titles(schema: dict[str, Any]) -> dict[str, Any]:
    """Drop pydantic's auto-generated ``title`` keys from a JSON schema.

    Tool schemas are re-sent on every model call, so the redundant titles are
    pure token cost. Descriptions, types, and defaults are all preserved.
    """
    cleaned = {k: v for k, v in schema.items() if k != "title"}
    if "properties" in cleaned:
        cleaned["properties"] = {
            name: _strip_titles(prop) if isinstance(prop, dict) else prop
            for name, prop in cleaned["properties"].items()
        }
    for key in ("anyOf", "oneOf", "allOf"):
        if key in cleaned and isinstance(cleaned[key], list):
            cleaned[key] = [
                _strip_titles(entry) if isinstance(entry, dict) else entry for entry in cleaned[key]
            ]
    return cleaned


class Tool(ABC, Generic[ArgsT]):
    """Base class for a model-callable tool.

    Subclasses set ``name``/``description``/``risk`` and an ``Args`` pydantic
    model. The JSON schema advertised to the model is derived from ``Args``, and
    the executor validates raw model input against it before ``run`` is called.
    """

    name: str
    description: str
    risk: Risk = Risk.WRITE
    # Per-tool wall-clock ceiling; ``None`` uses the global ``tool_timeout``.
    timeout: int | None = None
    Args: type[ArgsT]

    @property
    def read_only(self) -> bool:
        return self.risk is Risk.READ

    def risk_for(self, args: dict[str, Any]) -> Risk:
        """Risk of a *specific* call.

        Defaults to the tool's static ``risk``. Tools whose blast radius depends
        on their arguments (the shell, most obviously) override this to escalate.
        """
        return self.risk

    def input_schema(self) -> dict[str, Any]:
        return _strip_titles(self.Args.model_json_schema())

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

    def __repr__(self) -> str:
        return f"<Tool {self.name} risk={self.risk.value}>"
