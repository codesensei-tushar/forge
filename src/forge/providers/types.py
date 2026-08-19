"""Normalized, provider-agnostic message and response types.

These types are the lingua franca between the agent loop and any concrete
:class:`~forge.models.base.ModelProvider`. Each provider is responsible for
translating to and from its own SDK shapes, so the rest of Forge never depends
on a particular vendor's schema.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field


class TextBlock(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ToolUseBlock(BaseModel):
    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: dict[str, Any] = Field(default_factory=dict)


class ToolResultBlock(BaseModel):
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: str
    is_error: bool = False


ContentBlock = Annotated[
    TextBlock | ToolUseBlock | ToolResultBlock,
    Field(discriminator="type"),
]


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )


class Message(BaseModel):
    """A single conversation turn as a role plus a list of content blocks."""

    role: Literal["user", "assistant"]
    content: list[ContentBlock]

    @classmethod
    def user(cls, text: str) -> Message:
        return cls(role="user", content=[TextBlock(text=text)])

    @classmethod
    def assistant(cls, text: str) -> Message:
        return cls(role="assistant", content=[TextBlock(text=text)])

    @classmethod
    def tool_results(cls, results: list[ToolResultBlock]) -> Message:
        """Tool results are delivered back to the model as a user turn."""
        return cls(role="user", content=list(results))

    def text(self) -> str:
        return "".join(b.text for b in self.content if isinstance(b, TextBlock))


class ModelResponse(BaseModel):
    """The normalized result of one model completion."""

    blocks: list[ContentBlock]
    stop_reason: StopReason
    usage: Usage = Field(default_factory=Usage)
    model: str = ""

    def text(self) -> str:
        return "".join(b.text for b in self.blocks if isinstance(b, TextBlock))

    def tool_uses(self) -> list[ToolUseBlock]:
        return [b for b in self.blocks if isinstance(b, ToolUseBlock)]

    def to_message(self) -> Message:
        """Represent this response as an assistant message for history."""
        return Message(role="assistant", content=list(self.blocks))
