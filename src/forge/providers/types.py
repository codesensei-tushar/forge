"""Normalized, provider-agnostic message and response types.

These types are the lingua franca between the agent loop and any concrete
:class:`~forge.providers.base.ModelProvider`. Each provider translates to and
from its own SDK shapes, so the rest of Forge never depends on a particular
vendor's schema.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, get_args

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


# Why the model stopped generating. Mirrors the Anthropic Messages API set;
# other providers normalize into it via :func:`normalize_stop_reason`.
StopReason = Literal[
    "end_turn",  # model finished its turn
    "tool_use",  # model wants one or more tools run
    "max_tokens",  # hit the output-token ceiling mid-answer
    "stop_sequence",  # hit a configured stop sequence
    "pause_turn",  # long-running server tool paused; safe to continue
    "refusal",  # model declined to continue
    "content_filter",  # output blocked upstream
    "error",  # synthesized locally when a call failed
]

_KNOWN_STOP_REASONS: frozenset[str] = frozenset(get_args(StopReason))


def normalize_stop_reason(raw: object) -> StopReason:
    """Coerce a provider's stop reason into a known :data:`StopReason`.

    An unrecognized value must never abort a run, so anything unknown (or
    missing) degrades to ``"end_turn"``, which the loop treats as terminal.
    """
    if isinstance(raw, str) and raw in _KNOWN_STOP_REASONS:
        return raw  # type: ignore[return-value]
    return "end_turn"


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
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

    def tool_uses(self) -> list[ToolUseBlock]:
        return [b for b in self.content if isinstance(b, ToolUseBlock)]


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

    @property
    def truncated(self) -> bool:
        """True when the model was cut off by the output-token ceiling."""
        return self.stop_reason == "max_tokens"

    def to_message(self) -> Message:
        """Represent this response as an assistant message for history."""
        return Message(role="assistant", content=list(self.blocks))
