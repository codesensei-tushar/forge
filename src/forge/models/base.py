"""The model-provider abstraction plus an offline fake for tests."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any

from forge.models.types import Message, ModelResponse, TextBlock


class ProviderError(RuntimeError):
    """Raised when an underlying model API call fails."""


class ModelProvider(ABC):
    """A pluggable LLM backend.

    Concrete providers translate the normalized :class:`Message`/tool schema
    into their own SDK calls and translate the reply back into a
    :class:`ModelResponse`. Everything above this interface is vendor-neutral.
    """

    name: str = "base"

    def __init__(self, *, model: str) -> None:
        self.model = model

    @abstractmethod
    async def complete(
        self,
        *,
        system: str,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]],
        max_tokens: int,
        temperature: float,
    ) -> ModelResponse:
        """Run one completion and return the normalized response."""
        raise NotImplementedError


class FakeProvider(ModelProvider):
    """A deterministic provider for tests and offline development.

    Supply either a fixed ``script`` of responses (returned in order) or a
    ``responder`` callable that produces a response from the current messages.
    """

    name = "fake"

    def __init__(
        self,
        *,
        model: str = "fake-model",
        script: Sequence[ModelResponse] | None = None,
        responder: Callable[[Sequence[Message]], ModelResponse] | None = None,
    ) -> None:
        super().__init__(model=model)
        self._script = list(script or [])
        self._responder = responder
        self.calls: list[list[Message]] = []

    async def complete(
        self,
        *,
        system: str,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]],
        max_tokens: int,
        temperature: float,
    ) -> ModelResponse:
        self.calls.append(list(messages))
        if self._responder is not None:
            return self._responder(messages)
        if self._script:
            return self._script.pop(0)
        # Default: a terminal end_turn so a loop never hangs on an empty script.
        return ModelResponse(
            blocks=[TextBlock(text="(fake: no more scripted responses)")],
            stop_reason="end_turn",
            model=self.model,
        )
