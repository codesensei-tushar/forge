"""The model-provider abstraction plus an offline fake for tests.

Providers raise :class:`ProviderError` for any transport or API failure.
``retryable`` distinguishes "try again in a moment" (429s, 5xx, timeouts) from
"this will never work" (bad credentials, invalid request), which lets the agent
loop retry the former with backoff instead of ending the run.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any

from forge.providers.types import Message, ModelResponse, TextBlock


class ProviderError(RuntimeError):
    """Raised when an underlying model API call fails."""

    def __init__(self, message: str, *, retryable: bool = False, status: int | None = None) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.status = status


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

    async def aclose(self) -> None:
        """Release any transport resources. Overridden where relevant."""
        return None


class FakeProvider(ModelProvider):
    """A deterministic provider for tests and offline development.

    Supply either a fixed ``script`` of responses (returned in order) or a
    ``responder`` callable that produces a response from the current messages.
    Every call is recorded on ``self.calls`` for assertions.
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
        self.tools_seen: list[list[dict[str, Any]]] = []

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
        self.tools_seen.append(list(tools))
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
