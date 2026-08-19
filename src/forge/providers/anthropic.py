"""Anthropic Messages API provider.

Honors the standard ``ANTHROPIC_BASE_URL`` / ``ANTHROPIC_AUTH_TOKEN`` /
``ANTHROPIC_MODEL`` environment so Forge runs against an Anthropic-compatible
gateway out of the box. ``auth_token`` sets an ``Authorization: Bearer`` header
(what gateways typically expect); ``api_key`` is used as a fallback.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from forge.providers.base import ModelProvider, ProviderError
from forge.providers.types import (
    Message,
    ModelResponse,
    TextBlock,
    ToolUseBlock,
    Usage,
    normalize_stop_reason,
)

__all__ = ["AnthropicProvider", "ProviderError"]

# HTTP statuses worth retrying: rate limits, overload, and transient 5xx.
_RETRYABLE_STATUS: frozenset[int] = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})

# Substrings that mark a transport-level failure when no status code is exposed.
_RETRYABLE_HINTS: tuple[str, ...] = (
    "timeout",
    "timed out",
    "connection",
    "temporarily",
    "overloaded",
    "econnreset",
    "read operation",
)


def _classify(exc: Exception) -> tuple[bool, int | None]:
    """Decide whether an SDK exception is worth retrying, and its status."""
    status = getattr(exc, "status_code", None)
    if not isinstance(status, int):
        response = getattr(exc, "response", None)
        candidate = getattr(response, "status_code", None)
        status = candidate if isinstance(candidate, int) else None

    if status is not None:
        return status in _RETRYABLE_STATUS, status

    text = f"{type(exc).__name__} {exc}".lower()
    return any(hint in text for hint in _RETRYABLE_HINTS), None


class AnthropicProvider(ModelProvider):
    name = "anthropic"

    def __init__(
        self,
        *,
        model: str,
        base_url: str | None = None,
        auth_token: str | None = None,
        api_key: str | None = None,
        timeout: float = 600.0,
        max_retries: int = 2,
    ) -> None:
        super().__init__(model=model)
        try:
            from anthropic import AsyncAnthropic
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ProviderError(
                "The 'anthropic' package is required for the Anthropic provider."
            ) from exc

        if not (auth_token or api_key):
            raise ProviderError(
                "No credential configured. Set ANTHROPIC_AUTH_TOKEN (gateways) or "
                "ANTHROPIC_API_KEY."
            )

        client_kwargs: dict[str, Any] = {"timeout": timeout, "max_retries": max_retries}
        if base_url:
            client_kwargs["base_url"] = base_url
        # Prefer bearer-token auth (gateways); fall back to x-api-key.
        if auth_token:
            client_kwargs["auth_token"] = auth_token
        else:
            client_kwargs["api_key"] = api_key

        self._client = AsyncAnthropic(**client_kwargs)

    @staticmethod
    def _to_api_messages(messages: Sequence[Message]) -> list[dict[str, Any]]:
        """Serialize normalized messages into API block shapes.

        Empty text blocks are dropped and any message left with no content is
        skipped: the API rejects both, and either can arise from a model that
        replies with tool calls and no prose.
        """
        out: list[dict[str, Any]] = []
        for message in messages:
            blocks: list[dict[str, Any]] = []
            for block in message.content:
                if isinstance(block, TextBlock) and not block.text.strip():
                    continue
                blocks.append(block.model_dump())
            if blocks:
                out.append({"role": message.role, "content": blocks})
        return out

    def _from_api_response(self, resp: Any) -> ModelResponse:
        blocks: list[Any] = []
        for block in resp.content or ():
            btype = getattr(block, "type", None)
            if btype == "text":
                blocks.append(TextBlock(text=block.text))
            elif btype == "tool_use":
                blocks.append(
                    ToolUseBlock(id=block.id, name=block.name, input=dict(block.input or {}))
                )
            # Other block types (thinking, server tool use) are ignored for now.

        raw_usage = getattr(resp, "usage", None)
        usage = Usage(
            input_tokens=getattr(raw_usage, "input_tokens", 0) or 0,
            output_tokens=getattr(raw_usage, "output_tokens", 0) or 0,
            cache_read_tokens=getattr(raw_usage, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(raw_usage, "cache_creation_input_tokens", 0) or 0,
        )
        return ModelResponse(
            blocks=blocks,
            stop_reason=normalize_stop_reason(getattr(resp, "stop_reason", None)),
            usage=usage,
            model=getattr(resp, "model", "") or self.model,
        )

    async def complete(
        self,
        *,
        system: str,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]],
        max_tokens: int,
        temperature: float,
    ) -> ModelResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": self._to_api_messages(messages),
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = list(tools)
            kwargs["tool_choice"] = {"type": "auto"}

        try:
            resp = await self._client.messages.create(**kwargs)
        except Exception as exc:  # normalize any SDK/transport error
            retryable, status = _classify(exc)
            suffix = f" (HTTP {status})" if status else ""
            raise ProviderError(
                f"Anthropic API call failed{suffix}: {exc}", retryable=retryable, status=status
            ) from exc

        return self._from_api_response(resp)

    async def aclose(self) -> None:
        await self._client.close()
