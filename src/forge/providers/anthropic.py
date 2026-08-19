"""Anthropic Messages API provider.

Honors the standard ``ANTHROPIC_BASE_URL`` / ``ANTHROPIC_AUTH_TOKEN`` /
``ANTHROPIC_MODEL`` environment so Forge runs against an Anthropic-compatible
gateway out of the box. ``auth_token`` sets an ``Authorization: Bearer`` header
(what gateways typically expect); ``api_key`` is used as a fallback.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from forge.models.base import ModelProvider, ProviderError
from forge.models.types import (
    Message,
    ModelResponse,
    TextBlock,
    ToolUseBlock,
    Usage,
)


__all__ = ["AnthropicProvider", "ProviderError"]


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

        client_kwargs: dict[str, Any] = {"timeout": timeout, "max_retries": max_retries}
        if base_url:
            client_kwargs["base_url"] = base_url
        # Prefer bearer-token auth (gateways); fall back to x-api-key.
        if auth_token:
            client_kwargs["auth_token"] = auth_token
        elif api_key:
            client_kwargs["api_key"] = api_key

        self._client = AsyncAnthropic(**client_kwargs)

    @staticmethod
    def _to_api_messages(messages: Sequence[Message]) -> list[dict[str, Any]]:
        # Our normalized blocks serialize directly into the API's block shapes.
        return [
            {"role": m.role, "content": [b.model_dump() for b in m.content]}
            for m in messages
        ]

    def _from_api_response(self, resp: Any) -> ModelResponse:
        blocks: list[Any] = []
        for block in resp.content:
            btype = getattr(block, "type", None)
            if btype == "text":
                blocks.append(TextBlock(text=block.text))
            elif btype == "tool_use":
                blocks.append(
                    ToolUseBlock(
                        id=block.id,
                        name=block.name,
                        input=dict(block.input or {}),
                    )
                )
            # Other block types (e.g. thinking) are ignored in Phase 1.

        usage = Usage(
            input_tokens=getattr(resp.usage, "input_tokens", 0),
            output_tokens=getattr(resp.usage, "output_tokens", 0),
        )
        return ModelResponse(
            blocks=blocks,
            stop_reason=resp.stop_reason or "end_turn",
            usage=usage,
            model=getattr(resp, "model", self.model),
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
            raise ProviderError(f"Anthropic API call failed: {exc}") from exc

        return self._from_api_response(resp)
