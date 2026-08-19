"""Provider factory: map a provider name + settings to a live provider."""

from __future__ import annotations

from forge.config import Settings
from forge.providers.base import FakeProvider, ModelProvider

SUPPORTED_PROVIDERS: tuple[str, ...] = ("anthropic", "fake")


def create_provider(settings: Settings) -> ModelProvider:
    """Instantiate the configured :class:`ModelProvider`.

    New providers (OpenAI, OpenRouter, Ollama, ...) plug in here by adding a
    branch; the rest of Forge is unaffected because they all speak the same
    normalized message types.
    """
    provider = settings.provider.lower()

    if provider == "anthropic":
        if not settings.model:
            raise ValueError(
                "No model configured. Set ANTHROPIC_MODEL / FORGE_MODEL, or pass --model."
            )
        from forge.providers.anthropic import AnthropicProvider

        return AnthropicProvider(
            model=settings.model,
            base_url=settings.base_url,
            auth_token=settings.auth_token,
            api_key=settings.api_key,
            timeout=settings.request_timeout,
        )

    if provider == "fake":
        return FakeProvider(model=settings.model or "fake-model")

    raise ValueError(
        f"Unknown provider: {settings.provider!r}. "
        f"Supported: {', '.join(repr(p) for p in SUPPORTED_PROVIDERS)}."
    )
