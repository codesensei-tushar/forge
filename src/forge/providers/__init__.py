"""Provider package: normalized message types and the provider abstraction."""

from forge.providers.base import FakeProvider, ModelProvider, ProviderError
from forge.providers.registry import SUPPORTED_PROVIDERS, create_provider
from forge.providers.types import (
    ContentBlock,
    Message,
    ModelResponse,
    StopReason,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    Usage,
    normalize_stop_reason,
)

__all__ = [
    "SUPPORTED_PROVIDERS",
    "ContentBlock",
    "FakeProvider",
    "Message",
    "ModelProvider",
    "ModelResponse",
    "ProviderError",
    "StopReason",
    "TextBlock",
    "ToolResultBlock",
    "ToolUseBlock",
    "Usage",
    "create_provider",
    "normalize_stop_reason",
]
