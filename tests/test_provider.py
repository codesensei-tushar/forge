"""Tests for the provider layer: normalized types, the fake, and the Anthropic adapter.

No test here makes a network call. The Anthropic provider is exercised through
its pure translation helpers and its constructor guards.
"""

from __future__ import annotations

import pytest

from forge.config import Settings
from forge.providers.anthropic import AnthropicProvider, _classify
from forge.providers.base import FakeProvider, ModelProvider, ProviderError
from forge.providers.registry import SUPPORTED_PROVIDERS, create_provider
from forge.providers.types import (
    Message,
    ModelResponse,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    Usage,
    normalize_stop_reason,
)


# --------------------------------------------------------------------------- #
# Stop reasons — the field that must never reject a real API value
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "reason",
    [
        "end_turn",
        "tool_use",
        "max_tokens",
        "stop_sequence",
        "pause_turn",
        "refusal",
        "content_filter",
        "error",
    ],
)
def test_all_api_stop_reasons_are_accepted(reason: str) -> None:
    response = ModelResponse(blocks=[TextBlock(text="x")], stop_reason=reason)  # type: ignore[arg-type]
    assert response.stop_reason == reason


def test_tool_use_is_a_valid_stop_reason() -> None:
    """The one an agent loop cannot function without."""
    assert normalize_stop_reason("tool_use") == "tool_use"


@pytest.mark.parametrize("raw", ["length", "something_new", "", None, 42, object()])
def test_unknown_stop_reason_degrades_instead_of_raising(raw: object) -> None:
    assert normalize_stop_reason(raw) == "end_turn"


def test_truncated_only_for_max_tokens() -> None:
    assert ModelResponse(blocks=[], stop_reason="max_tokens").truncated
    assert not ModelResponse(blocks=[], stop_reason="end_turn").truncated


# --------------------------------------------------------------------------- #
# Normalized types
# --------------------------------------------------------------------------- #
def test_usage_adds_every_field() -> None:
    total = Usage(
        input_tokens=1, output_tokens=2, cache_read_tokens=3, cache_write_tokens=4
    ) + Usage(input_tokens=10, output_tokens=20, cache_read_tokens=30, cache_write_tokens=40)
    assert (total.input_tokens, total.output_tokens) == (11, 22)
    assert (total.cache_read_tokens, total.cache_write_tokens) == (33, 44)
    assert total.total_tokens == 33


def test_message_constructors() -> None:
    assert Message.user("hi").role == "user"
    assert Message.assistant("yo").role == "assistant"
    # Tool results go back to the model as a *user* turn — an API requirement.
    results = Message.tool_results([ToolResultBlock(tool_use_id="1", content="out")])
    assert results.role == "user"


def test_content_blocks_are_discriminated_on_type() -> None:
    message = Message.model_validate(
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "thinking"},
                {"type": "tool_use", "id": "t1", "name": "shell", "input": {"command": "ls"}},
            ],
        }
    )
    assert isinstance(message.content[0], TextBlock)
    assert isinstance(message.content[1], ToolUseBlock)
    assert message.tool_uses()[0].input == {"command": "ls"}


def test_response_text_concatenates_only_text_blocks() -> None:
    response = ModelResponse(
        blocks=[TextBlock(text="a"), ToolUseBlock(id="1", name="x"), TextBlock(text="b")],
        stop_reason="tool_use",
    )
    assert response.text() == "ab"
    assert len(response.tool_uses()) == 1
    assert len(response.to_message().content) == 3


# --------------------------------------------------------------------------- #
# FakeProvider
# --------------------------------------------------------------------------- #
async def test_fake_returns_script_in_order() -> None:
    provider = FakeProvider(
        script=[
            ModelResponse(blocks=[TextBlock(text="one")], stop_reason="end_turn"),
            ModelResponse(blocks=[TextBlock(text="two")], stop_reason="end_turn"),
        ]
    )
    kwargs: dict[str, object] = {
        "system": "s",
        "messages": [],
        "tools": [],
        "max_tokens": 10,
        "temperature": 0.0,
    }
    assert (await provider.complete(**kwargs)).text() == "one"  # type: ignore[arg-type]
    assert (await provider.complete(**kwargs)).text() == "two"  # type: ignore[arg-type]
    # Exhausted script must terminate rather than hang a loop.
    exhausted = await provider.complete(**kwargs)  # type: ignore[arg-type]
    assert exhausted.stop_reason == "end_turn"


async def test_fake_records_calls_and_tools() -> None:
    provider = FakeProvider()
    schema = [{"name": "shell", "description": "", "input_schema": {}}]
    await provider.complete(
        system="s",
        messages=[Message.user("hello")],
        tools=schema,
        max_tokens=10,
        temperature=0.0,
    )
    assert provider.calls[0][0].text() == "hello"
    assert provider.tools_seen[0] == schema


async def test_fake_responder_sees_history() -> None:
    def responder(messages: object) -> ModelResponse:
        assert isinstance(messages, list)
        return ModelResponse(
            blocks=[TextBlock(text=f"saw {len(messages)}")], stop_reason="end_turn"
        )

    provider = FakeProvider(responder=responder)
    response = await provider.complete(
        system="",
        messages=[Message.user("a"), Message.user("b")],
        tools=[],
        max_tokens=1,
        temperature=0.0,
    )
    assert response.text() == "saw 2"


async def test_base_provider_aclose_is_a_noop() -> None:
    assert await FakeProvider().aclose() is None


def test_model_provider_is_abstract() -> None:
    with pytest.raises(TypeError):
        ModelProvider(model="x")  # type: ignore[abstract]


# --------------------------------------------------------------------------- #
# Error classification
# --------------------------------------------------------------------------- #
def test_provider_error_carries_retryability() -> None:
    exc = ProviderError("boom", retryable=True, status=529)
    assert exc.retryable and exc.status == 529
    assert not ProviderError("nope").retryable


class _StatusError(Exception):
    def __init__(self, status: int) -> None:
        super().__init__(f"http {status}")
        self.status_code = status


@pytest.mark.parametrize("status", [408, 429, 500, 502, 503, 529])
def test_transient_statuses_are_retryable(status: int) -> None:
    assert _classify(_StatusError(status)) == (True, status)


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_client_errors_are_terminal(status: int) -> None:
    retryable, seen = _classify(_StatusError(status))
    assert not retryable and seen == status


def test_status_is_read_from_a_nested_response() -> None:
    class _Response:
        status_code = 503

    class _Wrapped(Exception):
        response = _Response()

    assert _classify(_Wrapped()) == (True, 503)


@pytest.mark.parametrize(
    "message",
    ["Request timeout", "connection reset by peer", "server overloaded", "temporarily unavailable"],
)
def test_transport_failures_without_a_status_are_retryable(message: str) -> None:
    retryable, status = _classify(RuntimeError(message))
    assert retryable and status is None


def test_unrecognized_failure_is_terminal() -> None:
    assert _classify(ValueError("invalid tool schema")) == (False, None)


# --------------------------------------------------------------------------- #
# Anthropic translation
# --------------------------------------------------------------------------- #
def test_empty_text_blocks_are_dropped() -> None:
    """The API rejects empty text; a tool-only assistant turn is common."""
    payload = AnthropicProvider._to_api_messages(
        [
            Message(
                role="assistant",
                content=[TextBlock(text="   "), ToolUseBlock(id="t1", name="shell")],
            )
        ]
    )
    assert len(payload) == 1
    assert [b["type"] for b in payload[0]["content"]] == ["tool_use"]


def test_messages_left_empty_are_skipped() -> None:
    payload = AnthropicProvider._to_api_messages(
        [Message.assistant(""), Message.user("real content")]
    )
    assert len(payload) == 1
    assert payload[0]["role"] == "user"


def test_tool_result_blocks_survive_translation() -> None:
    payload = AnthropicProvider._to_api_messages(
        [Message.tool_results([ToolResultBlock(tool_use_id="t1", content="out", is_error=True)])]
    )
    block = payload[0]["content"][0]
    assert block["tool_use_id"] == "t1" and block["is_error"] is True


def test_api_response_is_normalized() -> None:
    class _Block:
        def __init__(self, **kw: object) -> None:
            self.__dict__.update(kw)

    class _Usage:
        input_tokens = 11
        output_tokens = 7
        cache_read_input_tokens = 3
        cache_creation_input_tokens = 5

    class _Resp:
        content = [
            _Block(type="text", text="hello"),
            _Block(type="thinking", thinking="ignored"),
            _Block(type="tool_use", id="t1", name="read_file", input={"path": "a"}),
        ]
        stop_reason = "tool_use"
        usage = _Usage()
        model = "claude-test"

    provider = AnthropicProvider.__new__(AnthropicProvider)
    provider.model = "fallback"
    response = provider._from_api_response(_Resp())

    assert response.text() == "hello"
    assert [b.name for b in response.tool_uses()] == ["read_file"]
    assert response.stop_reason == "tool_use"
    assert response.usage.input_tokens == 11
    assert response.usage.cache_read_tokens == 3
    assert response.model == "claude-test"
    # The unknown "thinking" block is skipped rather than fatal.
    assert len(response.blocks) == 2


def test_missing_credentials_raise_a_provider_error() -> None:
    with pytest.raises(ProviderError, match="No credential configured"):
        AnthropicProvider(model="claude-test")


# --------------------------------------------------------------------------- #
# Provider factory
# --------------------------------------------------------------------------- #
def test_factory_builds_the_fake() -> None:
    provider = create_provider(Settings(provider="fake", model="m"))
    assert isinstance(provider, FakeProvider) and provider.model == "m"


def test_factory_requires_a_model_for_anthropic() -> None:
    with pytest.raises(ValueError, match="No model configured"):
        create_provider(Settings(provider="anthropic", model=None))


def test_factory_rejects_unknown_providers() -> None:
    with pytest.raises(ValueError, match="Unknown provider"):
        create_provider(Settings(provider="llamafile", model="m"))


def test_supported_providers_are_advertised() -> None:
    assert set(SUPPORTED_PROVIDERS) == {"anthropic", "fake"}


def test_factory_builds_anthropic_with_a_credential() -> None:
    provider = create_provider(
        Settings(
            provider="anthropic",
            model="claude-test",
            base_url="https://gateway.example/",
            auth_token="token-123",
        )
    )
    assert isinstance(provider, AnthropicProvider)
    assert provider.model == "claude-test"
