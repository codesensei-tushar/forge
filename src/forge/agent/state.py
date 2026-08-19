"""Mutable conversation state for an agent session."""

from __future__ import annotations

from dataclasses import dataclass, field

from forge.providers.types import Message, TextBlock, ToolResultBlock, ToolUseBlock, Usage

# Rough bytes-per-token ratio for English source code and prose. Good enough for
# a budget guard; the provider's real count arrives with each response.
_CHARS_PER_TOKEN = 4


@dataclass
class AgentState:
    """Conversation history plus loop bookkeeping.

    A single state may span multiple tasks — the REPL reuses one so the agent
    keeps context across turns.
    """

    system_prompt: str
    messages: list[Message] = field(default_factory=list)
    iterations: int = 0
    compactions: int = 0
    usage: Usage = field(default_factory=Usage)

    def add_user(self, text: str) -> None:
        self.messages.append(Message.user(text))

    def add_message(self, message: Message) -> None:
        self.messages.append(message)

    def add_tool_results(self, results: list[ToolResultBlock]) -> None:
        self.messages.append(Message.tool_results(results))

    def record_usage(self, usage: Usage) -> None:
        self.usage = self.usage + usage

    def last_assistant_text(self) -> str:
        for message in reversed(self.messages):
            if message.role == "assistant" and (text := message.text().strip()):
                return text
        return ""

    def estimate_tokens(self) -> int:
        """Rough token estimate over the system prompt plus full history."""
        chars = len(self.system_prompt)
        for message in self.messages:
            for block in message.content:
                if isinstance(block, TextBlock):
                    chars += len(block.text)
                elif isinstance(block, ToolResultBlock):
                    chars += len(block.content)
                elif isinstance(block, ToolUseBlock):
                    chars += len(str(block.input)) + len(block.name)
        return chars // _CHARS_PER_TOKEN
