"""Mutable conversation state for an agent session."""

from __future__ import annotations

from dataclasses import dataclass, field

from forge.models.types import Message, TextBlock, ToolResultBlock


@dataclass
class AgentState:
    """Conversation history plus loop bookkeeping.

    A single state may span multiple tasks (the REPL reuses one state so the
    agent keeps context across turns).
    """

    system_prompt: str
    messages: list[Message] = field(default_factory=list)
    iterations: int = 0

    def add_user(self, text: str) -> None:
        self.messages.append(Message.user(text))

    def add_message(self, message: Message) -> None:
        self.messages.append(message)

    def add_tool_results(self, results: list[ToolResultBlock]) -> None:
        self.messages.append(Message.tool_results(results))

    def estimate_tokens(self) -> int:
        """Rough token estimate (~4 chars/token) over the system prompt + history."""
        chars = len(self.system_prompt)
        for message in self.messages:
            for block in message.content:
                if isinstance(block, TextBlock):
                    chars += len(block.text)
                elif isinstance(block, ToolResultBlock):
                    chars += len(block.content)
                else:  # tool_use: count the serialized input
                    chars += len(str(getattr(block, "input", "")))
        return chars // 4
