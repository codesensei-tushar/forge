"""Agent package: the runtime loop, its state, and context assembly."""

from forge.agent.context import ContextManager, Environment, gather_environment
from forge.agent.loop import AgentResult, AgentRuntime, RunStatus, build_runtime
from forge.agent.prompts import build_system_prompt
from forge.agent.state import AgentState

__all__ = [
    "AgentResult",
    "AgentRuntime",
    "AgentState",
    "ContextManager",
    "Environment",
    "RunStatus",
    "build_runtime",
    "build_system_prompt",
    "gather_environment",
]
