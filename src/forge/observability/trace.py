"""Per-run tracing: model calls, tool calls, tokens, cost, and duration.

Kept as pure data + derived metrics (no rendering here) so it can later be
serialized to a store (Phase 7). The UI layer renders the summary.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field

# Best-effort per-million-token prices (USD), matched by keyword in the model
# name. Gateways may bill differently, so costs are labeled as estimates and
# fall back to None (shown as "n/a") for unknown models.
_PRICES: dict[str, tuple[float, float]] = {
    "opus": (15.0, 75.0),
    "sonnet": (3.0, 15.0),
    "haiku": (0.80, 4.0),
}

_run_counter = itertools.count(1)


@dataclass
class ModelCall:
    model: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    stop_reason: str


@dataclass
class ToolCall:
    name: str
    decision: str
    duration_s: float
    is_error: bool


@dataclass
class RunTrace:
    task: str = ""
    run_id: int = field(default_factory=lambda: next(_run_counter))
    started_at: float = field(default_factory=time.perf_counter)
    ended_at: float | None = None
    status: str = "running"
    model_calls: list[ModelCall] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)

    # --- recording ---
    def record_model_call(
        self, *, model: str, input_tokens: int, output_tokens: int, latency_s: float, stop_reason: str
    ) -> None:
        self.model_calls.append(
            ModelCall(model, input_tokens, output_tokens, latency_s, stop_reason)
        )

    def record_tool_call(self, *, name: str, decision: str, duration_s: float, is_error: bool) -> None:
        self.tool_calls.append(ToolCall(name, decision, duration_s, is_error))

    def finish(self, status: str) -> None:
        self.ended_at = time.perf_counter()
        self.status = status

    # --- derived metrics ---
    @property
    def duration_s(self) -> float:
        end = self.ended_at if self.ended_at is not None else time.perf_counter()
        return end - self.started_at

    @property
    def input_tokens(self) -> int:
        return sum(c.input_tokens for c in self.model_calls)

    @property
    def output_tokens(self) -> int:
        return sum(c.output_tokens for c in self.model_calls)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def num_model_calls(self) -> int:
        return len(self.model_calls)

    @property
    def num_tool_calls(self) -> int:
        return len(self.tool_calls)

    @property
    def num_tool_errors(self) -> int:
        return sum(1 for c in self.tool_calls if c.is_error)

    @property
    def estimated_cost(self) -> float | None:
        model = self.model_calls[0].model.lower() if self.model_calls else ""
        prices = next((p for key, p in _PRICES.items() if key in model), None)
        if prices is None:
            return None
        in_price, out_price = prices
        return (self.input_tokens * in_price + self.output_tokens * out_price) / 1_000_000
