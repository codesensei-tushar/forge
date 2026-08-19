"""Per-run tracing: model calls, tool calls, tokens, cost, and duration.

Pure data plus derived metrics — no rendering. The UI layer formats it for a
terminal; :meth:`RunTrace.to_dict` serializes it for ``--json`` and for later
ingestion into an observability backend.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import Any

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
    retries: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency_s": round(self.latency_s, 3),
            "stop_reason": self.stop_reason,
            "retries": self.retries,
        }


@dataclass
class ToolCall:
    name: str
    decision: str
    duration_s: float
    is_error: bool
    risk: str = "write"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "decision": self.decision,
            "risk": self.risk,
            "duration_s": round(self.duration_s, 3),
            "is_error": self.is_error,
        }


@dataclass
class RunTrace:
    task: str = ""
    run_id: int = field(default_factory=lambda: next(_run_counter))
    started_at: float = field(default_factory=time.perf_counter)
    ended_at: float | None = None
    status: str = "running"
    model_calls: list[ModelCall] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    compactions: int = 0

    # --- recording ---
    def record_model_call(
        self,
        *,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency_s: float,
        stop_reason: str,
        retries: int = 0,
    ) -> None:
        self.model_calls.append(
            ModelCall(model, input_tokens, output_tokens, latency_s, stop_reason, retries)
        )

    def record_tool_call(
        self,
        *,
        name: str,
        decision: str,
        duration_s: float,
        is_error: bool,
        risk: str = "write",
    ) -> None:
        self.tool_calls.append(ToolCall(name, decision, duration_s, is_error, risk))

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
    def num_retries(self) -> int:
        return sum(c.retries for c in self.model_calls)

    @property
    def model(self) -> str:
        return self.model_calls[0].model if self.model_calls else ""

    @property
    def estimated_cost(self) -> float | None:
        model = self.model.lower()
        prices = next((p for key, p in _PRICES.items() if key in model), None)
        if prices is None:
            return None
        in_price, out_price = prices
        return (self.input_tokens * in_price + self.output_tokens * out_price) / 1_000_000

    def tool_usage_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for call in self.tool_calls:
            counts[call.name] = counts.get(call.name, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))

    def to_dict(self) -> dict[str, Any]:
        """A JSON-serializable snapshot of the run."""
        return {
            "run_id": self.run_id,
            "task": self.task,
            "status": self.status,
            "model": self.model,
            "duration_s": round(self.duration_s, 3),
            "compactions": self.compactions,
            "usage": {
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "total_tokens": self.total_tokens,
                "estimated_cost_usd": (
                    round(self.estimated_cost, 6) if self.estimated_cost is not None else None
                ),
            },
            "counts": {
                "model_calls": self.num_model_calls,
                "tool_calls": self.num_tool_calls,
                "tool_errors": self.num_tool_errors,
                "provider_retries": self.num_retries,
            },
            "tools_used": self.tool_usage_counts(),
            "model_calls": [c.to_dict() for c in self.model_calls],
            "tool_calls": [c.to_dict() for c in self.tool_calls],
        }
