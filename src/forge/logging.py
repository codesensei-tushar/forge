"""Structured logging via structlog.

A pretty, human-readable renderer is used for interactive terminals; set
``log_json=True`` (or ``FORGE_LOG_JSON=1``) to emit newline-delimited JSON
suitable for later ingestion into an observability backend (Phase 7).
"""

from __future__ import annotations

import logging
import os
import sys

import structlog

_configured = False


def configure_logging(level: str = "INFO", *, json: bool | None = None) -> None:
    """Configure structlog once per process."""
    global _configured
    if _configured:
        return

    if json is None:
        json = os.environ.get("FORGE_LOG_JSON", "").lower() in {"1", "true", "yes", "on"}

    log_level = getattr(logging, level.upper(), logging.INFO)

    # Route stdlib logging (e.g. from the anthropic SDK) to stderr so it never
    # corrupts tool output on stdout.
    logging.basicConfig(format="%(message)s", stream=sys.stderr, level=log_level)

    shared: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]
    renderer: structlog.types.Processor = (
        structlog.processors.JSONRenderer()
        if json
        else structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())
    )

    structlog.configure(
        processors=[*shared, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        logger_factory=structlog.PrintLoggerFactory(),
        # Resolve sys.stderr at write time rather than caching a logger bound to
        # whatever the stream happened to be at configure time. A cached logger
        # keeps writing to a stream that may since have been replaced or closed
        # (CliRunner, capture fixtures, embedding Forge in another process). The
        # cost is negligible at the handful of events an agent step emits.
        cache_logger_on_first_use=False,
    )
    _configured = True


def get_logger(name: str = "forge") -> structlog.types.FilteringBoundLogger:
    """Return a bound structlog logger."""
    # structlog.get_logger is deliberately untyped (the bound class is chosen at
    # configure time); name the class we actually configure above.
    logger: structlog.types.FilteringBoundLogger = structlog.get_logger(name)
    return logger
