"""Configuration loading with explicit, predictable precedence.

Precedence (highest wins):

    CLI overrides  >  environment  >  ./forge.toml  >  ~/.config/forge/config.toml  >  defaults

Environment variables are read from ``FORGE_*`` and, for provider credentials,
the standard ``ANTHROPIC_*`` names so the agent runs against an existing
Anthropic-compatible gateway with zero extra configuration.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

# Tools that never mutate state and are always safe to auto-run.
READONLY_TOOLS: frozenset[str] = frozenset(
    {"read_file", "list_directory", "search_files"}
)

# Shell command substrings that are always denied, regardless of allow-list.
DEFAULT_DENY_PATTERNS: tuple[str, ...] = (
    "rm -rf /",
    "rm -rf ~",
    "rm -rf .",
    "sudo ",
    "git push",
    "git reset --hard",
    "mkfs",
    "dd if=",
    ":(){",  # fork bomb
    "shutdown",
    "reboot",
    "> /dev/sda",
)


class Settings(BaseModel):
    """Fully-resolved runtime configuration for a Forge session."""

    # --- model / provider ---
    provider: str = "anthropic"
    model: str | None = None
    base_url: str | None = None
    auth_token: str | None = None
    api_key: str | None = None
    max_tokens: int = 4096
    temperature: float = 0.0

    # --- agent loop ---
    max_iterations: int = 25
    # Soft ceiling on estimated context tokens; the loop stops cleanly if exceeded.
    max_context_tokens: int = 150_000

    # --- tools ---
    workspace_root: Path = Field(default_factory=Path.cwd)
    shell_timeout: int = 120

    # --- permissions ---
    auto_approve: bool = False
    allow: list[str] = Field(default_factory=list)
    deny: list[str] = Field(default_factory=lambda: list(DEFAULT_DENY_PATTERNS))

    # --- observability ---
    # Step events go to stderr; WARNING keeps the UI clean. --verbose raises this.
    log_level: str = "WARNING"
    log_json: bool = False

    @field_validator("workspace_root")
    @classmethod
    def _resolve_workspace(cls, v: Path) -> Path:
        return Path(v).expanduser().resolve()

    @field_validator("temperature")
    @classmethod
    def _clamp_temperature(cls, v: float) -> float:
        return max(0.0, min(1.0, v))


def user_config_path() -> Path:
    """Return the per-user config file path (respects ``XDG_CONFIG_HOME``)."""
    base = os.environ.get("XDG_CONFIG_HOME")
    root = Path(base) if base else Path.home() / ".config"
    return root / "forge" / "config.toml"


def _read_toml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    # Accept either a flat table or a [forge] section.
    if "forge" in data and isinstance(data["forge"], dict):
        return dict(data["forge"])
    return dict(data)


def _env_overrides() -> dict[str, Any]:
    """Read known settings from the environment.

    ``FORGE_*`` takes precedence over the provider-standard ``ANTHROPIC_*`` names.
    Only keys actually present in the environment are returned so they layer
    correctly over TOML values.
    """
    env = os.environ
    out: dict[str, Any] = {}

    def first(*names: str) -> str | None:
        for name in names:
            if name in env and env[name] != "":
                return env[name]
        return None

    if (v := first("FORGE_PROVIDER")) is not None:
        out["provider"] = v
    if (v := first("FORGE_MODEL", "ANTHROPIC_MODEL")) is not None:
        out["model"] = v
    if (v := first("FORGE_BASE_URL", "ANTHROPIC_BASE_URL")) is not None:
        out["base_url"] = v
    if (v := first("ANTHROPIC_AUTH_TOKEN")) is not None:
        out["auth_token"] = v
    if (v := first("ANTHROPIC_API_KEY")) is not None:
        out["api_key"] = v
    if (v := first("FORGE_MAX_TOKENS")) is not None:
        out["max_tokens"] = int(v)
    if (v := first("FORGE_TEMPERATURE")) is not None:
        out["temperature"] = float(v)
    if (v := first("FORGE_MAX_ITERATIONS")) is not None:
        out["max_iterations"] = int(v)
    if (v := first("FORGE_SHELL_TIMEOUT")) is not None:
        out["shell_timeout"] = int(v)
    if (v := first("FORGE_LOG_LEVEL")) is not None:
        out["log_level"] = v.upper()
    if (v := first("FORGE_AUTO_APPROVE")) is not None:
        out["auto_approve"] = v.lower() in {"1", "true", "yes", "on"}
    return out


def load_settings(
    *,
    workspace: Path | None = None,
    cli_overrides: dict[str, Any] | None = None,
) -> Settings:
    """Build a :class:`Settings` by layering all configuration sources."""
    workspace = (workspace or Path.cwd()).expanduser().resolve()
    merged: dict[str, Any] = {}

    # Lowest precedence first.
    merged.update(_read_toml(user_config_path()))
    merged.update(_read_toml(workspace / "forge.toml"))
    merged.update(_env_overrides())

    for key, value in (cli_overrides or {}).items():
        if value is not None:
            merged[key] = value

    # The active workspace is a runtime fact, not a stored preference.
    merged.setdefault("workspace_root", workspace)

    # Deny patterns from config extend, rather than replace, the built-in guards.
    if "deny" in merged:
        merged["deny"] = list(DEFAULT_DENY_PATTERNS) + list(merged["deny"])

    return Settings(**merged)
