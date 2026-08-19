"""Configuration loading with explicit, predictable precedence.

Precedence (highest wins)::

    CLI overrides  >  environment  >  ./forge.toml  >  ~/.config/forge/config.toml  >  defaults

Environment variables are read from ``FORGE_*`` and, for provider credentials,
the standard ``ANTHROPIC_*`` names so the agent runs against an existing
Anthropic-compatible gateway with zero extra configuration.
"""

from __future__ import annotations

import os
import tomllib
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator


class ApprovalMode(StrEnum):
    """How much of the agent's work runs without a human in the loop.

    Read operations are always automatic; the mode governs the rest.
    """

    CAUTIOUS = "cautious"  # writes and destructive actions both need approval
    AUTO = "auto"  # writes are automatic; destructive actions need approval
    YOLO = "yolo"  # everything runs unattended (deny rules still apply)


class SandboxMode(StrEnum):
    NONE = "none"  # run shell commands directly on the host
    DOCKER = "docker"  # run shell commands in a resource-capped container


# Shell command substrings that are always denied, regardless of allow-list.
DEFAULT_DENY_PATTERNS: tuple[str, ...] = (
    "rm -rf /",
    "rm -rf ~",
    "rm -rf .",
    "sudo ",
    "git push",
    "mkfs",
    "dd if=",
    ":(){",  # fork bomb
    "shutdown",
    "reboot",
    "> /dev/sd",
    "chmod -R 777 /",
)

# Shell command substrings treated as destructive: allowed, but they require
# approval even in ``auto`` mode because they discard work or mutate the host.
DEFAULT_DESTRUCTIVE_PATTERNS: tuple[str, ...] = (
    "rm -r",
    "rm -f",
    "rmdir",
    "git reset --hard",
    "git clean",
    "git checkout --",
    "git restore",
    "git rebase",
    "git filter-branch",
    "git remote",
    "git tag -d",
    "git branch -D",
    "truncate",
    "shred",
    "mv /",
    "chown",
    "chmod -R",
    "kill -9",
    "pkill",
    "systemctl",
    "docker rm",
    "docker rmi",
    "npm publish",
    "pip uninstall",
    "curl | sh",
    "| sh",
    "| bash",
)


class Settings(BaseModel):
    """Fully-resolved runtime configuration for a Forge session."""

    # --- model / provider ---
    provider: str = "anthropic"
    model: str | None = None
    base_url: str | None = None
    auth_token: str | None = None
    api_key: str | None = None
    max_tokens: int = 8192
    temperature: float = 0.0
    request_timeout: float = 600.0

    # --- agent loop ---
    max_iterations: int = 30
    # Soft ceiling on estimated context tokens; older tool output is compacted
    # away when the history crosses it, so long runs degrade instead of dying.
    max_context_tokens: int = 150_000
    # Retries for *transient* provider failures (429/5xx/timeouts) per step.
    max_provider_retries: int = 3
    retry_base_delay: float = 1.0

    # --- tools ---
    workspace_root: Path = Field(default_factory=Path.cwd)
    shell_timeout: int = 120
    tool_timeout: int = 300
    enable_git_tools: bool = True

    # --- permissions ---
    approval_mode: ApprovalMode = ApprovalMode.AUTO
    allow: list[str] = Field(default_factory=list)
    deny: list[str] = Field(default_factory=lambda: list(DEFAULT_DENY_PATTERNS))
    destructive: list[str] = Field(default_factory=lambda: list(DEFAULT_DESTRUCTIVE_PATTERNS))

    # --- sandbox ---
    sandbox: SandboxMode = SandboxMode.NONE
    sandbox_image: str = "python:3.12-slim"
    sandbox_cpus: float = 2.0
    sandbox_memory: str = "2g"
    sandbox_network: bool = False
    sandbox_pids_limit: int = 512

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

    @field_validator("log_level")
    @classmethod
    def _upper_log_level(cls, v: str) -> str:
        return v.upper()

    @property
    def auto_approve(self) -> bool:
        """True when nothing requires a human (``yolo`` mode)."""
        return self.approval_mode is ApprovalMode.YOLO


def user_config_path() -> Path:
    """Return the per-user config file path (respects ``XDG_CONFIG_HOME``)."""
    base = os.environ.get("XDG_CONFIG_HOME")
    root = Path(base) if base else Path.home() / ".config"
    return root / "forge" / "config.toml"


def _read_toml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        with path.open("rb") as fh:
            data = tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ValueError(f"Could not read config {path}: {exc}") from exc
    # Accept either a flat table or a [forge] section.
    if "forge" in data and isinstance(data["forge"], dict):
        return dict(data["forge"])
    return dict(data)


def _as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


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

    str_keys = {
        "provider": ("FORGE_PROVIDER",),
        "model": ("FORGE_MODEL", "ANTHROPIC_MODEL"),
        "base_url": ("FORGE_BASE_URL", "ANTHROPIC_BASE_URL"),
        "auth_token": ("FORGE_AUTH_TOKEN", "ANTHROPIC_AUTH_TOKEN"),
        "api_key": ("FORGE_API_KEY", "ANTHROPIC_API_KEY"),
        "approval_mode": ("FORGE_APPROVAL_MODE",),
        "sandbox": ("FORGE_SANDBOX",),
        "sandbox_image": ("FORGE_SANDBOX_IMAGE",),
        "sandbox_memory": ("FORGE_SANDBOX_MEMORY",),
    }
    int_keys = {
        "max_tokens": ("FORGE_MAX_TOKENS",),
        "max_iterations": ("FORGE_MAX_ITERATIONS",),
        "max_context_tokens": ("FORGE_MAX_CONTEXT_TOKENS",),
        "shell_timeout": ("FORGE_SHELL_TIMEOUT",),
        "tool_timeout": ("FORGE_TOOL_TIMEOUT",),
    }
    float_keys = {
        "temperature": ("FORGE_TEMPERATURE",),
        "request_timeout": ("FORGE_REQUEST_TIMEOUT",),
        "sandbox_cpus": ("FORGE_SANDBOX_CPUS",),
    }
    bool_keys = {
        "log_json": ("FORGE_LOG_JSON",),
        "sandbox_network": ("FORGE_SANDBOX_NETWORK",),
        "enable_git_tools": ("FORGE_ENABLE_GIT_TOOLS",),
    }

    for key, names in str_keys.items():
        if (v := first(*names)) is not None:
            out[key] = v
    for key, names in int_keys.items():
        if (v := first(*names)) is not None:
            out[key] = int(v)
    for key, names in float_keys.items():
        if (v := first(*names)) is not None:
            out[key] = float(v)
    for key, names in bool_keys.items():
        if (v := first(*names)) is not None:
            out[key] = _as_bool(v)

    if (v := first("FORGE_LOG_LEVEL")) is not None:
        out["log_level"] = v.upper()
    # Legacy toggle: FORGE_AUTO_APPROVE=1 is shorthand for yolo mode.
    if (v := first("FORGE_AUTO_APPROVE")) is not None and _as_bool(v):
        out["approval_mode"] = ApprovalMode.YOLO.value
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

    # Deny and destructive patterns from config EXTEND the built-in guards
    # rather than replacing them, so a config file cannot weaken the defaults.
    if "deny" in merged:
        merged["deny"] = [*DEFAULT_DENY_PATTERNS, *merged["deny"]]
    if "destructive" in merged:
        merged["destructive"] = [*DEFAULT_DESTRUCTIVE_PATTERNS, *merged["destructive"]]

    return Settings(**merged)
