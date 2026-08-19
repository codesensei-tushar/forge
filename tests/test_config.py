"""Tests for configuration loading.

The precedence chain is the contract::

    CLI overrides > environment > ./forge.toml > ~/.config/forge/config.toml > defaults

Every test runs with a scrubbed environment and a fake user-config directory, so
the developer's own shell and ``~/.config/forge`` cannot change the outcome.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from forge.config import (
    DEFAULT_DENY_PATTERNS,
    DEFAULT_DESTRUCTIVE_PATTERNS,
    ApprovalMode,
    SandboxMode,
    Settings,
    load_settings,
    user_config_path,
)


@pytest.fixture(autouse=True)
def isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """No inherited FORGE_*/ANTHROPIC_* variables and no real user config."""
    for key in list(os.environ):
        if key.startswith(("FORGE_", "ANTHROPIC_")):
            monkeypatch.delenv(key, raising=False)
    xdg = tmp_path / "xdg"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    return xdg


def write_user_config(xdg: Path, body: str) -> None:
    path = xdg / "forge" / "config.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)


# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #
def test_defaults(workspace: Path) -> None:
    settings = load_settings(workspace=workspace)

    assert settings.provider == "anthropic"
    assert settings.model is None
    assert settings.approval_mode is ApprovalMode.AUTO
    assert settings.sandbox is SandboxMode.NONE
    assert settings.sandbox_network is False
    assert settings.max_iterations == 30
    assert settings.enable_git_tools is True
    assert settings.log_level == "WARNING"
    assert settings.workspace_root == workspace


def test_default_rules_are_the_built_in_guards() -> None:
    settings = Settings()
    assert settings.deny == list(DEFAULT_DENY_PATTERNS)
    assert settings.destructive == list(DEFAULT_DESTRUCTIVE_PATTERNS)
    assert settings.allow == []


def test_git_push_is_denied_out_of_the_box() -> None:
    assert "git push" in Settings().deny


# --------------------------------------------------------------------------- #
# Precedence
# --------------------------------------------------------------------------- #
def test_workspace_toml_beats_the_user_config(workspace: Path, isolated_env: Path) -> None:
    write_user_config(isolated_env, 'model = "user-model"\nmax_tokens = 111\n')
    (workspace / "forge.toml").write_text('model = "repo-model"\n')

    settings = load_settings(workspace=workspace)

    assert settings.model == "repo-model"
    assert settings.max_tokens == 111, "unrelated user-config keys still apply"


def test_env_beats_the_workspace_toml(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (workspace / "forge.toml").write_text('model = "repo-model"\n')
    monkeypatch.setenv("FORGE_MODEL", "env-model")

    assert load_settings(workspace=workspace).model == "env-model"


def test_cli_beats_the_environment(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FORGE_MODEL", "env-model")
    settings = load_settings(workspace=workspace, cli_overrides={"model": "cli-model"})
    assert settings.model == "cli-model"


def test_unset_cli_overrides_do_not_erase_lower_layers(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The CLI passes None for every flag the user did not type."""
    monkeypatch.setenv("FORGE_MODEL", "env-model")
    settings = load_settings(
        workspace=workspace, cli_overrides={"model": None, "max_iterations": None}
    )
    assert settings.model == "env-model"
    assert settings.max_iterations == 30


def test_forge_env_beats_the_anthropic_env(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_MODEL", "vendor-model")
    monkeypatch.setenv("FORGE_MODEL", "forge-model")
    assert load_settings(workspace=workspace).model == "forge-model"


def test_anthropic_env_names_work_unassisted(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Forge should run against an existing gateway with zero extra config."""
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-test")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://gateway.example/")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "token-123")

    settings = load_settings(workspace=workspace)

    assert settings.model == "claude-test"
    assert settings.base_url == "https://gateway.example/"
    assert settings.auth_token == "token-123"


def test_empty_environment_values_are_ignored(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FORGE_MODEL", "")
    monkeypatch.setenv("ANTHROPIC_MODEL", "fallback")
    assert load_settings(workspace=workspace).model == "fallback"


# --------------------------------------------------------------------------- #
# TOML handling
# --------------------------------------------------------------------------- #
def test_forge_section_is_accepted(workspace: Path) -> None:
    (workspace / "forge.toml").write_text('[forge]\nmodel = "sectioned"\nshell_timeout = 45\n')
    settings = load_settings(workspace=workspace)
    assert settings.model == "sectioned"
    assert settings.shell_timeout == 45


def test_flat_toml_is_accepted(workspace: Path) -> None:
    (workspace / "forge.toml").write_text('model = "flat"\n')
    assert load_settings(workspace=workspace).model == "flat"


def test_unreadable_toml_is_reported(workspace: Path) -> None:
    (workspace / "forge.toml").write_text("this is not = = toml\n")
    with pytest.raises(ValueError, match="Could not read config"):
        load_settings(workspace=workspace)


def test_unknown_toml_keys_are_ignored(workspace: Path) -> None:
    """A stale key in someone's config should not stop the agent from starting."""
    (workspace / "forge.toml").write_text('model = "m"\nnot_a_setting = 1\n')
    assert load_settings(workspace=workspace).model == "m"


def test_missing_toml_is_fine(workspace: Path) -> None:
    assert not (workspace / "forge.toml").exists()
    assert load_settings(workspace=workspace).model is None


# --------------------------------------------------------------------------- #
# Typed environment coercion
# --------------------------------------------------------------------------- #
def test_int_float_and_bool_env_values(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FORGE_MAX_ITERATIONS", "7")
    monkeypatch.setenv("FORGE_SHELL_TIMEOUT", "15")
    monkeypatch.setenv("FORGE_TEMPERATURE", "0.4")
    monkeypatch.setenv("FORGE_SANDBOX_CPUS", "3.5")
    monkeypatch.setenv("FORGE_LOG_JSON", "yes")
    monkeypatch.setenv("FORGE_ENABLE_GIT_TOOLS", "off")

    settings = load_settings(workspace=workspace)

    assert settings.max_iterations == 7
    assert settings.shell_timeout == 15
    assert settings.temperature == 0.4
    assert settings.sandbox_cpus == 3.5
    assert settings.log_json is True
    assert settings.enable_git_tools is False


@pytest.mark.parametrize("raw", ["1", "true", "TRUE", "yes", "on"])
def test_truthy_spellings(workspace: Path, monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
    monkeypatch.setenv("FORGE_LOG_JSON", raw)
    assert load_settings(workspace=workspace).log_json is True


@pytest.mark.parametrize("raw", ["0", "false", "no", "off", "maybe"])
def test_falsy_spellings(workspace: Path, monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
    monkeypatch.setenv("FORGE_LOG_JSON", raw)
    assert load_settings(workspace=workspace).log_json is False


def test_enum_valued_env_vars(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FORGE_APPROVAL_MODE", "cautious")
    monkeypatch.setenv("FORGE_SANDBOX", "docker")

    settings = load_settings(workspace=workspace)

    assert settings.approval_mode is ApprovalMode.CAUTIOUS
    assert settings.sandbox is SandboxMode.DOCKER


def test_legacy_auto_approve_means_yolo(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FORGE_AUTO_APPROVE", "1")
    settings = load_settings(workspace=workspace)
    assert settings.approval_mode is ApprovalMode.YOLO
    assert settings.auto_approve is True


def test_legacy_auto_approve_off_changes_nothing(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FORGE_AUTO_APPROVE", "0")
    assert load_settings(workspace=workspace).approval_mode is ApprovalMode.AUTO


def test_log_level_is_normalized(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FORGE_LOG_LEVEL", "debug")
    assert load_settings(workspace=workspace).log_level == "DEBUG"


# --------------------------------------------------------------------------- #
# Validators
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(("given", "expected"), [(-1.0, 0.0), (0.0, 0.0), (0.5, 0.5), (9.0, 1.0)])
def test_temperature_is_clamped(given: float, expected: float) -> None:
    assert Settings(temperature=given).temperature == expected


def test_workspace_root_is_expanded_and_resolved(workspace: Path) -> None:
    (workspace / "sub").mkdir()
    settings = Settings(workspace_root=workspace / "sub" / ".." / "sub")
    assert settings.workspace_root == (workspace / "sub").resolve()


def test_log_level_is_upper_cased_on_the_model() -> None:
    assert Settings(log_level="info").log_level == "INFO"


# --------------------------------------------------------------------------- #
# Rule merging
# --------------------------------------------------------------------------- #
def test_deny_and_destructive_extend_the_defaults(workspace: Path) -> None:
    (workspace / "forge.toml").write_text(
        'deny = ["terraform destroy"]\ndestructive = ["alembic downgrade"]\n'
    )
    settings = load_settings(workspace=workspace)

    assert "terraform destroy" in settings.deny
    assert "alembic downgrade" in settings.destructive
    assert set(DEFAULT_DENY_PATTERNS) <= set(settings.deny)
    assert set(DEFAULT_DESTRUCTIVE_PATTERNS) <= set(settings.destructive)


def test_allow_rules_replace_rather_than_extend(workspace: Path) -> None:
    """There is nothing to protect in the allow list — it starts empty."""
    (workspace / "forge.toml").write_text('allow = ["pytest"]\n')
    assert load_settings(workspace=workspace).allow == ["pytest"]


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
def test_user_config_path_follows_xdg(isolated_env: Path) -> None:
    assert user_config_path() == isolated_env / "forge" / "config.toml"


def test_user_config_path_falls_back_to_home(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    assert user_config_path() == Path.home() / ".config" / "forge" / "config.toml"


def test_workspace_is_a_runtime_fact_not_a_stored_one(workspace: Path, isolated_env: Path) -> None:
    """The active workspace comes from the invocation, not from a config file."""
    write_user_config(isolated_env, "max_tokens = 4096\n")
    assert load_settings(workspace=workspace).workspace_root == workspace
