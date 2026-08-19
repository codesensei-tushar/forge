"""Tests for the command-line interface.

The behaviour that matters is the non-interactive surface: a bare task works as
the first argument, ``--json`` emits a single machine-readable payload, and the
exit code reports the outcome. Every test runs against the ``fake`` provider, so
nothing here touches a network or a real model.
"""

from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
from typing import Any

import pytest
import typer
from typer.testing import CliRunner

from forge import __version__
from forge.cli import COMMANDS, _insert_run_command, app, main

runner = CliRunner()


@pytest.fixture(autouse=True)
def isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Scrub inherited configuration so the CLI's own defaults are what we see."""
    for key in list(os.environ):
        if key.startswith(("FORGE_", "ANTHROPIC_")):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))


def parse_json(output: str) -> Any:
    starts = [i for i in (output.find("{"), output.find("[")) if i >= 0]
    assert starts, f"no JSON payload in output:\n{output}"
    return json.loads(output[min(starts) :])


def invoke(*args: str) -> Any:
    return runner.invoke(app, list(args))


def output_of(result: Any) -> str:
    """Everything the invocation printed, wherever Click routed it."""
    parts = [result.output]
    with contextlib.suppress(ValueError):
        parts.append(result.stderr)
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# argv preprocessing: `forge "task"` must work without a subcommand
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        ([], []),
        (["run", "fix it"], ["run", "fix it"]),
        (["repl"], ["repl"]),
        (["config"], ["config"]),
        (["version"], ["version"]),
        (["fix the failing tests"], ["run", "fix the failing tests"]),
        (["--json", "analyze this"], ["--json", "run", "analyze this"]),
        (["--verbose", "-q", "do it"], ["--verbose", "-q", "run", "do it"]),
        (["--model", "claude-x", "do it"], ["--model", "claude-x", "run", "do it"]),
        (["-C", "/tmp/proj", "do it"], ["-C", "/tmp/proj", "run", "do it"]),
        (["--model", "claude-x", "tools"], ["--model", "claude-x", "tools"]),
        (["--version"], ["--version"]),
        (["--yes", "fix it"], ["--yes", "run", "fix it"]),
        (["--mode", "yolo", "repl"], ["--mode", "yolo", "repl"]),
    ],
)
def test_insert_run_command(argv: list[str], expected: list[str]) -> None:
    assert _insert_run_command(argv) == expected


def test_insert_run_command_does_not_mutate_its_input() -> None:
    argv = ["--json", "a task"]
    _insert_run_command(argv)
    assert argv == ["--json", "a task"]


def test_commands_matches_what_typer_actually_registered() -> None:
    """A name missing from COMMANDS would be silently swallowed as a task."""
    registered = set(typer.main.get_command(app).commands)  # type: ignore[attr-defined]
    assert registered == set(COMMANDS)


# --------------------------------------------------------------------------- #
# version
# --------------------------------------------------------------------------- #
def test_version_flag() -> None:
    result = invoke("--version")
    assert result.exit_code == 0
    assert f"Forge {__version__}" in result.output


def test_version_command() -> None:
    result = invoke("version")
    assert result.exit_code == 0
    assert f"Forge {__version__}" in result.output


def test_main_wires_argv_to_the_app(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert f"Forge {__version__}" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# tools
# --------------------------------------------------------------------------- #
def test_tools_json_lists_every_tool(workspace: Path) -> None:
    result = invoke("-C", str(workspace), "--json", "tools")
    assert result.exit_code == 0

    entries = parse_json(result.output)
    names = {entry["name"] for entry in entries}
    assert {"read_file", "write_file", "edit_file", "shell", "git_status"} <= names
    assert not any("push" in name for name in names)
    for entry in entries:
        assert entry["risk"] in {"read", "write", "destructive"}
        assert entry["description"]


def test_tools_table_renders(workspace: Path) -> None:
    result = invoke("-C", str(workspace), "tools")
    assert result.exit_code == 0
    assert "read_file" in result.output


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
def test_config_json_reports_the_resolved_settings(workspace: Path) -> None:
    result = invoke("-C", str(workspace), "--json", "config")
    assert result.exit_code == 0

    payload = parse_json(result.output)
    assert payload["workspace_root"] == str(workspace)
    assert payload["approval_mode"] == "auto"
    assert payload["sandbox"] == "none"
    assert payload["credential"] == "MISSING"
    assert payload["model"] == "(unset)"


def test_config_redacts_the_credential(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The token itself must never be printed, only whether one is present."""
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "sk-do-not-print-me")

    result = invoke("-C", str(workspace), "--json", "config")

    assert parse_json(result.output)["credential"] == "set"
    assert "sk-do-not-print-me" not in result.output


def test_cli_flags_reach_the_resolved_config(workspace: Path) -> None:
    result = invoke(
        "-C",
        str(workspace),
        "--json",
        "--yes",
        "--max-iterations",
        "5",
        "--model",
        "claude-x",
        "config",
    )
    payload = parse_json(result.output)
    assert payload["approval_mode"] == "yolo"
    assert payload["max_iterations"] == "5"
    assert payload["model"] == "claude-x"


def test_cautious_flag(workspace: Path) -> None:
    result = invoke("-C", str(workspace), "--json", "--cautious", "config")
    assert parse_json(result.output)["approval_mode"] == "cautious"


def test_mode_flag(workspace: Path) -> None:
    result = invoke("-C", str(workspace), "--json", "--mode", "CAUTIOUS", "config")
    assert parse_json(result.output)["approval_mode"] == "cautious"


def test_bad_mode_is_rejected(workspace: Path) -> None:
    result = invoke("-C", str(workspace), "--mode", "nonsense", "config")
    assert result.exit_code != 0
    assert "yolo" in output_of(result), "the error should list the valid modes"


def test_bad_sandbox_is_rejected(workspace: Path) -> None:
    result = invoke("-C", str(workspace), "--sandbox", "podman", "config")
    assert result.exit_code != 0
    assert "docker" in output_of(result)


def test_unreadable_config_exits_misconfigured(workspace: Path) -> None:
    (workspace / "forge.toml").write_text("= not toml\n")
    result = invoke("-C", str(workspace), "config")
    assert result.exit_code == 2


# --------------------------------------------------------------------------- #
# run (one-shot)
# --------------------------------------------------------------------------- #
def test_bare_task_runs_and_reports_json(workspace: Path) -> None:
    result = runner.invoke(
        app,
        _insert_run_command(["-C", str(workspace), "--json", "--provider", "fake", "say hello"]),
    )
    assert result.exit_code == 0

    payload = parse_json(result.output)
    assert payload["status"] == "completed"
    assert payload["ok"] is True
    assert payload["iterations"] == 1
    assert "fake" in payload["response"]
    assert payload["trace"]["task"] == "say hello"


def test_run_reads_the_task_from_stdin(workspace: Path) -> None:
    result = runner.invoke(
        app,
        ["-C", str(workspace), "--json", "--provider", "fake", "run"],
        input="audit the config loader\n",
    )
    assert result.exit_code == 0
    assert parse_json(result.output)["trace"]["task"] == "audit the config loader"


def test_run_without_a_task_is_misconfigured(workspace: Path) -> None:
    result = runner.invoke(app, ["-C", str(workspace), "--provider", "fake", "run"], input="")
    assert result.exit_code == 2


def test_missing_model_for_anthropic_exits_misconfigured(workspace: Path) -> None:
    """No credential and no model is a configuration problem, not a crash."""
    result = invoke("-C", str(workspace), "--provider", "anthropic", "run", "do it")
    assert result.exit_code == 2
    assert "No model configured" in output_of(result)


def test_max_iterations_is_honoured(workspace: Path) -> None:
    """A ceiling of zero can do no work, which proves the flag is threaded through."""
    result = invoke(
        "-C",
        str(workspace),
        "--json",
        "--provider",
        "fake",
        "--max-iterations",
        "0",
        "run",
        "spin forever",
    )
    payload = parse_json(result.output)
    assert payload["status"] == "max_iterations"
    assert payload["ok"] is False
    assert result.exit_code == 1
