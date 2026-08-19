"""Tests for the sandbox layer.

The local backend is exercised for real; the Docker backend is exercised through
its argv construction, because asserting on the flags is the actual contract —
running a container in unit tests would be slow and machine-dependent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from forge.config import SandboxMode, Settings
from forge.sandbox import create_sandbox
from forge.sandbox.base import ExecRequest, ExecResult, Sandbox, SandboxError, scrub_env
from forge.sandbox.docker import DockerSandbox
from forge.sandbox.local import LocalSandbox


def request_for(command: str, cwd: Path, *, timeout: int = 10, **env: str) -> ExecRequest:
    return ExecRequest(command=command, cwd=cwd, timeout=timeout, env=dict(env))


# --------------------------------------------------------------------------- #
# scrub_env
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "key",
    [
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "OPENAI_API_KEY",
        "AWS_SECRET_ACCESS_KEY",
        "GITHUB_TOKEN",
        "GH_TOKEN",
        "MY_PASSWORD",
        "DB_PASSWD",
        "SOME_CREDENTIAL",
        "SESSION_ID",
        "HTTP_COOKIE",
        "npm_token",
        "Stripe_Key",
    ],
)
def test_secret_looking_keys_are_removed(key: str) -> None:
    assert key not in scrub_env({key: "sensitive", "PATH": "/usr/bin"})


@pytest.mark.parametrize("key", ["PATH", "HOME", "LANG", "TERM", "CI", "PYTHONPATH", "VIRTUAL_ENV"])
def test_ordinary_keys_survive(key: str) -> None:
    assert scrub_env({key: "value"}) == {key: "value"}


def test_scrub_env_defaults_to_the_process_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FORGE_SCRUB_PROBE", "visible")
    monkeypatch.setenv("FORGE_SCRUB_SECRET", "hidden")
    scrubbed = scrub_env()
    assert scrubbed["FORGE_SCRUB_PROBE"] == "visible"
    assert "FORGE_SCRUB_SECRET" not in scrubbed


def test_scrub_env_does_not_mutate_its_input() -> None:
    source = {"API_KEY": "x", "PATH": "/bin"}
    scrub_env(source)
    assert "API_KEY" in source


# --------------------------------------------------------------------------- #
# ExecResult
# --------------------------------------------------------------------------- #
def test_exec_result_ok_requires_everything_to_go_right() -> None:
    assert ExecResult(exit_code=0, stdout="", stderr="").ok
    assert not ExecResult(exit_code=1, stdout="", stderr="").ok
    assert not ExecResult(exit_code=0, stdout="", stderr="", timed_out=True).ok
    assert not ExecResult(exit_code=0, stdout="", stderr="", startup_error="nope").ok


def test_sandbox_is_abstract() -> None:
    with pytest.raises(TypeError):
        Sandbox()  # type: ignore[abstract]


# --------------------------------------------------------------------------- #
# LocalSandbox
# --------------------------------------------------------------------------- #
async def test_local_runs_a_command(workspace: Path) -> None:
    result = await LocalSandbox().exec(request_for("echo hi", workspace))
    assert result.ok and result.stdout.strip() == "hi"
    assert result.duration_s >= 0


async def test_local_reports_a_non_zero_exit(workspace: Path) -> None:
    result = await LocalSandbox().exec(request_for("exit 7", workspace))
    assert result.exit_code == 7 and not result.ok


async def test_local_captures_stderr(workspace: Path) -> None:
    result = await LocalSandbox().exec(request_for("echo bad >&2", workspace))
    assert result.stderr.strip() == "bad"


async def test_local_runs_in_the_requested_directory(workspace: Path) -> None:
    (workspace / "here").mkdir()
    result = await LocalSandbox().exec(request_for("pwd", workspace / "here"))
    assert result.stdout.strip().endswith("here")


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")
async def test_local_times_out_and_kills_the_group(workspace: Path) -> None:
    result = await LocalSandbox().exec(request_for("sleep 30", workspace, timeout=1))
    assert result.timed_out
    assert result.exit_code == 124
    assert not result.ok


async def test_local_reports_a_startup_failure(workspace: Path) -> None:
    result = await LocalSandbox().exec(request_for("true", workspace / "does-not-exist"))
    assert result.startup_error is not None
    assert result.exit_code == -1
    assert "Failed to start command" in result.startup_error


async def test_local_strips_secrets_by_default(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SANDBOX_TEST_TOKEN", "leak")
    result = await LocalSandbox().exec(request_for("env", workspace))
    assert "SANDBOX_TEST_TOKEN" not in result.stdout


async def test_local_can_be_told_not_to_strip(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SANDBOX_TEST_TOKEN", "leak")
    result = await LocalSandbox(scrub_secrets=False).exec(request_for("env", workspace))
    assert "SANDBOX_TEST_TOKEN" in result.stdout


async def test_explicit_env_reaches_the_command(workspace: Path) -> None:
    result = await LocalSandbox().exec(request_for("echo $GREETING", workspace, GREETING="ahoy"))
    assert result.stdout.strip() == "ahoy"


async def test_local_neutralizes_pagers_and_colour(workspace: Path) -> None:
    result = await LocalSandbox().exec(request_for("echo $TERM/$GIT_PAGER/$NO_COLOR", workspace))
    assert result.stdout.strip() == "dumb/cat/1"


def test_local_describes_its_limits() -> None:
    assert "credentials stripped" in LocalSandbox().describe()
    assert "full environment" in LocalSandbox(scrub_secrets=False).describe()
    assert LocalSandbox().name == "local"


async def test_local_aclose_is_a_noop() -> None:
    assert await LocalSandbox().aclose() is None


# --------------------------------------------------------------------------- #
# DockerSandbox
# --------------------------------------------------------------------------- #
@pytest.fixture
def fake_docker(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend the docker binary exists, without needing a daemon."""
    monkeypatch.setattr("forge.sandbox.docker.docker_available", lambda: True)


def test_docker_requires_the_binary(monkeypatch: pytest.MonkeyPatch, workspace: Path) -> None:
    """An unavailable backend is an error, never a silent downgrade to the host."""
    monkeypatch.setattr("forge.sandbox.docker.docker_available", lambda: False)
    with pytest.raises(SandboxError, match="not found on PATH"):
        DockerSandbox(workspace_root=workspace)


def test_docker_argv_applies_every_limit(fake_docker: None, workspace: Path) -> None:
    sandbox = DockerSandbox(
        workspace_root=workspace, image="python:3.12-slim", cpus=1.5, memory="512m", pids_limit=64
    )
    argv = sandbox._docker_argv(request_for("pytest -q", workspace))

    assert argv[:3] == ["docker", "run", "--rm"]
    assert f"--volume={workspace}:/workspace" in argv
    assert "--workdir=/workspace" in argv
    assert "--cpus=1.5" in argv
    assert "--memory=512m" in argv
    assert "--pids-limit=64" in argv
    assert "--cap-drop=ALL" in argv
    assert "--security-opt=no-new-privileges" in argv
    assert argv[-4:] == ["python:3.12-slim", "/bin/sh", "-c", "pytest -q"]


def test_docker_disables_the_network_by_default(fake_docker: None, workspace: Path) -> None:
    assert "--network=none" in DockerSandbox(workspace_root=workspace)._docker_argv(
        request_for("true", workspace)
    )


def test_docker_network_can_be_enabled(fake_docker: None, workspace: Path) -> None:
    sandbox = DockerSandbox(workspace_root=workspace, network=True)
    assert "--network=none" not in sandbox._docker_argv(request_for("true", workspace))


def test_docker_maps_a_subdirectory_cwd(fake_docker: None, workspace: Path) -> None:
    (workspace / "pkg" / "sub").mkdir(parents=True)
    sandbox = DockerSandbox(workspace_root=workspace)
    argv = sandbox._docker_argv(request_for("true", workspace / "pkg" / "sub"))
    assert "--workdir=/workspace/pkg/sub" in argv


def test_docker_clamps_an_outside_cwd_to_the_mount(fake_docker: None, workspace: Path) -> None:
    sandbox = DockerSandbox(workspace_root=workspace)
    argv = sandbox._docker_argv(request_for("true", Path("/etc")))
    assert "--workdir=/workspace" in argv


def test_docker_forwards_request_env(fake_docker: None, workspace: Path) -> None:
    sandbox = DockerSandbox(workspace_root=workspace)
    argv = sandbox._docker_argv(request_for("true", workspace, GIT_TERMINAL_PROMPT="0"))
    assert "--env=GIT_TERMINAL_PROMPT=0" in argv
    assert "--env=TERM=dumb" in argv


def test_docker_describes_its_limits(fake_docker: None, workspace: Path) -> None:
    described = DockerSandbox(workspace_root=workspace, cpus=2.0, memory="2g").describe()
    assert "cpus=2.0" in described
    assert "memory=2g" in described
    assert "network=disabled" in described


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def test_factory_defaults_to_local(workspace: Path) -> None:
    sandbox = create_sandbox(Settings(workspace_root=workspace, sandbox=SandboxMode.NONE))
    assert isinstance(sandbox, LocalSandbox)


def test_factory_builds_docker_with_the_configured_limits(
    fake_docker: None, workspace: Path
) -> None:
    sandbox = create_sandbox(
        Settings(
            workspace_root=workspace,
            sandbox=SandboxMode.DOCKER,
            sandbox_image="custom:latest",
            sandbox_cpus=4.0,
            sandbox_memory="8g",
            sandbox_pids_limit=128,
        )
    )
    assert isinstance(sandbox, DockerSandbox)
    assert sandbox.image == "custom:latest"
    assert (sandbox.cpus, sandbox.memory, sandbox.pids_limit) == (4.0, "8g", 128)
    assert sandbox.network is False


def test_factory_surfaces_a_missing_backend(
    monkeypatch: pytest.MonkeyPatch, workspace: Path
) -> None:
    monkeypatch.setattr("forge.sandbox.docker.docker_available", lambda: False)
    with pytest.raises(SandboxError):
        create_sandbox(Settings(workspace_root=workspace, sandbox=SandboxMode.DOCKER))
