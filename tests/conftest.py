"""Shared test fixtures.

Every fixture here is offline: the ``fake`` provider means no test ever reaches a
real API, and the workspace is a throwaway tmp_path so filesystem and git tools
have somewhere safe to write.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from forge.config import ApprovalMode, Settings
from forge.permissions.policy import Approval, PermissionPolicy, PermissionResult
from forge.tools import ToolExecutor, default_registry
from forge.tools.base import Tool
from forge.tools.context import ToolContext
from forge.tools.registry import ToolRegistry
from forge.ui.console import Console


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def settings(workspace: Path) -> Settings:
    return Settings(
        workspace_root=workspace,
        provider="fake",
        model="fake-model",
        approval_mode=ApprovalMode.YOLO,
        max_iterations=10,
        shell_timeout=20,
    )


@pytest.fixture
def ctx(settings: Settings) -> ToolContext:
    return ToolContext(settings)


@pytest.fixture
def quiet_console() -> Console:
    """A console that renders nothing, so test output stays readable."""
    return Console(quiet=True)


@pytest.fixture
def registry(settings: Settings) -> ToolRegistry:
    return default_registry(settings)


@pytest.fixture
def executor(settings: Settings, registry: ToolRegistry, ctx: ToolContext) -> ToolExecutor:
    return ToolExecutor(
        registry=registry,
        policy=PermissionPolicy(settings),
        ctx=ctx,
        settings=settings,
    )


@pytest.fixture
def approve_all() -> object:
    """An approver that accepts every gated call."""

    async def approve(tool: Tool[object], target: str, perm: PermissionResult) -> Approval:
        return Approval.ALLOW

    return approve


@pytest.fixture
def git_workspace(workspace: Path) -> Path:
    """An initialized git repo with one commit, for the git tools.

    The global and system configs are pointed at /dev/null so the developer's own
    git settings (signing keys, hooks, default branch) can't change the fixture.
    """
    env = {
        "HOME": str(workspace),
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_SYSTEM": "/dev/null",
    }
    run = lambda *args: subprocess.run(  # noqa: E731 - local shorthand
        args, cwd=workspace, check=True, capture_output=True, env=env
    )
    run("git", "init", "--initial-branch=main")
    run("git", "config", "user.email", "test@example.com")
    run("git", "config", "user.name", "Test")
    run("git", "config", "commit.gpgsign", "false")
    (workspace / "README.md").write_text("# fixture\n")
    run("git", "add", "README.md")
    run("git", "commit", "-m", "initial commit")
    return workspace
