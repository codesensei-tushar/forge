"""Shared test fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from forge.config import Settings
from forge.tools.context import ToolContext


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def settings(workspace: Path) -> Settings:
    return Settings(
        workspace_root=workspace,
        provider="fake",
        model="fake-model",
        auto_approve=True,
        max_iterations=10,
    )


@pytest.fixture
def ctx(settings: Settings) -> ToolContext:
    return ToolContext(settings)
