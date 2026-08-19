"""Tests for the tool registry and argument validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from forge.tools import default_registry
from forge.tools.filesystem import ReadFile
from forge.tools.registry import ToolRegistry


def test_register_and_lookup() -> None:
    reg = ToolRegistry([ReadFile()])
    assert "read_file" in reg
    assert reg.get("read_file") is not None
    assert reg.get("missing") is None
    assert "read_file" in reg.names()
    assert len(reg) == 1


def test_duplicate_register_raises() -> None:
    reg = ToolRegistry([ReadFile()])
    with pytest.raises(ValueError, match="already registered"):
        reg.register(ReadFile())


def test_provider_schema_shape() -> None:
    reg = default_registry()
    schema = reg.to_provider_schema()
    assert len(schema) == len(reg)
    for entry in schema:
        assert set(entry) == {"name", "description", "input_schema"}
        assert entry["input_schema"]["type"] == "object"


def test_parse_args_validation_error() -> None:
    with pytest.raises(ValidationError):
        ReadFile().parse_args({})  # missing required 'path'
