"""Tests for the permission policy."""

from __future__ import annotations

from forge.config import Settings
from forge.permissions.policy import Decision, PermissionPolicy
from forge.tools.filesystem import ReadFile, WriteFile
from forge.tools.shell import Shell


def test_readonly_tool_is_allowed() -> None:
    policy = PermissionPolicy(Settings())
    assert policy.decide(ReadFile(), {"path": "x"}).decision is Decision.ALLOW


def test_write_asks_by_default() -> None:
    policy = PermissionPolicy(Settings(auto_approve=False))
    result = policy.decide(WriteFile(), {"path": "x", "content": "y"})
    assert result.decision is Decision.ASK


def test_deny_wins_over_auto_approve() -> None:
    policy = PermissionPolicy(Settings(auto_approve=True))
    result = policy.decide(Shell(), {"command": "rm -rf /"})
    assert result.decision is Decision.DENY
    assert "deny rule" in result.reason


def test_allow_prefix_matches() -> None:
    policy = PermissionPolicy(Settings(allow=["echo "]))
    assert policy.decide(Shell(), {"command": "echo hi"}).decision is Decision.ALLOW


def test_auto_approve_allows_writes() -> None:
    policy = PermissionPolicy(Settings(auto_approve=True))
    assert policy.decide(WriteFile(), {"path": "x", "content": "y"}).decision is Decision.ALLOW


def test_session_always_allow_grant() -> None:
    policy = PermissionPolicy(Settings(auto_approve=False))
    policy.always_allow_tool("write_file")
    assert policy.decide(WriteFile(), {"path": "x", "content": "y"}).decision is Decision.ALLOW
