"""Tests for the git tools.

Two things matter here beyond the happy paths: every argument is shell-quoted, so
a branch name or commit message cannot smuggle in a second command, and there is
no tool anywhere in the set that can push.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from forge.tools.context import ToolContext
from forge.tools.git import (
    GitAdd,
    GitBranch,
    GitCheckout,
    GitCommit,
    GitDiff,
    GitLog,
    GitReset,
    GitRevert,
    GitShow,
    GitStatus,
    collect_repo_summary,
    git_tools,
)

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")


async def call(tool: object, ctx: ToolContext, **args: object) -> object:
    return await tool.run(tool.parse_args(args), ctx)  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- #
# Reads
# --------------------------------------------------------------------------- #
async def test_status_on_a_clean_repo(git_workspace: Path, ctx: ToolContext) -> None:
    result = await call(GitStatus(), ctx)
    assert not result.is_error  # type: ignore[attr-defined]
    assert "## main" in result.content  # type: ignore[attr-defined]


async def test_status_sees_a_new_file(git_workspace: Path, ctx: ToolContext) -> None:
    (git_workspace / "extra.py").write_text("x = 1\n")
    result = await call(GitStatus(), ctx)
    assert "extra.py" in result.content  # type: ignore[attr-defined]


async def test_non_repository_gets_an_actionable_message(workspace: Path, ctx: ToolContext) -> None:
    result = await call(GitStatus(), ctx)
    assert result.is_error  # type: ignore[attr-defined]
    assert "not a git repository" in result.content  # type: ignore[attr-defined]
    assert "git init" in result.content  # type: ignore[attr-defined]


async def test_diff_shows_the_change(git_workspace: Path, ctx: ToolContext) -> None:
    (git_workspace / "README.md").write_text("# changed\n")
    result = await call(GitDiff(), ctx)
    assert "-# fixture" in result.content  # type: ignore[attr-defined]
    assert "+# changed" in result.content  # type: ignore[attr-defined]


async def test_diff_stat_summarizes(git_workspace: Path, ctx: ToolContext) -> None:
    (git_workspace / "README.md").write_text("# changed\n")
    result = await call(GitDiff(), ctx, stat=True)
    assert "1 file changed" in result.content  # type: ignore[attr-defined]


async def test_log_lists_commits(git_workspace: Path, ctx: ToolContext) -> None:
    result = await call(GitLog(), ctx, limit=5)
    assert "initial commit" in result.content  # type: ignore[attr-defined]


async def test_show_renders_head(git_workspace: Path, ctx: ToolContext) -> None:
    result = await call(GitShow(), ctx)
    assert "initial commit" in result.content  # type: ignore[attr-defined]
    assert "README.md" in result.content  # type: ignore[attr-defined]


async def test_repo_summary_is_empty_outside_a_repo(workspace: Path, ctx: ToolContext) -> None:
    assert await collect_repo_summary(ctx) == ""


async def test_repo_summary_names_the_branch(git_workspace: Path, ctx: ToolContext) -> None:
    assert "## main" in await collect_repo_summary(ctx)


# --------------------------------------------------------------------------- #
# Writes
# --------------------------------------------------------------------------- #
async def test_add_then_commit(git_workspace: Path, ctx: ToolContext) -> None:
    (git_workspace / "feature.py").write_text("def f(): return 1\n")

    staged = await call(GitAdd(), ctx, paths=["feature.py"])
    assert not staged.is_error  # type: ignore[attr-defined]
    assert "feature.py" in staged.content, "add reports the resulting status"  # type: ignore[attr-defined]

    committed = await call(GitCommit(), ctx, message="feat: add feature")
    assert not committed.is_error  # type: ignore[attr-defined]

    log = await call(GitLog(), ctx)
    assert "feat: add feature" in log.content  # type: ignore[attr-defined]


async def test_commit_can_stage_its_own_paths(git_workspace: Path, ctx: ToolContext) -> None:
    (git_workspace / "a.py").write_text("a = 1\n")
    result = await call(GitCommit(), ctx, message="add a", paths=["a.py"])
    assert not result.is_error  # type: ignore[attr-defined]


async def test_add_needs_paths(git_workspace: Path, ctx: ToolContext) -> None:
    result = await call(GitAdd(), ctx, paths=[])
    assert result.is_error and "No paths" in result.content  # type: ignore[attr-defined]


async def test_commit_needs_a_message(git_workspace: Path, ctx: ToolContext) -> None:
    result = await call(GitCommit(), ctx, message="   ")
    assert result.is_error and "message is required" in result.content  # type: ignore[attr-defined]


async def test_empty_commit_explains_what_to_do(git_workspace: Path, ctx: ToolContext) -> None:
    result = await call(GitCommit(), ctx, message="nothing changed")
    assert result.is_error  # type: ignore[attr-defined]
    assert "Nothing to commit" in result.content  # type: ignore[attr-defined]
    assert "git_add" in result.content  # type: ignore[attr-defined]


async def test_branch_lists_then_creates(git_workspace: Path, ctx: ToolContext) -> None:
    listing = await call(GitBranch(), ctx)
    assert "main" in listing.content  # type: ignore[attr-defined]

    created = await call(GitBranch(), ctx, name="feature/x", create=True)
    assert not created.is_error  # type: ignore[attr-defined]

    after = await call(GitBranch(), ctx)
    assert "feature/x" in after.content  # type: ignore[attr-defined]


async def test_checkout_switches_back(git_workspace: Path, ctx: ToolContext) -> None:
    await call(GitBranch(), ctx, name="side", create=True)
    result = await call(GitCheckout(), ctx, ref="main")
    assert not result.is_error  # type: ignore[attr-defined]
    assert "## main" in (await call(GitStatus(), ctx)).content  # type: ignore[attr-defined]


async def test_checkout_needs_something_to_do(git_workspace: Path, ctx: ToolContext) -> None:
    result = await call(GitCheckout(), ctx)
    assert result.is_error and "Give a ref" in result.content  # type: ignore[attr-defined]


async def test_checkout_restores_a_path(git_workspace: Path, ctx: ToolContext) -> None:
    (git_workspace / "README.md").write_text("clobbered\n")
    result = await call(GitCheckout(), ctx, ref="HEAD", paths=["README.md"])
    assert not result.is_error  # type: ignore[attr-defined]
    assert (git_workspace / "README.md").read_text() == "# fixture\n"


# --------------------------------------------------------------------------- #
# Destructive
# --------------------------------------------------------------------------- #
async def test_revert_undoes_a_commit(git_workspace: Path, ctx: ToolContext) -> None:
    (git_workspace / "oops.py").write_text("bad = True\n")
    await call(GitCommit(), ctx, message="add oops", paths=["oops.py"])
    assert (git_workspace / "oops.py").exists()

    result = await call(GitRevert(), ctx, ref="HEAD")
    assert not result.is_error, result.content  # type: ignore[attr-defined]
    assert not (git_workspace / "oops.py").exists()


async def test_reset_rejects_an_unknown_mode(git_workspace: Path, ctx: ToolContext) -> None:
    result = await call(GitReset(), ctx, mode="nuclear")
    assert result.is_error and "Unknown reset mode" in result.content  # type: ignore[attr-defined]


async def test_hard_reset_discards_work(git_workspace: Path, ctx: ToolContext) -> None:
    (git_workspace / "README.md").write_text("scratch\n")
    result = await call(GitReset(), ctx, ref="HEAD", mode="hard")
    assert not result.is_error  # type: ignore[attr-defined]
    assert (git_workspace / "README.md").read_text() == "# fixture\n"


# --------------------------------------------------------------------------- #
# Injection and exposure
# --------------------------------------------------------------------------- #
async def test_a_commit_message_cannot_smuggle_in_a_command(
    git_workspace: Path, ctx: ToolContext
) -> None:
    """Every git argument goes through shlex.quote; this proves it end to end."""
    (git_workspace / "a.py").write_text("a = 1\n")
    hostile = "feat: add $(touch pwned) && touch pwned2"

    result = await call(GitCommit(), ctx, message=hostile, paths=["a.py"])
    assert not result.is_error, result.content  # type: ignore[attr-defined]

    assert not (git_workspace / "pwned").exists()
    assert not (git_workspace / "pwned2").exists()
    log = await call(GitLog(), ctx)
    assert hostile in log.content, "the message is stored literally"  # type: ignore[attr-defined]


async def test_a_branch_name_cannot_smuggle_in_a_command(
    git_workspace: Path, ctx: ToolContext
) -> None:
    await call(GitBranch(), ctx, name="x; touch pwned")
    assert not (git_workspace / "pwned").exists()


async def test_a_pathspec_cannot_be_read_as_an_option(
    git_workspace: Path, ctx: ToolContext
) -> None:
    """The `--` separator is what stops a path called '--force' from being a flag."""
    (git_workspace / "a.py").write_text("a = 1\n")
    result = await call(GitDiff(), ctx, paths=["--stat"])
    # git rejects the unknown pathspec instead of honouring it as a flag.
    assert "1 file changed" not in result.content  # type: ignore[attr-defined]


def test_no_git_tool_can_push() -> None:
    names = [t.name for t in git_tools()]
    assert "git_push" not in names
    assert not any("push" in name for name in names)
    assert not any("remote" in name for name in names)


def test_git_tool_risks_match_the_documented_table() -> None:
    from forge.tools.base import Risk

    by_name = {t.name: t.risk for t in git_tools()}
    assert by_name["git_status"] is Risk.READ
    assert by_name["git_diff"] is Risk.READ
    assert by_name["git_log"] is Risk.READ
    assert by_name["git_show"] is Risk.READ
    assert by_name["git_add"] is Risk.WRITE
    assert by_name["git_commit"] is Risk.WRITE
    assert by_name["git_revert"] is Risk.DESTRUCTIVE
    assert by_name["git_reset"] is Risk.DESTRUCTIVE
