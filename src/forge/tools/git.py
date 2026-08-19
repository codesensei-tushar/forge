"""Git tools.

Git gets first-class tools rather than being left to raw ``shell`` calls for
three reasons: the model gets typed schemas instead of having to remember flags,
every invocation is shell-quoted so a branch name can't smuggle in a second
command, and each operation carries an explicit risk level the permission policy
can act on.

Risk mapping follows the project's rule — reads and ordinary writes run
unattended, anything that discards work needs a human:

============================  ===============
``git_status`` ``git_diff``   ``READ``
``git_log``  ``git_show``     ``READ``
``git_add``  ``git_commit``   ``WRITE``
``git_branch`` (create/list)  ``WRITE``
``git_branch --delete``       ``DESTRUCTIVE``
``git_checkout`` (branch)     ``WRITE``
``git_checkout`` (paths)      ``DESTRUCTIVE`` — overwrites uncommitted work
``git_revert`` ``git_reset``  ``DESTRUCTIVE``
============================  ===============

``git push`` is deliberately absent. Publishing is the operator's call, not the
agent's, and there is no tool here that can perform it.
"""

from __future__ import annotations

import shlex
from typing import Any

from pydantic import BaseModel, Field

from forge.tools.base import Risk, Tool, ToolResult
from forge.tools.context import ToolContext
from forge.tools.shell import format_exec_result, truncate_middle

# Applied to every invocation: never page, never colour, never prompt for creds.
_GIT_PREFIX = "git --no-pager -c color.ui=false -c core.pager=cat"
_GIT_ENV = {"GIT_TERMINAL_PROMPT": "0", "GIT_ASKPASS": "/bin/false"}


async def _git(ctx: ToolContext, argv: list[str], *, timeout: int | None = None) -> ToolResult:
    """Run one git command and normalize its outcome into a ToolResult."""
    command = f"{_GIT_PREFIX} {' '.join(argv)}"
    outcome = await ctx.exec(
        command, timeout=timeout or ctx.settings.shell_timeout, env=dict(_GIT_ENV)
    )

    if outcome.startup_error:
        return ToolResult.error(outcome.startup_error)
    if outcome.timed_out:
        return ToolResult.error(f"git command timed out: {command}")

    combined = f"{outcome.stdout}\n{outcome.stderr}".lower()
    if outcome.exit_code != 0 and "not a git repository" in combined:
        return ToolResult.error(
            "This workspace is not a git repository. Run `git init` via the shell "
            "tool first if the task needs version control."
        )

    body = format_exec_result(command, outcome.stdout, outcome.stderr, outcome.exit_code)
    return ToolResult(
        content=body,
        is_error=outcome.exit_code != 0,
        metadata={"exit_code": outcome.exit_code},
    )


def _quote(value: str) -> str:
    return shlex.quote(value)


def _quote_paths(paths: list[str] | None) -> list[str]:
    """Quote pathspecs and stop them from being read as options."""
    if not paths:
        return []
    return ["--", *(_quote(p) for p in paths)]


# --------------------------------------------------------------------------- #
# Read operations
# --------------------------------------------------------------------------- #
class GitStatusArgs(BaseModel):
    short: bool = Field(default=True, description="Use the compact --short format.")


class GitStatus(Tool[GitStatusArgs]):
    name = "git_status"
    description = "Show the working-tree status: staged, unstaged, and untracked files."
    risk = Risk.READ
    Args = GitStatusArgs

    async def run(self, args: GitStatusArgs, ctx: ToolContext) -> ToolResult:
        argv = ["status"]
        argv.append("--short --branch" if args.short else "--long")
        return await _git(ctx, argv)


class GitDiffArgs(BaseModel):
    staged: bool = Field(default=False, description="Diff the index against HEAD instead.")
    ref: str | None = Field(
        default=None, description="Compare against this commit or range, e.g. 'HEAD~1' or 'main'."
    )
    paths: list[str] | None = Field(default=None, description="Limit the diff to these paths.")
    stat: bool = Field(default=False, description="Show a summary of changed files only.")
    context_lines: int = Field(default=3, ge=0, le=25, description="Lines of context per hunk.")


class GitDiff(Tool[GitDiffArgs]):
    name = "git_diff"
    description = (
        "Show changes as a unified diff. Use this to review your own edits before "
        "committing, and to confirm what a task actually changed."
    )
    risk = Risk.READ
    Args = GitDiffArgs

    async def run(self, args: GitDiffArgs, ctx: ToolContext) -> ToolResult:
        argv = ["diff", f"--unified={args.context_lines}"]
        if args.staged:
            argv.append("--staged")
        if args.stat:
            argv.append("--stat")
        if args.ref:
            argv.append(_quote(args.ref))
        argv += _quote_paths(args.paths)
        return await _git(ctx, argv)


class GitLogArgs(BaseModel):
    limit: int = Field(default=20, ge=1, le=200, description="Number of commits to show.")
    oneline: bool = Field(default=True, description="One commit per line.")
    paths: list[str] | None = Field(default=None, description="Only commits touching these paths.")


class GitLog(Tool[GitLogArgs]):
    name = "git_log"
    description = "Show recent commit history."
    risk = Risk.READ
    Args = GitLogArgs

    async def run(self, args: GitLogArgs, ctx: ToolContext) -> ToolResult:
        argv = ["log", f"--max-count={args.limit}"]
        argv.append("--oneline --no-decorate" if args.oneline else "--stat")
        argv += _quote_paths(args.paths)
        return await _git(ctx, argv)


class GitShowArgs(BaseModel):
    ref: str = Field(default="HEAD", description="Commit, tag, or ref to display.")
    paths: list[str] | None = Field(default=None, description="Limit output to these paths.")
    stat: bool = Field(default=False, description="Show only the summary of changed files.")


class GitShow(Tool[GitShowArgs]):
    name = "git_show"
    description = "Show a commit's message and diff."
    risk = Risk.READ
    Args = GitShowArgs

    async def run(self, args: GitShowArgs, ctx: ToolContext) -> ToolResult:
        argv = ["show", _quote(args.ref)]
        if args.stat:
            argv.append("--stat")
        argv += _quote_paths(args.paths)
        return await _git(ctx, argv)


# --------------------------------------------------------------------------- #
# Write operations
# --------------------------------------------------------------------------- #
class GitAddArgs(BaseModel):
    paths: list[str] = Field(description="Paths to stage. Use ['.'] to stage everything.")


class GitAdd(Tool[GitAddArgs]):
    name = "git_add"
    description = "Stage changes for the next commit."
    risk = Risk.WRITE
    Args = GitAddArgs

    async def run(self, args: GitAddArgs, ctx: ToolContext) -> ToolResult:
        if not args.paths:
            return ToolResult.error("No paths given to stage.")
        result = await _git(ctx, ["add", *_quote_paths(args.paths)])
        if result.is_error:
            return result
        return await _git(ctx, ["status", "--short"])


class GitCommitArgs(BaseModel):
    message: str = Field(description="Commit message. First line is the subject.")
    paths: list[str] | None = Field(
        default=None, description="Stage these paths first. Omit to commit what is already staged."
    )
    all_tracked: bool = Field(
        default=False, description="Stage all modified tracked files first (git commit -a)."
    )


class GitCommit(Tool[GitCommitArgs]):
    name = "git_commit"
    description = (
        "Create a commit. Commits only what is staged unless paths or all_tracked "
        "are given. Never pushes."
    )
    risk = Risk.WRITE
    Args = GitCommitArgs

    async def run(self, args: GitCommitArgs, ctx: ToolContext) -> ToolResult:
        if not args.message.strip():
            return ToolResult.error("A commit message is required.")
        if args.paths:
            staged = await _git(ctx, ["add", *_quote_paths(args.paths)])
            if staged.is_error:
                return staged

        argv = ["commit", "--message", _quote(args.message)]
        if args.all_tracked:
            argv.append("--all")
        result = await _git(ctx, argv)
        if result.is_error and "nothing to commit" in result.content.lower():
            return ToolResult.error(
                "Nothing to commit — the working tree is clean or nothing is staged. "
                "Stage changes with git_add first."
            )
        return result


class GitBranchArgs(BaseModel):
    name: str | None = Field(
        default=None, description="Branch to create or delete. Omit to list branches."
    )
    create: bool = Field(default=False, description="Create the branch and switch to it.")
    delete: bool = Field(default=False, description="Delete the branch (destructive).")
    from_ref: str | None = Field(default=None, description="Base ref when creating.")


class GitBranch(Tool[GitBranchArgs]):
    name = "git_branch"
    description = "List branches, or create/switch to a new branch. Deletion is gated."
    risk = Risk.WRITE
    Args = GitBranchArgs

    def risk_for(self, args: dict[str, Any]) -> Risk:
        if args.get("delete"):
            return Risk.DESTRUCTIVE
        if not args.get("name"):
            return Risk.READ  # a bare listing
        return Risk.WRITE

    async def run(self, args: GitBranchArgs, ctx: ToolContext) -> ToolResult:
        if not args.name:
            return await _git(ctx, ["branch", "--list", "--all", "--no-color"])
        if args.delete:
            return await _git(ctx, ["branch", "--delete", "--", _quote(args.name)])
        if args.create:
            argv = ["switch", "--create", _quote(args.name)]
            if args.from_ref:
                argv.append(_quote(args.from_ref))
            return await _git(ctx, argv)
        return await _git(ctx, ["switch", "--", _quote(args.name)])


class GitCheckoutArgs(BaseModel):
    ref: str | None = Field(default=None, description="Branch or commit to switch to.")
    paths: list[str] | None = Field(
        default=None,
        description="Restore these paths from ref, DISCARDING uncommitted changes to them.",
    )


class GitCheckout(Tool[GitCheckoutArgs]):
    name = "git_checkout"
    description = (
        "Switch to a branch or commit, or restore specific paths from a ref. "
        "Restoring paths discards uncommitted changes and requires approval."
    )
    risk = Risk.WRITE
    Args = GitCheckoutArgs

    def risk_for(self, args: dict[str, Any]) -> Risk:
        # Switching refs is recoverable; overwriting working-tree files is not.
        return Risk.DESTRUCTIVE if args.get("paths") else Risk.WRITE

    async def run(self, args: GitCheckoutArgs, ctx: ToolContext) -> ToolResult:
        if not args.ref and not args.paths:
            return ToolResult.error("Give a ref to switch to, or paths to restore.")
        argv = ["checkout"]
        if args.ref:
            argv.append(_quote(args.ref))
        argv += _quote_paths(args.paths)
        return await _git(ctx, argv)


# --------------------------------------------------------------------------- #
# Destructive operations
# --------------------------------------------------------------------------- #
class GitRevertArgs(BaseModel):
    ref: str = Field(description="Commit to revert.")
    no_commit: bool = Field(
        default=False, description="Stage the inverse changes without committing."
    )


class GitRevert(Tool[GitRevertArgs]):
    name = "git_revert"
    description = "Create a commit that undoes an earlier commit."
    risk = Risk.DESTRUCTIVE
    Args = GitRevertArgs

    async def run(self, args: GitRevertArgs, ctx: ToolContext) -> ToolResult:
        argv = ["revert", "--no-edit", _quote(args.ref)]
        if args.no_commit:
            argv.append("--no-commit")
        return await _git(ctx, argv)


class GitResetArgs(BaseModel):
    ref: str = Field(default="HEAD", description="Ref to reset to.")
    mode: str = Field(
        default="mixed",
        description="'soft' keeps changes staged, 'mixed' unstages them, "
        "'hard' DISCARDS all uncommitted work.",
    )


class GitReset(Tool[GitResetArgs]):
    name = "git_reset"
    description = "Move HEAD to a ref. 'hard' mode permanently discards uncommitted work."
    risk = Risk.DESTRUCTIVE
    Args = GitResetArgs

    async def run(self, args: GitResetArgs, ctx: ToolContext) -> ToolResult:
        mode = args.mode.lower().strip()
        if mode not in {"soft", "mixed", "hard"}:
            return ToolResult.error(f"Unknown reset mode {args.mode!r}; use soft, mixed, or hard.")
        return await _git(ctx, ["reset", f"--{mode}", _quote(args.ref)])


def git_tools() -> list[Tool[Any]]:
    return [
        GitStatus(),
        GitDiff(),
        GitLog(),
        GitShow(),
        GitAdd(),
        GitCommit(),
        GitBranch(),
        GitCheckout(),
        GitRevert(),
        GitReset(),
    ]


async def collect_repo_summary(ctx: ToolContext) -> str:
    """A short branch/status snapshot for the system prompt. Empty if not a repo."""
    outcome = await ctx.exec(
        f"{_GIT_PREFIX} status --short --branch", timeout=15, env=dict(_GIT_ENV)
    )
    if outcome.exit_code != 0:
        return ""
    return truncate_middle(outcome.stdout.strip(), 1_500)
