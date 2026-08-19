"""Forge command-line interface.

Two ways in, one runtime behind both::

    forge                                  # interactive REPL
    forge "fix the failing tests"           # one-shot, exits when done
    forge --model claude-sonnet-5 "..."     # any global option works with either
    forge --json "analyze this repository"  # machine-readable result on stdout
    echo "audit the config loader" | forge  # task from stdin

    forge config                            # show resolved configuration
    forge tools                             # list registered tools and risk
    forge version

One-shot mode is what makes Forge usable from CI, git hooks, scripts, and other
agents: it never prompts (gated calls are refused rather than blocking on a
terminal nobody is watching) and its exit code reports the outcome —
``0`` completed, ``130`` interrupted, ``2`` misconfigured, ``1`` anything else.
"""

from __future__ import annotations

import asyncio
import json
import sys
from enum import StrEnum
from pathlib import Path
from typing import Any, TypeVar

import typer

from forge import __version__
from forge.agent.loop import AgentResult, AgentRuntime, build_runtime
from forge.config import ApprovalMode, SandboxMode, Settings, load_settings
from forge.logging import configure_logging
from forge.ui.console import Console

EnumT = TypeVar("EnumT", bound=StrEnum)

EXIT_MISCONFIGURED = 2

# Subcommand names. Anything else in the first positional slot is a task, which
# is what lets `forge "do the thing"` work without a `run` subcommand.
COMMANDS: frozenset[str] = frozenset({"run", "repl", "config", "tools", "version"})

# Global options that consume the following argv token as their value.
_VALUE_OPTIONS: frozenset[str] = frozenset(
    {
        "--model",
        "--provider",
        "--workspace",
        "-C",
        "--max-iterations",
        "--mode",
        "--sandbox",
        "--max-tokens",
    }
)

app = typer.Typer(
    add_completion=False,
    no_args_is_help=False,
    help="Forge — an autonomous software-engineering agent.",
    context_settings={"help_option_names": ["-h", "--help"]},
)


class CLIState:
    def __init__(self, settings: Settings, console: Console, *, as_json: bool) -> None:
        self.settings = settings
        self.console = console
        self.as_json = as_json


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"Forge {__version__}")
        raise typer.Exit(0)


@app.callback(invoke_without_command=True)
def main_callback(
    ctx: typer.Context,
    model: str | None = typer.Option(
        None, "--model", help="Model id (defaults to $ANTHROPIC_MODEL)."
    ),
    provider: str | None = typer.Option(
        None, "--provider", help="Provider: 'anthropic' (default) or 'fake'."
    ),
    workspace: str | None = typer.Option(
        None, "-C", "--workspace", help="Workspace root the agent operates in (default: cwd)."
    ),
    mode: str | None = typer.Option(
        None,
        "--mode",
        help="Approval mode: cautious | auto | yolo. Default auto "
        "(writes run unattended, destructive actions ask).",
    ),
    yes: bool = typer.Option(
        False, "--yes", "-y", "--yolo", help="Shorthand for --mode yolo. Approves everything."
    ),
    cautious: bool = typer.Option(
        False, "--cautious", help="Shorthand for --mode cautious. Approve every write."
    ),
    sandbox: str | None = typer.Option(
        None, "--sandbox", help="Shell isolation: none (default) or docker."
    ),
    max_iterations: int | None = typer.Option(
        None, "--max-iterations", help="Hard ceiling on loop iterations per task."
    ),
    max_tokens: int | None = typer.Option(
        None, "--max-tokens", help="Max output tokens per model call."
    ),
    as_json: bool = typer.Option(
        False, "--json", help="Emit a single JSON result on stdout and nothing else."
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show structured step logs on stderr."
    ),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress the activity log."),
    _version: bool = typer.Option(
        False, "--version", callback=_version_callback, is_eager=True, help="Show version and exit."
    ),
) -> None:
    overrides: dict[str, Any] = {
        "model": model,
        "provider": provider,
        "max_iterations": max_iterations,
        "max_tokens": max_tokens,
    }
    if yes:
        overrides["approval_mode"] = ApprovalMode.YOLO
    elif cautious:
        overrides["approval_mode"] = ApprovalMode.CAUTIOUS
    elif mode:
        overrides["approval_mode"] = _parse_enum(ApprovalMode, mode, "--mode")
    if sandbox:
        overrides["sandbox"] = _parse_enum(SandboxMode, sandbox, "--sandbox")

    root = Path(workspace).expanduser().resolve() if workspace else Path.cwd()
    try:
        settings = load_settings(workspace=root, cli_overrides=overrides)
    except (ValueError, TypeError) as exc:
        Console().error(str(exc))
        raise typer.Exit(EXIT_MISCONFIGURED) from exc

    configure_logging("INFO" if verbose else settings.log_level, json=settings.log_json)
    ctx.obj = CLIState(settings, Console(quiet=quiet or as_json), as_json=as_json)
    if ctx.invoked_subcommand is None:
        _run_repl(ctx.obj)


def _parse_enum(enum_cls: type[EnumT], raw: str, flag: str) -> EnumT:
    try:
        return enum_cls(raw.lower())
    except ValueError as exc:
        allowed = ", ".join(member.value for member in enum_cls)
        raise typer.BadParameter(f"{flag} must be one of: {allowed}", param_hint=flag) from exc


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #
@app.command()
def run(
    ctx: typer.Context,
    task: str | None = typer.Argument(
        None, help="The task for the agent. Reads stdin when omitted."
    ),
) -> None:
    """Run a single task to completion and exit."""
    state: CLIState = ctx.obj
    prompt = task if task is not None else _read_stdin_task()
    if not prompt:
        state.console.error("No task given. Pass one as an argument or pipe it on stdin.")
        raise typer.Exit(EXIT_MISCONFIGURED)

    result = asyncio.run(_run_once(state, prompt))
    if state.as_json:
        state.console.print_raw(json.dumps(result.to_dict(), indent=2))
    raise typer.Exit(result.exit_code)


@app.command()
def repl(ctx: typer.Context) -> None:
    """Start the interactive session (also the default with no arguments)."""
    _run_repl(ctx.obj)


@app.command()
def config(ctx: typer.Context) -> None:
    """Show the resolved configuration (secrets redacted)."""
    state: CLIState = ctx.obj
    s = state.settings
    rows: list[tuple[str, str]] = [
        ("provider", s.provider),
        ("model", s.model or "(unset)"),
        ("base_url", s.base_url or "(provider default)"),
        ("credential", "set" if (s.auth_token or s.api_key) else "MISSING"),
        ("workspace_root", str(s.workspace_root)),
        ("approval_mode", s.approval_mode.value),
        ("sandbox", s.sandbox.value),
        ("max_iterations", str(s.max_iterations)),
        ("max_tokens", str(s.max_tokens)),
        ("max_context_tokens", f"{s.max_context_tokens:,}"),
        ("shell_timeout", f"{s.shell_timeout}s"),
        ("git_tools", "enabled" if s.enable_git_tools else "disabled"),
        ("allow rules", str(len(s.allow))),
        ("deny rules", str(len(s.deny))),
        ("destructive rules", str(len(s.destructive))),
        ("log_level", s.log_level),
    ]

    if state.as_json:
        state.console.print_raw(json.dumps(dict(rows), indent=2))
        return

    from rich.table import Table

    table = Table(title="Forge configuration", title_justify="left", show_header=False)
    table.add_column(style="dim", justify="right")
    table.add_column()
    for key, value in rows:
        table.add_row(key, value)
    state.console.print(table)


@app.command()
def tools(ctx: typer.Context) -> None:
    """List the tools the model can call, with their risk level."""
    from forge.tools import default_registry

    state: CLIState = ctx.obj
    registry = default_registry(state.settings)
    entries = [
        {"name": tool.name, "risk": tool.risk.value, "description": tool.description}
        for tool in sorted(registry, key=lambda t: (t.risk.value, t.name))
    ]

    if state.as_json:
        state.console.print_raw(json.dumps(entries, indent=2))
        return

    from rich.table import Table

    table = Table(title=f"Forge tools ({len(entries)})", title_justify="left")
    table.add_column("tool", style="bold cyan")
    table.add_column("risk")
    table.add_column("description")
    styles = {"read": "green", "write": "yellow", "destructive": "red"}
    for entry in entries:
        risk = entry["risk"]
        table.add_row(
            entry["name"], f"[{styles[risk]}]{risk}[/{styles[risk]}]", entry["description"]
        )
    state.console.print(table)


@app.command()
def version() -> None:
    """Print the Forge version."""
    typer.echo(f"Forge {__version__}")


# --------------------------------------------------------------------------- #
# Execution helpers
# --------------------------------------------------------------------------- #
def _read_stdin_task() -> str:
    """Take the task from a pipe, if there is one."""
    if sys.stdin.isatty():
        return ""
    return sys.stdin.read().strip()


def _build(state: CLIState) -> AgentRuntime:
    try:
        return build_runtime(state.settings, state.console)
    except Exception as exc:  # noqa: BLE001 - config/credential/sandbox problems
        state.console.error(str(exc))
        raise typer.Exit(EXIT_MISCONFIGURED) from exc


async def _run_once(state: CLIState, task: str) -> AgentResult:
    runtime = _build(state)
    try:
        return await runtime.run_task(task)
    finally:
        await runtime.aclose()


def _run_repl(state: CLIState) -> None:
    asyncio.run(_repl_loop(state))


_REPL_HELP = """\
[bold]Commands[/bold]
  /help              show this message
  /tools             list available tools
  /config            show resolved configuration
  /clear             forget the conversation so far
  /exit              quit (also: exit, quit, Ctrl-D)

Anything else is sent to the agent as a task.
"""


async def _repl_loop(state: CLIState) -> None:
    console = state.console
    runtime = _build(state)
    console.banner(
        model=state.settings.model or "(unset)",
        workspace=str(state.settings.workspace_root),
        mode=state.settings.approval_mode.value,
        sandbox=runtime.ctx.sandbox.describe(),
    )

    agent_state = None
    try:
        while True:
            try:
                raw = await asyncio.to_thread(console.input, "[bold cyan]›[/bold cyan] ")
            except (EOFError, KeyboardInterrupt):
                console.print("")
                return

            task = raw.strip()
            if not task:
                continue
            if task.lower() in {"exit", "quit", ":q", "/exit", "/quit"}:
                return
            if task.startswith("/"):
                agent_state = _handle_repl_command(state, task, agent_state)
                continue

            try:
                result = await runtime.run_task(task, agent_state)
            except KeyboardInterrupt:
                console.warning("Interrupted; the session is still open.")
                continue
            agent_state = result.state
    finally:
        await runtime.aclose()


def _handle_repl_command(state: CLIState, command: str, agent_state: Any) -> Any:
    """Handle a ``/command``. Returns the (possibly reset) agent state."""
    from rich.text import Text

    name = command.split()[0].lower()
    if name == "/help":
        state.console.print(Text.from_markup(_REPL_HELP))
    elif name == "/clear":
        state.console.info("Conversation cleared.")
        return None
    elif name == "/tools":
        _invoke_subcommand(tools, state)
    elif name == "/config":
        _invoke_subcommand(config, state)
    else:
        state.console.warning(f"Unknown command {name}. Try /help.")
    return agent_state


def _invoke_subcommand(command: Any, state: CLIState) -> None:
    """Call a Typer command function directly with a minimal context."""
    ctx = typer.Context(typer.main.get_command(app))
    ctx.obj = state
    command(ctx)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def _insert_run_command(argv: list[str]) -> list[str]:
    """Rewrite ``forge "task"`` into ``forge run "task"``.

    Typer needs a subcommand, but a coding agent should accept a bare task as its
    first positional argument. Scan past global options (and their values) to the
    first positional token; if it is not a known subcommand, it is a task.
    """
    index = 0
    while index < len(argv):
        token = argv[index]
        if token == "--":
            index += 1
            break
        if token.startswith("-"):
            index += 2 if token in _VALUE_OPTIONS else 1
            continue
        break

    if index >= len(argv) or argv[index] in COMMANDS:
        return argv
    return [*argv[:index], "run", *argv[index:]]


def main(argv: list[str] | None = None) -> None:
    """Console-script entry point."""
    args = list(sys.argv[1:] if argv is None else argv)
    app(args=_insert_run_command(args), prog_name="forge", standalone_mode=True)


if __name__ == "__main__":
    main()
