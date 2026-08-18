"""Forge command-line interface (Typer + Rich).

Usage:
    forge                         # interactive REPL (default)
    forge [GLOBAL OPTIONS] run "<task>"   # one-shot, non-interactive-friendly
    forge config                  # show resolved configuration
    forge version

Global options (given before the subcommand, git-style) configure the session:
    forge -C path --model M --yes run "..."
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

import typer

from forge import __version__
from forge.agent.loop import AgentResult, build_runtime
from forge.config import Settings, load_settings
from forge.logging import configure_logging
from forge.ui.console import Console

app = typer.Typer(
    add_completion=False,
    no_args_is_help=False,
    help="Forge — an autonomous software-engineering agent.",
)


class CLIState:
    def __init__(self, settings: Settings, console: Console) -> None:
        self.settings = settings
        self.console = console


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    model: Optional[str] = typer.Option(None, "--model", help="Model id (defaults to $ANTHROPIC_MODEL)."),
    provider: Optional[str] = typer.Option(None, "--provider", help="Provider: anthropic (default) or fake."),
    workspace: Optional[str] = typer.Option(
        None, "-C", "--workspace", help="Workspace root the agent operates in (default: cwd)."
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Auto-approve gated tool calls."),
    max_iterations: Optional[int] = typer.Option(None, "--max-iterations", help="Max loop iterations per task."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show structured step logs (stderr)."),
) -> None:
    overrides: dict[str, object] = {}
    if model is not None:
        overrides["model"] = model
    if provider is not None:
        overrides["provider"] = provider
    if max_iterations is not None:
        overrides["max_iterations"] = max_iterations
    if yes:
        overrides["auto_approve"] = True

    ws = Path(workspace).expanduser().resolve() if workspace else Path.cwd()
    settings = load_settings(workspace=ws, cli_overrides=overrides)
    configure_logging("INFO" if verbose else settings.log_level, json=settings.log_json)

    ctx.obj = CLIState(settings, Console())
    if ctx.invoked_subcommand is None:
        _repl(ctx.obj)


@app.command()
def repl(ctx: typer.Context) -> None:
    """Start the interactive REPL (this is also the default with no subcommand)."""
    _repl(ctx.obj)


@app.command()
def run(ctx: typer.Context, task: str = typer.Argument(..., help="The task for the agent.")) -> None:
    """Run a single task to completion and exit."""
    state: CLIState = ctx.obj

    async def go() -> AgentResult:
        runtime = build_runtime(state.settings, state.console)
        return await runtime.run_task(task)

    try:
        result = asyncio.run(go())
    except Exception as exc:  # noqa: BLE001 - surface init/transport errors cleanly
        state.console.error(str(exc))
        raise typer.Exit(1) from exc
    raise typer.Exit(0 if result.status == "completed" else 1)


@app.command()
def config(ctx: typer.Context) -> None:
    """Show the resolved configuration (secrets redacted)."""
    from rich.table import Table

    s = ctx.obj.settings
    table = Table(title="Forge configuration", title_justify="left", show_header=False)
    table.add_column(style="dim", justify="right")
    table.add_column()
    table.add_row("provider", str(s.provider))
    table.add_row("model", str(s.model))
    table.add_row("base_url", str(s.base_url or "(default)"))
    table.add_row("credential", "set" if (s.auth_token or s.api_key) else "MISSING")
    table.add_row("workspace_root", str(s.workspace_root))
    table.add_row("max_iterations", str(s.max_iterations))
    table.add_row("auto_approve", str(s.auto_approve))
    table.add_row("deny rules", str(len(s.deny)))
    table.add_row("allow rules", str(len(s.allow)))
    table.add_row("log_level", str(s.log_level))
    ctx.obj.console.print(table)


@app.command()
def version() -> None:
    """Print the Forge version."""
    typer.echo(f"forge {__version__}")


def _repl(state: CLIState) -> None:
    console = state.console

    async def loop() -> None:
        try:
            runtime = build_runtime(state.settings, console)
        except Exception as exc:  # noqa: BLE001
            console.error(f"Failed to initialize: {exc}")
            raise typer.Exit(1) from exc

        console.banner(
            model=str(state.settings.model or "(unset)"),
            workspace=str(state.settings.workspace_root),
        )
        agent_state = None
        while True:
            try:
                task = (await asyncio.to_thread(console.input, "[bold cyan]›[/bold cyan] ")).strip()
            except (EOFError, KeyboardInterrupt):
                console.print("")
                break
            if not task:
                continue
            if task.lower() in {"exit", "quit", ":q"}:
                break
            result = await runtime.run_task(task, agent_state)
            agent_state = result.state

    asyncio.run(loop())


if __name__ == "__main__":
    app()
