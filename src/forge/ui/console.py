"""Terminal UI rendering with Rich.

The single place that touches the terminal: assistant output, tool-call and
result rendering, the human approval prompt, and the end-of-run summary. Keeping
it here means ``--json`` mode is just a Console with ``quiet=True`` — no output
suppression logic scattered through the agent.
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any

from rich.console import Console as RichConsole
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from forge.observability.trace import RunTrace
from forge.permissions.policy import Approval, PermissionResult
from forge.tools.base import Risk, Tool
from forge.tools.executor import ToolOutcome

_RISK_STYLE: dict[Risk, str] = {
    Risk.READ: "dim",
    Risk.WRITE: "yellow",
    Risk.DESTRUCTIVE: "bold red",
}


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}m {secs:02d}s"


def _args_preview(tool_name: str, args: dict[str, Any], *, width: int = 70) -> str:
    """A one-line rendering of a call's arguments, for the activity log."""
    if command := args.get("command"):
        return str(command)
    bits = []
    for key in ("path", "paths", "pattern", "ref", "message", "glob", "name", "old_string"):
        value = args.get(key)
        if value in (None, "", [], {}):
            continue
        text = str(value).replace("\n", "\\n")
        if len(text) > width:
            text = text[: width - 3] + "..."
        bits.append(f"{key}={text}")
    return "  ".join(bits)


class Console:
    """Terminal renderer. With ``quiet=True`` every method becomes a no-op."""

    def __init__(self, *, quiet: bool = False) -> None:
        self._c = RichConsole()
        self._err = RichConsole(stderr=True)
        self.quiet = quiet

    # --- generic ---
    def print(self, *args: Any, **kwargs: Any) -> None:
        if not self.quiet:
            self._c.print(*args, **kwargs)

    def print_raw(self, text: str) -> None:
        """Write to stdout unstyled and unwrapped — used for --json payloads."""
        sys.stdout.write(text if text.endswith("\n") else text + "\n")
        sys.stdout.flush()

    def info(self, message: str) -> None:
        if not self.quiet:
            self._c.print(f"[dim]{message}[/dim]")

    def input(self, prompt: str) -> str:
        return self._c.input(prompt)

    def warning(self, message: str) -> None:
        if not self.quiet:
            self._err.print(f"[yellow]! {message}[/yellow]")

    def error(self, message: str) -> None:
        # Errors are reported even in quiet mode — on stderr, so a --json
        # stdout payload stays machine-parsable.
        self._err.print(f"[red]✗ {message}[/red]")

    def banner(self, *, model: str, workspace: str, mode: str, sandbox: str) -> None:
        if self.quiet:
            return
        self._c.print(
            Panel.fit(
                Text.from_markup(
                    "[bold]Forge[/bold] — autonomous software-engineering agent\n"
                    f"[dim]model:[/dim] {model}\n"
                    f"[dim]workspace:[/dim] {workspace}\n"
                    f"[dim]approvals:[/dim] {mode}   [dim]sandbox:[/dim] {sandbox}\n"
                    "[dim]Type a task, /help for commands, or 'exit' to quit.[/dim]"
                ),
                border_style="cyan",
            )
        )

    def status(self, message: str) -> Any:
        """A spinner context manager shown while the model is thinking."""
        if self.quiet:
            return _NullStatus()
        return self._c.status(f"[cyan]{message}[/cyan]", spinner="dots")

    # --- agent output ---
    def assistant_text(self, text: str) -> None:
        if self.quiet:
            return
        if stripped := text.strip():
            self._c.print(Markdown(stripped))

    def tool_call(self, name: str, args: dict[str, Any]) -> None:
        if self.quiet:
            return
        line = Text()
        line.append("⚙ ", style="cyan")
        line.append(name, style="bold cyan")
        if preview := _args_preview(name, args):
            line.append(f"  {preview}", style="dim")
        self._c.print(line)

    def tool_outcome(self, outcome: ToolOutcome, *, max_lines: int = 6) -> None:
        if self.quiet:
            return
        content = outcome.block.content
        lines = content.splitlines()
        preview = "\n".join(lines[:max_lines])
        if len(lines) > max_lines:
            preview += f"\n[... +{len(lines) - max_lines} more lines ...]"
        style = "red" if outcome.is_error else "green"
        marker = "✗" if outcome.is_error else "✓"
        title = f"[{style}]{marker} {outcome.name}[/{style}]"
        if outcome.duration_s >= 1.0:
            title += f" [dim]{format_duration(outcome.duration_s)}[/dim]"
        self._c.print(
            Panel(
                Text(preview or "(no output)"),
                title=title,
                border_style=style,
                title_align="left",
                padding=(0, 1),
            )
        )

    def denied(self, name: str, reason: str) -> None:
        if not self.quiet:
            self._c.print(f"[red]✗ {name} denied[/red] [dim]({reason})[/dim]")

    # --- approval ---
    def ask_approval(self, name: str, target: str, perm: PermissionResult) -> Approval:
        """Prompt the human to allow/deny a gated tool call.

        Falls back to a safe DENY when there is no interactive terminal, so a CI
        run never blocks forever waiting for input nobody can give.
        """
        if self.quiet or not sys.stdin.isatty():
            self.warning(f"Non-interactive session; refusing gated call: {name}")
            return Approval.DENY

        style = _RISK_STYLE.get(perm.risk, "yellow")
        self._c.print(
            Panel(
                Text.from_markup(
                    f"[bold]Forge wants to run:[/bold] [cyan]{name}[/cyan]\n"
                    f"[dim]{target}[/dim]\n"
                    f"[{style}]risk: {perm.risk.value}[/{style}] [dim]({perm.reason})[/dim]"
                ),
                title="[yellow]⚠ approval required[/yellow]",
                border_style="yellow",
                title_align="left",
            )
        )
        choice = Prompt.ask(
            "  [a]llow once / [d]eny / [always] allow this tool",
            choices=["a", "d", "always"],
            default="d",
        )
        return {"a": Approval.ALLOW, "always": Approval.ALWAYS, "d": Approval.DENY}[choice]

    # --- summary ---
    def run_summary(self, trace: RunTrace) -> None:
        if self.quiet:
            return
        table = Table.grid(padding=(0, 2))
        table.add_column(justify="right", style="dim")
        table.add_column()

        cost = trace.estimated_cost
        cost_str = f"${cost:.4f} (est.)" if cost is not None else "n/a"
        status_style = {
            "completed": "green",
            "error": "red",
            "aborted": "yellow",
        }.get(trace.status, "yellow")

        table.add_row("Task", trace.task[:70] + ("..." if len(trace.task) > 70 else ""))
        table.add_row("Duration", format_duration(trace.duration_s))
        table.add_row("Model calls", str(trace.num_model_calls))
        table.add_row("Tool calls", f"{trace.num_tool_calls} ({trace.num_tool_errors} errors)")
        if tools := trace.tool_usage_counts():
            table.add_row("Tools used", ", ".join(f"{n}×{c}" for n, c in tools.items()))
        table.add_row(
            "Tokens",
            f"{trace.total_tokens:,} "
            f"[dim](in {trace.input_tokens:,} / out {trace.output_tokens:,})[/dim]",
        )
        table.add_row("Cost", cost_str)
        if trace.compactions:
            table.add_row("Compactions", str(trace.compactions))
        if trace.num_retries:
            table.add_row("Retries", str(trace.num_retries))
        table.add_row("Status", f"[{status_style}]{trace.status}[/{status_style}]")

        self._c.print(
            Panel(
                table,
                title=f"[bold]Forge run #{trace.run_id}[/bold]",
                border_style=status_style,
                title_align="left",
            )
        )


class _NullStatus:
    """Stand-in for the Rich spinner when output is suppressed."""

    def __enter__(self) -> _NullStatus:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def console_approver(console: Console) -> Any:
    """Adapt :meth:`Console.ask_approval` to the async ``Approver`` protocol.

    The prompt blocks on stdin, so it runs in a worker thread to avoid stalling
    the event loop (and any in-flight tool work) while a human decides.
    """

    async def approve(tool: Tool[Any], target: str, perm: PermissionResult) -> Approval:
        return await asyncio.to_thread(console.ask_approval, tool.name, target, perm)

    return approve
