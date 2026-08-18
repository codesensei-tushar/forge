"""Terminal UI rendering with Rich.

The single place that touches the terminal: assistant output, tool-call and
result rendering, the human approval prompt, and the end-of-run summary.
"""

from __future__ import annotations

import sys
from enum import Enum
from typing import Any

from rich.console import Console as RichConsole
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from forge.observability.trace import RunTrace
from forge.permissions.policy import PermissionResult
from forge.tools.base import ToolResult


class Approval(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    ALWAYS = "always"


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}m {secs:02d}s"


def _args_preview(tool_name: str, args: dict[str, Any]) -> str:
    if tool_name == "shell":
        return str(args.get("command", ""))
    bits = []
    for key in ("path", "pattern", "glob", "old_string", "new_string"):
        if key in args and args[key] is not None:
            val = str(args[key]).replace("\n", "\\n")
            if len(val) > 60:
                val = val[:57] + "..."
            bits.append(f"{key}={val}")
    return "  ".join(bits) if bits else ""


class Console:
    def __init__(self, *, quiet: bool = False) -> None:
        self._c = RichConsole()
        self._err = RichConsole(stderr=True)
        self.quiet = quiet

    # --- generic ---
    def print(self, *args: Any, **kwargs: Any) -> None:
        self._c.print(*args, **kwargs)

    def info(self, message: str) -> None:
        if not self.quiet:
            self._c.print(f"[dim]{message}[/dim]")

    def input(self, prompt: str) -> str:
        return self._c.input(prompt)

    def warning(self, message: str) -> None:
        self._err.print(f"[yellow]! {message}[/yellow]")

    def error(self, message: str) -> None:
        self._err.print(f"[red]✗ {message}[/red]")

    def banner(self, *, model: str, workspace: str) -> None:
        self._c.print(
            Panel.fit(
                Text.from_markup(
                    "[bold]Forge[/bold] — autonomous software-engineering agent\n"
                    f"[dim]model:[/dim] {model}   [dim]workspace:[/dim] {workspace}\n"
                    "[dim]Type a task, or 'exit' to quit.[/dim]"
                ),
                border_style="cyan",
            )
        )

    def status(self, message: str) -> Any:
        """A spinner context manager shown while the model is thinking."""
        return self._c.status(f"[cyan]{message}[/cyan]", spinner="dots")

    # --- agent output ---
    def assistant_text(self, text: str) -> None:
        text = text.strip()
        if text:
            self._c.print(Markdown(text))

    def tool_call(self, name: str, args: dict[str, Any]) -> None:
        preview = _args_preview(name, args)
        line = Text()
        line.append("⚙ ", style="cyan")
        line.append(name, style="bold cyan")
        if preview:
            line.append(f"  {preview}", style="dim")
        self._c.print(line)

    def tool_result(self, name: str, result: ToolResult, *, max_lines: int = 6) -> None:
        lines = result.content.splitlines()
        preview = "\n".join(lines[:max_lines])
        if len(lines) > max_lines:
            preview += f"\n[... +{len(lines) - max_lines} lines ...]"
        style = "red" if result.is_error else "green"
        marker = "✗" if result.is_error else "✓"
        self._c.print(
            Panel(
                Text(preview or "(no output)"),
                title=f"[{style}]{marker} {name}[/{style}]",
                border_style=style,
                title_align="left",
                padding=(0, 1),
            )
        )

    def denied(self, name: str, reason: str) -> None:
        self._c.print(f"[red]✗ {name} denied[/red] [dim]({reason})[/dim]")

    # --- approval ---
    def ask_approval(self, name: str, target: str, perm: PermissionResult) -> Approval:
        """Prompt the human to allow/deny a gated tool call.

        Falls back to a safe DENY when there is no interactive terminal.
        """
        if not sys.stdin.isatty():
            self.warning(f"Non-interactive session; denying gated call: {name}")
            return Approval.DENY

        self._c.print(
            Panel(
                Text.from_markup(
                    f"[bold]Forge wants to run:[/bold] [cyan]{name}[/cyan]\n"
                    f"[dim]{target}[/dim]"
                ),
                title="[yellow]⚠ approval required[/yellow]",
                border_style="yellow",
                title_align="left",
            )
        )
        choice = Prompt.ask(
            "  [A]llow / [D]eny / allow [always] this tool",
            choices=["a", "d", "always"],
            default="d",
        )
        return {"a": Approval.ALLOW, "always": Approval.ALWAYS, "d": Approval.DENY}[choice]

    # --- summary ---
    def run_summary(self, trace: RunTrace) -> None:
        table = Table.grid(padding=(0, 2))
        table.add_column(justify="right", style="dim")
        table.add_column()

        cost = trace.estimated_cost
        cost_str = f"${cost:.4f} (est.)" if cost is not None else "n/a"
        status_style = {"completed": "green", "error": "red"}.get(trace.status, "yellow")

        table.add_row("Task", trace.task[:70] + ("..." if len(trace.task) > 70 else ""))
        table.add_row("Duration", _format_duration(trace.duration_s))
        table.add_row("Model calls", str(trace.num_model_calls))
        table.add_row(
            "Tool calls",
            f"{trace.num_tool_calls} ({trace.num_tool_errors} errors)",
        )
        table.add_row(
            "Tokens",
            f"{trace.total_tokens:,} "
            f"[dim](in {trace.input_tokens:,} / out {trace.output_tokens:,})[/dim]",
        )
        table.add_row("Cost", cost_str)
        table.add_row("Status", f"[{status_style}]{trace.status}[/{status_style}]")

        self._c.print(
            Panel(
                table,
                title=f"[bold]Forge Run #{trace.run_id}[/bold]",
                border_style=status_style,
                title_align="left",
            )
        )
