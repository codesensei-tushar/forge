# Forge

**An open-source, terminal-native autonomous software-engineering agent.**

Forge is a mini "coding agent runtime": the interesting engineering is the
**harness** — a provider abstraction, a tool registry, a permission policy, a
robust tool-use loop, and per-run observability — not just the prompt.

```
forge
› Fix the authentication bug in this repository, add a test, and run the suite.
```

Forge inspects the repo, plans, reads and edits files, runs commands, observes
failures, and iterates — pausing to ask you before anything destructive.

> **Status:** Phase 1 (core agent). Filesystem + shell tools, a pluggable model
> provider (Anthropic-compatible), human-in-the-loop approvals, and a run-trace
> summary. The architecture is built to grow through git tooling, a Docker
> sandbox, memory/context ranking, planning & multi-agent, MCP/GitHub, and a
> FastAPI/Postgres control plane.

## Install

Requires Python 3.11+ and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync                      # create the venv and install
uv run forge version
```

## Configure

Forge speaks the Anthropic Messages API and honors the standard environment, so
an existing Anthropic-compatible gateway works out of the box:

```bash
export ANTHROPIC_BASE_URL="https://your-gateway/"   # optional
export ANTHROPIC_AUTH_TOKEN="..."                    # or ANTHROPIC_API_KEY
export ANTHROPIC_MODEL="claude-..."                  # the model to use
```

Configuration precedence (highest first): **CLI flags → environment →
`./forge.toml` → `~/.config/forge/config.toml` → defaults**. See
[`.env.example`](.env.example) and [`forge.toml`](forge.toml).

```bash
uv run forge config          # show the resolved configuration (secrets redacted)
```

## Use

```bash
uv run forge                             # interactive REPL (default)
uv run forge run "read the LICENSE and summarize it in one line"
uv run forge -C /path/to/repo --yes run "add a --version flag and test it"
```

Global options go before the subcommand (git-style): `--model`, `--provider`,
`-C/--workspace`, `--yes` (auto-approve), `--max-iterations`, `--verbose`.

## How it works

```
CLI (Typer/Rich) → AgentRuntime loop → ModelProvider (normalized messages)
                         │
                ToolRegistry + PermissionPolicy
                         │
             filesystem + shell tools  (confined to the workspace)
                         │
                     RunTrace → summary + structured logs
```

**Tools** — `read_file`, `write_file`, `edit_file`, `apply_patch`,
`list_directory`, `search_files`, `shell`. Each declares a Pydantic argument
schema; the loop validates model input before running it, and confines all
paths to the workspace root.

**Permissions** — read-only tools run automatically; writes and shell prompt for
approval (Allow / Deny / Always-allow). Deny patterns (`rm -rf /`, `sudo`,
`git push`, …) are always refused. `--yes` auto-approves.

**Robust loop** — validation errors and tool exceptions are fed back to the
model as tool results (it recovers instead of crashing); hard guards on
iteration count and context size; graceful Ctrl-C.

## Develop

```bash
uv run pytest          # offline test suite (uses a FakeProvider — no network)
uv run ruff check
uv run mypy src
```

## License

MIT © 2026 Tushar Umbarkar
