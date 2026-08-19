"""Filesystem tools: read, write, edit, apply_patch, list, search.

All paths are resolved through :meth:`ToolContext.resolve_path`, which confines
access to the workspace root (symlink escapes included). Expected failures are
returned as ``ToolResult.error`` so the model can read and recover from them.
"""

from __future__ import annotations

import fnmatch
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from forge.tools.base import Risk, Tool, ToolResult
from forge.tools.context import PathOutsideWorkspaceError, ToolContext

# Directories that are noise for search/listing.
IGNORE_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        "dist",
        "build",
        ".idea",
        ".tox",
        ".egg-info",
    }
)

_READ_LIMIT = 200_000
_SEARCH_FILE_LIMIT = 2_000_000


def _looks_binary(data: bytes) -> bool:
    return b"\x00" in data


# --------------------------------------------------------------------------- #
# Unified-diff application (exact-context, deterministic)
# --------------------------------------------------------------------------- #
def _parse_hunks(patch: str) -> list[tuple[list[str], list[str]]]:
    """Parse a unified diff into a list of (before_lines, after_lines) hunks."""
    hunks: list[tuple[list[str], list[str]]] = []
    before: list[str] = []
    after: list[str] = []
    in_hunk = False
    for raw in patch.splitlines():
        if raw.startswith("@@"):
            if in_hunk:
                hunks.append((before, after))
            before, after = [], []
            in_hunk = True
            continue
        if not in_hunk:
            continue  # skip ---/+++/diff/index headers before the first hunk
        if raw.startswith("\\"):
            continue  # "\ No newline at end of file"
        tag, text = raw[:1], raw[1:]
        if tag in (" ", ""):
            before.append(text)
            after.append(text)
        elif tag == "-":
            before.append(text)
        elif tag == "+":
            after.append(text)
        else:
            raise ValueError(f"Malformed patch line: {raw!r}")
    if in_hunk:
        hunks.append((before, after))
    return hunks


def _find_block(lines: list[str], block: list[str], start: int) -> int | None:
    m = len(block)
    for i in range(start, len(lines) - m + 1):
        if lines[i : i + m] == block:
            return i
    return None


def apply_unified_diff(original: str, patch: str) -> str:
    """Apply ``patch`` to ``original`` requiring exact context matches.

    Raises ``ValueError`` if the patch is malformed or its context does not
    match, so the caller can surface an actionable error.
    """
    hunks = _parse_hunks(patch)
    if not hunks:
        raise ValueError("No @@ hunks found in patch")

    result = original.splitlines()
    search_from = 0
    for before, after in hunks:
        if not before:
            raise ValueError("Hunk has no context/removed lines; cannot locate it")
        idx = _find_block(result, before, search_from)
        if idx is None:  # hunks may be out of order — retry from the top
            idx = _find_block(result, before, 0)
        if idx is None:
            raise ValueError("Patch context did not match the file contents")
        result[idx : idx + len(before)] = after
        search_from = idx + len(after)

    text = "\n".join(result)
    if original.endswith("\n"):
        text += "\n"
    return text


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #
class ReadFileArgs(BaseModel):
    path: str = Field(description="Path to the file, relative to the workspace root.")
    offset: int = Field(
        default=0, ge=0, description="Line number to start reading from (0-based). For large files."
    )
    limit: int | None = Field(
        default=None, description="Maximum number of lines to return. Omit to read to the end."
    )
    max_bytes: int = Field(
        default=_READ_LIMIT,
        description="Maximum number of characters to return before truncating.",
    )


class ReadFile(Tool[ReadFileArgs]):
    name = "read_file"
    description = (
        "Read a UTF-8 text file and return its contents. Supports offset/limit for "
        "reading a slice of a large file."
    )
    risk = Risk.READ
    Args = ReadFileArgs

    async def run(self, args: ReadFileArgs, ctx: ToolContext) -> ToolResult:
        try:
            path = ctx.resolve_path(args.path)
        except PathOutsideWorkspaceError as exc:
            return ToolResult.error(str(exc))
        if not path.exists():
            return ToolResult.error(f"File not found: {args.path}")
        if path.is_dir():
            return ToolResult.error(f"{args.path} is a directory; use list_directory")
        try:
            data = path.read_bytes()
        except OSError as exc:
            return ToolResult.error(f"Could not read {args.path}: {exc}")
        if _looks_binary(data):
            return ToolResult.error(f"{args.path} appears to be a binary file")

        text = data.decode("utf-8", errors="replace")
        notes: list[str] = []

        if args.offset or args.limit is not None:
            lines = text.splitlines()
            total = len(lines)
            if args.offset >= total and total:
                return ToolResult.error(
                    f"offset {args.offset} is past the end of {args.path} ({total} lines)"
                )
            end = total if args.limit is None else min(total, args.offset + args.limit)
            text = "\n".join(lines[args.offset : end])
            if args.offset or end < total:
                notes.append(f"[lines {args.offset + 1}-{end} of {total}]")

        if len(text) > args.max_bytes:
            text = text[: args.max_bytes]
            notes.append("[... truncated ...]")

        body = text if not notes else f"{text}\n\n{' '.join(notes)}"
        return ToolResult.ok(body, truncated=bool(notes), bytes=len(data))


class WriteFileArgs(BaseModel):
    path: str = Field(description="Path to write, relative to the workspace root.")
    content: str = Field(description="Full file contents to write.")


class WriteFile(Tool[WriteFileArgs]):
    name = "write_file"
    description = "Create or overwrite a file with the given contents."
    risk = Risk.WRITE
    Args = WriteFileArgs

    async def run(self, args: WriteFileArgs, ctx: ToolContext) -> ToolResult:
        try:
            path = ctx.resolve_path(args.path)
        except PathOutsideWorkspaceError as exc:
            return ToolResult.error(str(exc))
        existed = path.exists()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(args.content, encoding="utf-8")
        except OSError as exc:
            return ToolResult.error(f"Could not write {args.path}: {exc}")
        verb = "Overwrote" if existed else "Created"
        return ToolResult.ok(
            f"{verb} {ctx.relative(path)} ({len(args.content)} chars)",
            created=not existed,
        )


class EditFileArgs(BaseModel):
    path: str = Field(description="Path to the file to edit.")
    old_string: str = Field(description="Exact text to find.")
    new_string: str = Field(description="Replacement text.")
    replace_all: bool = Field(
        default=False, description="Replace every occurrence instead of requiring uniqueness."
    )


class EditFile(Tool[EditFileArgs]):
    name = "edit_file"
    description = (
        "Replace an exact string in a file. Fails if old_string is missing, or "
        "matches more than once and replace_all is false."
    )
    risk = Risk.WRITE
    Args = EditFileArgs

    async def run(self, args: EditFileArgs, ctx: ToolContext) -> ToolResult:
        try:
            path = ctx.resolve_path(args.path)
        except PathOutsideWorkspaceError as exc:
            return ToolResult.error(str(exc))
        if not path.is_file():
            return ToolResult.error(f"File not found: {args.path}")
        if args.old_string == args.new_string:
            return ToolResult.error("old_string and new_string are identical")
        text = path.read_text(encoding="utf-8", errors="replace")
        count = text.count(args.old_string)
        if count == 0:
            return ToolResult.error(f"old_string not found in {args.path}")
        if count > 1 and not args.replace_all:
            return ToolResult.error(
                f"old_string matches {count} times in {args.path}; add context "
                "to make it unique or set replace_all=true"
            )
        new_text = (
            text.replace(args.old_string, args.new_string)
            if args.replace_all
            else text.replace(args.old_string, args.new_string, 1)
        )
        try:
            path.write_text(new_text, encoding="utf-8")
        except OSError as exc:
            return ToolResult.error(f"Could not write {args.path}: {exc}")
        n = count if args.replace_all else 1
        return ToolResult.ok(f"Edited {ctx.relative(path)} ({n} replacement(s))")


class ApplyPatchArgs(BaseModel):
    path: str = Field(description="File the patch applies to.")
    patch: str = Field(
        description="Unified-diff hunks (@@ ... @@ with ' ', '-', '+' lines). "
        "Context must match the file exactly."
    )


class ApplyPatch(Tool[ApplyPatchArgs]):
    name = "apply_patch"
    description = "Apply unified-diff hunks to an existing file (exact context match)."
    risk = Risk.WRITE
    Args = ApplyPatchArgs

    async def run(self, args: ApplyPatchArgs, ctx: ToolContext) -> ToolResult:
        try:
            path = ctx.resolve_path(args.path)
        except PathOutsideWorkspaceError as exc:
            return ToolResult.error(str(exc))
        if not path.is_file():
            return ToolResult.error(f"File not found: {args.path} (use write_file to create it)")
        text = path.read_text(encoding="utf-8", errors="replace")
        try:
            new_text = apply_unified_diff(text, args.patch)
        except ValueError as exc:
            return ToolResult.error(f"Could not apply patch: {exc}")
        try:
            path.write_text(new_text, encoding="utf-8")
        except OSError as exc:
            return ToolResult.error(f"Could not write {args.path}: {exc}")
        return ToolResult.ok(f"Applied patch to {ctx.relative(path)}")


class ListDirectoryArgs(BaseModel):
    path: str = Field(default=".", description="Directory to list, relative to the workspace.")
    recursive: bool = Field(default=False, description="Recurse into subdirectories.")
    max_entries: int = Field(default=200, description="Maximum entries to return.")


class ListDirectory(Tool[ListDirectoryArgs]):
    name = "list_directory"
    description = "List directory contents (optionally recursive)."
    risk = Risk.READ
    Args = ListDirectoryArgs

    async def run(self, args: ListDirectoryArgs, ctx: ToolContext) -> ToolResult:
        try:
            root = ctx.resolve_path(args.path)
        except PathOutsideWorkspaceError as exc:
            return ToolResult.error(str(exc))
        if not root.exists():
            return ToolResult.error(f"Directory not found: {args.path}")
        if not root.is_dir():
            return ToolResult.error(f"{args.path} is not a directory")

        entries: list[str] = []
        capped = False
        if args.recursive:
            for dirpath, dirnames, filenames in _walk(root):
                dirnames.sort()
                for name in sorted(filenames):
                    rel = ctx.relative(Path(dirpath) / name)
                    entries.append(rel)
                    if len(entries) >= args.max_entries:
                        capped = True
                        break
                if capped:
                    break
        else:
            children = sorted(root.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
            for child in children:
                if child.is_dir():
                    entries.append(f"{child.name}/")
                else:
                    size = child.stat().st_size if child.exists() else 0
                    entries.append(f"{child.name} ({size} bytes)")
                if len(entries) >= args.max_entries:
                    capped = True
                    break

        if not entries:
            return ToolResult.ok("(empty directory)")
        body = "\n".join(entries)
        if capped:
            body += f"\n[... truncated at {args.max_entries} entries ...]"
        return ToolResult.ok(body, count=len(entries), truncated=capped)


class SearchFilesArgs(BaseModel):
    pattern: str = Field(description="Regular expression to search for.")
    path: str = Field(default=".", description="Directory to search under.")
    glob: str | None = Field(default=None, description="Optional filename glob, e.g. '*.py'.")
    ignore_case: bool = Field(default=False, description="Case-insensitive match.")
    max_results: int = Field(default=100, description="Maximum matching lines to return.")


class SearchFiles(Tool[SearchFilesArgs]):
    name = "search_files"
    description = "Search file contents by regex, returning 'path:line: text' matches."
    risk = Risk.READ
    Args = SearchFilesArgs

    async def run(self, args: SearchFilesArgs, ctx: ToolContext) -> ToolResult:
        try:
            root = ctx.resolve_path(args.path)
        except PathOutsideWorkspaceError as exc:
            return ToolResult.error(str(exc))
        if not root.exists():
            return ToolResult.error(f"Path not found: {args.path}")
        try:
            regex = re.compile(args.pattern, re.IGNORECASE if args.ignore_case else 0)
        except re.error as exc:
            return ToolResult.error(f"Invalid regex: {exc}")

        results: list[str] = []
        capped = False
        for dirpath, dirnames, filenames in _walk(root):
            dirnames.sort()
            for name in sorted(filenames):
                if args.glob and not fnmatch.fnmatch(name, args.glob):
                    continue
                fpath = Path(dirpath) / name
                try:
                    data = fpath.read_bytes()
                except OSError:
                    continue
                if len(data) > _SEARCH_FILE_LIMIT or _looks_binary(data):
                    continue
                text = data.decode("utf-8", errors="replace")
                for lineno, line in enumerate(text.splitlines(), start=1):
                    if regex.search(line):
                        rel = ctx.relative(fpath)
                        results.append(f"{rel}:{lineno}: {line.strip()[:200]}")
                        if len(results) >= args.max_results:
                            capped = True
                            break
                if capped:
                    break
            if capped:
                break

        if not results:
            return ToolResult.ok("(no matches)")
        body = "\n".join(results)
        if capped:
            body += f"\n[... truncated at {args.max_results} matches ...]"
        return ToolResult.ok(body, count=len(results), truncated=capped)


def _walk(root: Path) -> list[tuple[str, list[str], list[str]]]:
    """os.walk over ``root`` with ignore-dirs pruned. Returns a materialized list."""
    import os

    out: list[tuple[str, list[str], list[str]]] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS]
        out.append((dirpath, dirnames, filenames))
    return out


def filesystem_tools() -> list[Tool[Any]]:
    return [
        ReadFile(),
        WriteFile(),
        EditFile(),
        ApplyPatch(),
        ListDirectory(),
        SearchFiles(),
    ]
