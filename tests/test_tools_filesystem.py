"""Tests for the filesystem tools and the workspace boundary."""

from __future__ import annotations

from pathlib import Path

import pytest

from forge.tools.context import PathOutsideWorkspaceError, ToolContext
from forge.tools.filesystem import (
    ApplyPatch,
    EditFile,
    ListDirectory,
    ReadFile,
    SearchFiles,
    WriteFile,
    apply_unified_diff,
    filesystem_tools,
)


async def run_tool(tool: object, ctx: ToolContext, **args: object) -> object:
    """Validate args the way the executor does, then run."""
    return await tool.run(tool.parse_args(args), ctx)  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- #
# read_file
# --------------------------------------------------------------------------- #
async def test_read_file(workspace: Path, ctx: ToolContext) -> None:
    (workspace / "a.txt").write_text("hello\nworld\n")
    result = await run_tool(ReadFile(), ctx, path="a.txt")
    assert not result.is_error  # type: ignore[attr-defined]
    assert "hello" in result.content  # type: ignore[attr-defined]


async def test_read_missing_file_is_an_error(ctx: ToolContext) -> None:
    result = await run_tool(ReadFile(), ctx, path="nope.txt")
    assert result.is_error and "not found" in result.content  # type: ignore[attr-defined]


async def test_read_directory_points_at_the_right_tool(workspace: Path, ctx: ToolContext) -> None:
    (workspace / "sub").mkdir()
    result = await run_tool(ReadFile(), ctx, path="sub")
    assert result.is_error and "list_directory" in result.content  # type: ignore[attr-defined]


async def test_read_binary_file_is_refused(workspace: Path, ctx: ToolContext) -> None:
    (workspace / "blob.bin").write_bytes(b"\x89PNG\x00\x01\x02")
    result = await run_tool(ReadFile(), ctx, path="blob.bin")
    assert result.is_error and "binary" in result.content  # type: ignore[attr-defined]


async def test_read_slice_with_offset_and_limit(workspace: Path, ctx: ToolContext) -> None:
    (workspace / "many.txt").write_text("\n".join(f"line{i}" for i in range(1, 21)))
    result = await run_tool(ReadFile(), ctx, path="many.txt", offset=5, limit=3)

    body = result.content  # type: ignore[attr-defined]
    assert "line6\nline7\nline8" in body
    assert "line5" not in body and "line9" not in body
    assert "[lines 6-8 of 20]" in body


async def test_read_offset_past_the_end_is_an_error(workspace: Path, ctx: ToolContext) -> None:
    (workspace / "short.txt").write_text("only one line\n")
    result = await run_tool(ReadFile(), ctx, path="short.txt", offset=99)
    assert result.is_error and "past the end" in result.content  # type: ignore[attr-defined]


async def test_read_truncates_at_max_bytes(workspace: Path, ctx: ToolContext) -> None:
    (workspace / "big.txt").write_text("x" * 5_000)
    result = await run_tool(ReadFile(), ctx, path="big.txt", max_bytes=100)
    assert "truncated" in result.content  # type: ignore[attr-defined]
    assert result.metadata["truncated"] is True  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- #
# write_file / edit_file
# --------------------------------------------------------------------------- #
async def test_write_creates_then_overwrites(workspace: Path, ctx: ToolContext) -> None:
    created = await run_tool(WriteFile(), ctx, path="new.txt", content="one")
    assert "Created" in created.content  # type: ignore[attr-defined]
    assert created.metadata["created"] is True  # type: ignore[attr-defined]

    overwritten = await run_tool(WriteFile(), ctx, path="new.txt", content="two")
    assert "Overwrote" in overwritten.content  # type: ignore[attr-defined]
    assert (workspace / "new.txt").read_text() == "two"


async def test_write_creates_parent_directories(workspace: Path, ctx: ToolContext) -> None:
    await run_tool(WriteFile(), ctx, path="deep/nested/x.py", content="pass\n")
    assert (workspace / "deep" / "nested" / "x.py").read_text() == "pass\n"


async def test_edit_replaces_a_unique_string(workspace: Path, ctx: ToolContext) -> None:
    (workspace / "m.py").write_text("value = 1\nother = 2\n")
    result = await run_tool(
        EditFile(), ctx, path="m.py", old_string="value = 1", new_string="value = 42"
    )
    assert not result.is_error  # type: ignore[attr-defined]
    assert (workspace / "m.py").read_text() == "value = 42\nother = 2\n"


async def test_edit_refuses_an_ambiguous_match(workspace: Path, ctx: ToolContext) -> None:
    (workspace / "m.py").write_text("x = 1\nx = 1\n")
    result = await run_tool(EditFile(), ctx, path="m.py", old_string="x = 1", new_string="x = 2")
    assert result.is_error  # type: ignore[attr-defined]
    assert "matches 2 times" in result.content  # type: ignore[attr-defined]
    assert (workspace / "m.py").read_text() == "x = 1\nx = 1\n", "no partial edit"


async def test_edit_replace_all(workspace: Path, ctx: ToolContext) -> None:
    (workspace / "m.py").write_text("x = 1\nx = 1\n")
    result = await run_tool(
        EditFile(), ctx, path="m.py", old_string="x = 1", new_string="x = 2", replace_all=True
    )
    assert "2 replacement" in result.content  # type: ignore[attr-defined]
    assert (workspace / "m.py").read_text() == "x = 2\nx = 2\n"


async def test_edit_missing_string_is_an_error(workspace: Path, ctx: ToolContext) -> None:
    (workspace / "m.py").write_text("nothing here\n")
    result = await run_tool(EditFile(), ctx, path="m.py", old_string="absent", new_string="x")
    assert result.is_error and "not found" in result.content  # type: ignore[attr-defined]


async def test_edit_rejects_a_no_op(workspace: Path, ctx: ToolContext) -> None:
    (workspace / "m.py").write_text("same\n")
    result = await run_tool(EditFile(), ctx, path="m.py", old_string="same", new_string="same")
    assert result.is_error and "identical" in result.content  # type: ignore[attr-defined]


async def test_edit_missing_file_is_an_error(ctx: ToolContext) -> None:
    result = await run_tool(EditFile(), ctx, path="ghost.py", old_string="a", new_string="b")
    assert result.is_error and "not found" in result.content  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- #
# apply_patch
# --------------------------------------------------------------------------- #
def test_apply_unified_diff_replaces_a_line() -> None:
    original = "def f():\n    return 1\n"
    patch = "@@ -1,2 +1,2 @@\n def f():\n-    return 1\n+    return 2\n"
    assert apply_unified_diff(original, patch) == "def f():\n    return 2\n"


def test_apply_unified_diff_handles_multiple_hunks() -> None:
    original = "a\nb\nc\nd\ne\nf\ng\nh\n"
    patch = "@@ -1,2 +1,2 @@\n a\n-b\n+B\n@@ -7,2 +7,2 @@\n g\n-h\n+H\n"
    assert apply_unified_diff(original, patch) == "a\nB\nc\nd\ne\nf\ng\nH\n"


def test_apply_unified_diff_preserves_a_missing_trailing_newline() -> None:
    assert apply_unified_diff("a\nb", "@@ @@\n a\n-b\n+B\n") == "a\nB"


@pytest.mark.parametrize(
    ("original", "patch", "match"),
    [
        ("a\n", "no hunks here\n", "No @@ hunks"),
        ("a\n", "@@ @@\n zzz\n-zzz\n+y\n", "context did not match"),
        ("a\n", "@@ @@\n+added\n", "cannot locate"),
    ],
)
def test_apply_unified_diff_rejects_bad_patches(original: str, patch: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        apply_unified_diff(original, patch)


async def test_apply_patch_tool(workspace: Path, ctx: ToolContext) -> None:
    (workspace / "p.py").write_text("def f():\n    return 1\n")
    result = await run_tool(
        ApplyPatch(),
        ctx,
        path="p.py",
        patch="@@ -1,2 +1,2 @@\n def f():\n-    return 1\n+    return 2\n",
    )
    assert not result.is_error  # type: ignore[attr-defined]
    assert (workspace / "p.py").read_text() == "def f():\n    return 2\n"


async def test_apply_patch_reports_a_context_mismatch(workspace: Path, ctx: ToolContext) -> None:
    (workspace / "p.py").write_text("actual content\n")
    result = await run_tool(ApplyPatch(), ctx, path="p.py", patch="@@ @@\n-expected\n+new\n")
    assert result.is_error and "Could not apply patch" in result.content  # type: ignore[attr-defined]


async def test_apply_patch_needs_an_existing_file(ctx: ToolContext) -> None:
    result = await run_tool(ApplyPatch(), ctx, path="ghost.py", patch="@@ @@\n-a\n+b\n")
    assert result.is_error and "write_file" in result.content  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- #
# list_directory / search_files
# --------------------------------------------------------------------------- #
async def test_list_directory(workspace: Path, ctx: ToolContext) -> None:
    (workspace / "pkg").mkdir()
    (workspace / "top.txt").write_text("x")
    result = await run_tool(ListDirectory(), ctx, path=".")
    assert "pkg/" in result.content and "top.txt" in result.content  # type: ignore[attr-defined]


async def test_list_directory_recursive_skips_noise(workspace: Path, ctx: ToolContext) -> None:
    (workspace / "src").mkdir()
    (workspace / "src" / "app.py").write_text("x")
    (workspace / "__pycache__").mkdir()
    (workspace / "__pycache__" / "app.cpython.pyc").write_text("junk")

    result = await run_tool(ListDirectory(), ctx, path=".", recursive=True)
    assert "src/app.py" in result.content  # type: ignore[attr-defined]
    assert "__pycache__" not in result.content  # type: ignore[attr-defined]


async def test_list_empty_directory(workspace: Path, ctx: ToolContext) -> None:
    (workspace / "void").mkdir()
    result = await run_tool(ListDirectory(), ctx, path="void")
    assert "(empty directory)" in result.content  # type: ignore[attr-defined]


async def test_list_directory_caps_entries(workspace: Path, ctx: ToolContext) -> None:
    for i in range(20):
        (workspace / f"f{i:02d}.txt").write_text("x")
    result = await run_tool(ListDirectory(), ctx, path=".", max_entries=5)
    assert result.metadata["truncated"] is True  # type: ignore[attr-defined]
    assert "truncated at 5 entries" in result.content  # type: ignore[attr-defined]


async def test_list_missing_directory(ctx: ToolContext) -> None:
    result = await run_tool(ListDirectory(), ctx, path="ghost")
    assert result.is_error and "not found" in result.content  # type: ignore[attr-defined]


async def test_search_files(workspace: Path, ctx: ToolContext) -> None:
    (workspace / "a.py").write_text("def target():\n    pass\n")
    (workspace / "b.txt").write_text("target here too\n")
    result = await run_tool(SearchFiles(), ctx, pattern=r"target")

    assert "a.py:1:" in result.content and "b.txt:1:" in result.content  # type: ignore[attr-defined]
    assert result.metadata["count"] == 2  # type: ignore[attr-defined]


async def test_search_respects_a_glob(workspace: Path, ctx: ToolContext) -> None:
    (workspace / "a.py").write_text("needle\n")
    (workspace / "b.txt").write_text("needle\n")
    result = await run_tool(SearchFiles(), ctx, pattern="needle", glob="*.py")
    assert "a.py" in result.content and "b.txt" not in result.content  # type: ignore[attr-defined]


async def test_search_is_case_insensitive_on_request(workspace: Path, ctx: ToolContext) -> None:
    (workspace / "a.py").write_text("NEEDLE\n")
    assert "(no matches)" in (await run_tool(SearchFiles(), ctx, pattern="needle")).content  # type: ignore[attr-defined]
    hit = await run_tool(SearchFiles(), ctx, pattern="needle", ignore_case=True)
    assert "a.py" in hit.content  # type: ignore[attr-defined]


async def test_search_reports_an_invalid_regex(ctx: ToolContext) -> None:
    result = await run_tool(SearchFiles(), ctx, pattern="a(")
    assert result.is_error and "Invalid regex" in result.content  # type: ignore[attr-defined]


async def test_search_skips_binaries(workspace: Path, ctx: ToolContext) -> None:
    (workspace / "blob.bin").write_bytes(b"needle\x00\x01")
    result = await run_tool(SearchFiles(), ctx, pattern="needle")
    assert "(no matches)" in result.content  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- #
# Workspace containment
# --------------------------------------------------------------------------- #
def test_resolve_path_accepts_the_root_and_its_children(ctx: ToolContext) -> None:
    assert ctx.resolve_path(".") == ctx.workspace_root
    assert ctx.resolve_path("a/b.py") == ctx.workspace_root / "a" / "b.py"


@pytest.mark.parametrize("path", ["../escape.txt", "../../etc/passwd", "/etc/passwd", "a/../../x"])
def test_resolve_path_rejects_escapes(ctx: ToolContext, path: str) -> None:
    with pytest.raises(PathOutsideWorkspaceError):
        ctx.resolve_path(path)


def test_resolve_path_follows_symlinks_before_checking(workspace: Path, ctx: ToolContext) -> None:
    """A link pointing outside the workspace is still outside the workspace."""
    outside = workspace.parent / "outside-target"
    outside.mkdir(exist_ok=True)
    (workspace / "link").symlink_to(outside)
    with pytest.raises(PathOutsideWorkspaceError):
        ctx.resolve_path("link")


@pytest.mark.parametrize("tool", filesystem_tools())
async def test_escaping_calls_become_error_results(tool: object, ctx: ToolContext) -> None:
    """The boundary is reported to the model, never raised at the loop."""
    args = {
        "path": "../escape.txt",
        "content": "x",
        "old_string": "a",
        "new_string": "b",
        "patch": "@@ @@\n-a\n+b\n",
        "pattern": "x",
    }
    accepted = {k: v for k, v in args.items() if k in tool.Args.model_fields}  # type: ignore[attr-defined]
    result = await run_tool(tool, ctx, **accepted)
    assert result.is_error and "outside the workspace" in result.content  # type: ignore[attr-defined]


def test_relative_formats_paths_for_display(ctx: ToolContext) -> None:
    assert ctx.relative(ctx.workspace_root / "src" / "a.py") == "src/a.py"
    assert ctx.relative(Path("/tmp")) == "/tmp"
