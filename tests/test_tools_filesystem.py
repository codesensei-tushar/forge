"""Tests for the filesystem tools and the unified-diff applier."""

from __future__ import annotations

from pathlib import Path

from forge.tools.context import ToolContext
from forge.tools.filesystem import (
    ApplyPatch,
    ApplyPatchArgs,
    EditFile,
    EditFileArgs,
    ListDirectory,
    ListDirectoryArgs,
    ReadFile,
    ReadFileArgs,
    SearchFiles,
    SearchFilesArgs,
    WriteFile,
    WriteFileArgs,
    apply_unified_diff,
)


async def test_write_then_read_roundtrip(ctx: ToolContext, workspace: Path) -> None:
    res = await WriteFile().run(WriteFileArgs(path="notes/a.txt", content="hello"), ctx)
    assert not res.is_error
    assert (workspace / "notes" / "a.txt").read_text() == "hello"

    read = await ReadFile().run(ReadFileArgs(path="notes/a.txt"), ctx)
    assert not read.is_error
    assert read.content == "hello"


async def test_read_missing_file_is_error(ctx: ToolContext) -> None:
    res = await ReadFile().run(ReadFileArgs(path="nope.txt"), ctx)
    assert res.is_error
    assert "not found" in res.content.lower()


async def test_path_traversal_rejected(ctx: ToolContext) -> None:
    res = await ReadFile().run(ReadFileArgs(path="../../etc/passwd"), ctx)
    assert res.is_error
    assert "outside the workspace" in res.content


async def test_edit_unique_replacement(ctx: ToolContext, workspace: Path) -> None:
    (workspace / "f.py").write_text("a = 1\nb = 2\n")
    res = await EditFile().run(
        EditFileArgs(path="f.py", old_string="b = 2", new_string="b = 3"), ctx
    )
    assert not res.is_error
    assert (workspace / "f.py").read_text() == "a = 1\nb = 3\n"


async def test_edit_ambiguous_without_replace_all(ctx: ToolContext, workspace: Path) -> None:
    (workspace / "f.py").write_text("x\nx\n")
    res = await EditFile().run(EditFileArgs(path="f.py", old_string="x", new_string="y"), ctx)
    assert res.is_error
    assert "matches 2 times" in res.content


async def test_edit_replace_all(ctx: ToolContext, workspace: Path) -> None:
    (workspace / "f.py").write_text("x\nx\n")
    res = await EditFile().run(
        EditFileArgs(path="f.py", old_string="x", new_string="y", replace_all=True), ctx
    )
    assert not res.is_error
    assert (workspace / "f.py").read_text() == "y\ny\n"


async def test_list_directory(ctx: ToolContext, workspace: Path) -> None:
    (workspace / "sub").mkdir()
    (workspace / "top.txt").write_text("t")
    res = await ListDirectory().run(ListDirectoryArgs(path="."), ctx)
    assert not res.is_error
    assert "sub/" in res.content
    assert "top.txt" in res.content


async def test_search_files(ctx: ToolContext, workspace: Path) -> None:
    (workspace / "a.py").write_text("def target():\n    pass\n")
    (workspace / "b.py").write_text("nothing here\n")
    res = await SearchFiles().run(SearchFilesArgs(pattern=r"def target", glob="*.py"), ctx)
    assert not res.is_error
    assert "a.py:1:" in res.content
    assert "b.py" not in res.content


async def test_apply_patch_tool(ctx: ToolContext, workspace: Path) -> None:
    (workspace / "f.txt").write_text("one\ntwo\nthree\n")
    patch = "@@ -1,3 +1,3 @@\n one\n-two\n+TWO\n three\n"
    res = await ApplyPatch().run(ApplyPatchArgs(path="f.txt", patch=patch), ctx)
    assert not res.is_error
    assert (workspace / "f.txt").read_text() == "one\nTWO\nthree\n"


async def test_apply_patch_context_mismatch(ctx: ToolContext, workspace: Path) -> None:
    (workspace / "f.txt").write_text("one\ntwo\n")
    patch = "@@ -1,1 +1,1 @@\n-nonexistent\n+x\n"
    res = await ApplyPatch().run(ApplyPatchArgs(path="f.txt", patch=patch), ctx)
    assert res.is_error


def test_apply_unified_diff_multi_hunk() -> None:
    original = "a\nb\nc\nd\ne\n"
    patch = "@@ -1,2 +1,2 @@\n a\n-b\n+B\n@@ -4,2 +4,2 @@\n d\n-e\n+E\n"
    assert apply_unified_diff(original, patch) == "a\nB\nc\nd\nE\n"
