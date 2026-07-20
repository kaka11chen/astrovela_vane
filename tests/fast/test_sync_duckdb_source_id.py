# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
from pathlib import Path

from scripts.sync_duckdb_source_id import filesystem_tree_id


def _git(repository: Path, *args: str) -> str:
    result = subprocess.run(
        ("git", *args),
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def test_filesystem_tree_id_matches_git_tree_encoding(tmp_path):
    source = tmp_path / "external" / "duckdb"
    (source / "foo").mkdir(parents=True)
    (source / "foo" / "nested.txt").write_text("nested\n", encoding="ascii")
    (source / "foo.bar").write_text("prefix ordering\n", encoding="ascii")
    executable = source / "tool.sh"
    executable.write_text("#!/bin/sh\n", encoding="ascii")
    executable.chmod(0o755)
    group_executable = source / "group-executable.sh"
    group_executable.write_text("#!/bin/sh\n", encoding="ascii")
    group_executable.chmod(0o654)
    os.symlink("foo.bar", source / "link")

    _git(tmp_path, "init", "--quiet")
    _git(tmp_path, "config", "core.filemode", "true")
    _git(tmp_path, "add", "external/duckdb")
    repository_tree = _git(tmp_path, "write-tree")
    expected = _git(tmp_path, "rev-parse", f"{repository_tree}:external/duckdb")

    assert filesystem_tree_id(source) == expected
