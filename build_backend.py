# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""PEP 517 backend that adds DuckDB's generated identity to source archives."""

from __future__ import annotations

from scikit_build_core import build as _backend

from scripts.sync_duckdb_source_id import REPOSITORY_ROOT, SOURCE_ID_FILE, synchronize_source_id

build_editable = _backend.build_editable
build_wheel = _backend.build_wheel
get_requires_for_build_editable = _backend.get_requires_for_build_editable
get_requires_for_build_sdist = _backend.get_requires_for_build_sdist
get_requires_for_build_wheel = _backend.get_requires_for_build_wheel
prepare_metadata_for_build_editable = _backend.prepare_metadata_for_build_editable
prepare_metadata_for_build_wheel = _backend.prepare_metadata_for_build_wheel


def build_sdist(
    sdist_directory: str,
    config_settings: dict[str, list[str] | str] | None = None,
) -> str:
    """Build an sdist containing the exact DuckDB tree ID for this checkout."""
    if (REPOSITORY_ROOT / ".git").exists():
        synchronize_source_id()
    elif not SOURCE_ID_FILE.is_file():
        raise RuntimeError("DUCKDB_SOURCE_ID is required when building an sdist without Git metadata")

    return _backend.build_sdist(sdist_directory, config_settings)
