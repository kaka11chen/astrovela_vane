# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

import pytest

import build_backend


def _stub_backend(monkeypatch, expected_directory: str, expected_settings):
    def build_sdist(directory, settings):
        assert directory == expected_directory
        assert settings == expected_settings
        return "vane_ai-0.1.0a1.tar.gz"

    monkeypatch.setattr(build_backend._backend, "build_sdist", build_sdist)


def test_build_sdist_materializes_source_id_in_git_checkout(tmp_path, monkeypatch):
    (tmp_path / ".git").mkdir()
    manifest = tmp_path / "DUCKDB_SOURCE_ID"
    settings = {"build-dir": "build/test"}
    _stub_backend(monkeypatch, str(tmp_path / "dist"), settings)
    monkeypatch.setattr(build_backend, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(build_backend, "SOURCE_ID_FILE", manifest)

    def synchronize_source_id():
        manifest.write_text("a" * 40 + "\n", encoding="ascii")
        return "a" * 40

    monkeypatch.setattr(build_backend, "synchronize_source_id", synchronize_source_id)

    result = build_backend.build_sdist(str(tmp_path / "dist"), settings)

    assert result == "vane_ai-0.1.0a1.tar.gz"
    assert manifest.read_text(encoding="ascii") == "a" * 40 + "\n"


def test_build_sdist_reuses_manifest_without_git_metadata(tmp_path, monkeypatch):
    manifest = tmp_path / "DUCKDB_SOURCE_ID"
    manifest.write_text("b" * 40 + "\n", encoding="ascii")
    _stub_backend(monkeypatch, str(tmp_path / "dist"), None)
    monkeypatch.setattr(build_backend, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(build_backend, "SOURCE_ID_FILE", manifest)
    monkeypatch.setattr(
        build_backend,
        "synchronize_source_id",
        lambda: pytest.fail("an sdist without Git metadata must reuse its manifest"),
    )

    result = build_backend.build_sdist(str(tmp_path / "dist"))

    assert result == "vane_ai-0.1.0a1.tar.gz"


def test_build_sdist_requires_manifest_without_git_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(build_backend, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(build_backend, "SOURCE_ID_FILE", tmp_path / "DUCKDB_SOURCE_ID")

    with pytest.raises(RuntimeError, match="DUCKDB_SOURCE_ID is required"):
        build_backend.build_sdist(str(tmp_path / "dist"))
