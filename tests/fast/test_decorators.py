# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

"""Public decorator API tests."""

from __future__ import annotations


def test_duckdb_func_remains_available():
    import duckdb.func as duckdb_func

    for name in ("NATIVE", "ARROW", "DEFAULT", "SPECIAL", "FunctionNullHandling", "PythonUDFType"):
        assert getattr(duckdb_func, name) is not None


def test_relation_udf_methods_remain_available():
    import duckdb

    con = duckdb.connect()
    rel = con.sql("SELECT 1 AS x")

    assert callable(rel.map)
    assert callable(rel.map_batches)
