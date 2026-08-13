# SPDX-FileCopyrightText: 2026 Vane contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import uuid

import pytest

import vane
from vane import _native
from vane._ray_cxx import new_distributed_operation_id, require_ray_cxx_attr


def test_require_ray_cxx_attr_returns_registered_binding():
    ray_cxx = getattr(vane, "ray_cxx", None)
    if ray_cxx is None or not hasattr(ray_cxx, "PyLogicalPlan"):
        pytest.skip("vane.ray_cxx.PyLogicalPlan not available in this environment")

    assert require_ray_cxx_attr("PyLogicalPlan") is ray_cxx.PyLogicalPlan


def test_require_ray_cxx_attr_missing_binding_raises_clear_importerror(monkeypatch):
    monkeypatch.setattr(_native, "ray_cxx", object())

    with pytest.raises(ImportError, match=r"vane\.ray_cxx\.MissingBinding"):
        require_ray_cxx_attr("MissingBinding")


def test_distributed_operation_ids_are_uuidv7():
    operation_id = new_distributed_operation_id()

    assert uuid.UUID(operation_id).version == 7
