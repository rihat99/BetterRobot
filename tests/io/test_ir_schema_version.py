"""``IRModel.schema_version`` validation in ``build_model``."""

from __future__ import annotations

import pytest
import torch

from better_robot.exceptions import IRSchemaVersionError
from better_robot.io import IR_SCHEMA_VERSION, ModelBuilder, build_model
from better_robot.io.ir import IRModel


def _trivial_ir() -> IRModel:
    b = ModelBuilder("trivial")
    base = b.add_body("base", mass=0.0)
    tip = b.add_body("tip", mass=1.0)
    b.add_revolute_z("j", parent=base, child=tip)
    return b.finalize()


def test_default_schema_version_is_current():
    ir = _trivial_ir()
    assert ir.schema_version == IR_SCHEMA_VERSION


def test_mismatched_schema_version_raises():
    ir = _trivial_ir()
    ir.schema_version = 99
    with pytest.raises(IRSchemaVersionError, match="schema_version"):
        build_model(ir)


def test_matching_schema_version_passes():
    ir = _trivial_ir()
    model = build_model(ir)
    assert model.njoints == 3
