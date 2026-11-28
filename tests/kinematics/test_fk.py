"""Tests for forward kinematics — the dtypes and state shapes it refuses (its values:
``tests/kinematics/test_fk_contracts.py`` and Pinocchio parity).
"""

from __future__ import annotations

import pytest
import torch

from better_robot.exceptions import DtypeMismatchError, ShapeError
from better_robot.io import build_model, load
from better_robot.kinematics import KinematicState, forward_kinematics
from tests.support.robots import one_revolute_arm_ir, panda_urdf


@pytest.fixture(scope="module")
def arm():
    return build_model(one_revolute_arm_ir())


# ── shape checks ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_fk_rejects_unsupported_low_precision(arm, dtype):
    with pytest.raises(DtypeMismatchError, match="is not float32 or float64"):
        forward_kinematics(arm, arm.q_neutral.to(dtype))


def test_fk_rejects_model_query_dtype_mismatch(arm):
    with pytest.raises(DtypeMismatchError, match="model.dtype"):
        forward_kinematics(arm, arm.q_neutral.to(torch.float64))


def test_a_hand_built_state_with_the_wrong_shapes_fails_at_construction() -> None:
    """A hand-built state whose tables do not fit the model fails here, not later in frame_jacobian with a raw
    IndexError."""
    model = load(panda_urdf())
    state = forward_kinematics(model, model.q_neutral.expand(2, -1))
    tables = {"q": state.q, "joint_transforms": state.joint_transforms}
    bad_tables = [
        ("q", state.q[:, :3], r"q must have shape \(B\.\.\., \d+\) with the batch \(2,\) of q"),
        ("joint_transforms", state.joint_transforms[:, :2], r"joint_transforms must have shape \(B\.\.\., \d+, 3, 4\)"),
        ("joint_transforms", state.joint_transforms[:1], r"joint_transforms .* with the batch \(2,\) of q, got \(1,"),
    ]
    for name, table, match in bad_tables:
        with pytest.raises(ShapeError, match=match):
            KinematicState(model=model, **{**tables, name: table})
