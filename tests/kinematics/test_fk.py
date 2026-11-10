"""Tests for forward kinematics — correctness and shape checks.

See ``docs/concepts/kinematics-and-jacobians.md``.
"""

from __future__ import annotations

import math

import pytest
import torch

from better_robot.exceptions import DtypeMismatchError, ShapeError
from better_robot.io import build_model, load
from better_robot.io.builder import ModelBuilder
from better_robot.kinematics import KinematicState, forward_kinematics
from tests.support.robots import panda_urdf


def _id7() -> torch.Tensor:
    return torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])


def _simple_arm():
    b = ModelBuilder("arm")
    b.add_body("base", mass=0.5)
    b.add_body("link1", mass=1.0)
    b.add_revolute(
        "j1",
        axis=(0.0, 0.0, 1.0),
        parent="base",
        child="link1",
        origin=torch.tensor([0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0]),
        lower=-math.pi,
        upper=math.pi,
    )
    return build_model(b.finalize())


@pytest.fixture(scope="module")
def arm():
    return _simple_arm()


# ── shape checks ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_fk_rejects_unsupported_low_precision(arm, dtype):
    with pytest.raises(DtypeMismatchError, match="is not float32 or float64"):
        forward_kinematics(arm, arm.q_neutral.to(dtype))


def test_fk_rejects_model_query_dtype_mismatch(arm):
    with pytest.raises(DtypeMismatchError, match="model.dtype"):
        forward_kinematics(arm, arm.q_neutral.to(torch.float64))


# ── correctness at neutral ────────────────────────────────────────────────────


def test_universe_joint_is_identity(arm):
    q = arm.q_neutral
    result = forward_kinematics(arm, q)
    T0 = result.joint_poses[0]
    assert T0[:3].norm() < 1e-6, "universe translation should be zero"
    assert abs(float(T0[6]) - 1.0) < 1e-6, "universe quaternion w should be 1"


def test_joint1_at_neutral(arm):
    """Root joint (fixed) at neutral has identity pose."""
    q = arm.q_neutral
    result = forward_kinematics(arm, q)
    T1 = result.joint_poses[1]
    # root_joint is fixed, joint_placements[1] is identity
    assert T1[:3].norm() < 1e-5


def test_joint2_translation_at_neutral(arm):
    """Joint 2 (j1 revolute) should be at z=0.1 at neutral (q=0)."""
    q = arm.q_neutral
    result = forward_kinematics(arm, q)
    T2 = result.joint_poses[2]
    assert abs(float(T2[2]) - 0.1) < 1e-5, f"Expected z=0.1, got {float(T2[2])}"


def test_revolute_z_rotation(arm):
    """Rotating joint 2 by π/2 around Z should change X→Y direction."""
    q = arm.q_neutral.clone()
    q[0] = math.pi / 2  # rotate j1 by 90 degrees
    result = forward_kinematics(arm, q)
    T2 = result.joint_poses[2]
    # Rotation part: qz should be ~sin(π/4), qw should be ~cos(π/4)
    qz = float(T2[5])
    qw = float(T2[6])
    assert abs(qz - math.sin(math.pi / 4)) < 1e-5, f"Expected qz≈{math.sin(math.pi / 4)}, got {qz}"
    assert abs(qw - math.cos(math.pi / 4)) < 1e-5


# ── frame placements ─────────────────────────────────────────────────────────


def test_body_frame_is_at_joint(arm):
    """universe frame should be at identity."""
    q = arm.q_neutral
    state = forward_kinematics(arm, q)
    f_id = arm.frame_id("universe")
    T_f = state.frame_poses[f_id]
    assert T_f[:3].norm() < 1e-6


# ── panda smoke test ─────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def panda_model():
    from better_robot.io import load  # noqa: PLC0415

    return load(panda_urdf())


def test_a_hand_built_state_with_the_wrong_shapes_fails_at_construction() -> None:
    """The review built a state whose tables did not fit the model; frame_jacobian then raised a raw IndexError."""
    model = load(panda_urdf())
    state = forward_kinematics(model, model.q_neutral.expand(2, -1))
    tables = {"q": state.q, "joint_transforms": state.joint_transforms, "frame_transforms": state.frame_transforms}
    bad_tables = [
        ("q", state.q[:, :3], r"q must have shape \(B\.\.\., \d+\) with the batch \(2,\) of q"),
        ("joint_transforms", state.joint_transforms[:, :2], r"joint_transforms must have shape \(B\.\.\., \d+, 3, 4\)"),
        ("frame_transforms", state.frame_transforms[:1], r"frame_transforms .* with the batch \(2,\) of q, got \(1,"),
    ]
    for name, table, match in bad_tables:
        with pytest.raises(ShapeError, match=match):
            KinematicState(model=model, **{**tables, name: table})
