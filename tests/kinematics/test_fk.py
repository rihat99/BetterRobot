"""Tests for forward kinematics — correctness and shape checks.

See ``docs/concepts/kinematics_and_jacobians.md``.
"""

from __future__ import annotations

import math

import pytest
import torch

from better_robot.exceptions import DtypeMismatchError
from better_robot.io.build_model import build_model
from better_robot.io.parsers.programmatic import ModelBuilder
from better_robot.kinematics import forward_kinematics_state


def _id7() -> torch.Tensor:
    return torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])


def _simple_arm():
    b = ModelBuilder("arm")
    b.add_body("base", mass=0.5)
    b.add_body("link1", mass=1.0)
    b.add_revolute_z(
        "j1",
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
    low_precision_model = arm.to(dtype=dtype)
    with pytest.raises(DtypeMismatchError, match="unsupported"):
        forward_kinematics_state(low_precision_model, low_precision_model.q_neutral)


def test_fk_rejects_model_query_dtype_mismatch(arm):
    with pytest.raises(DtypeMismatchError, match="model.dtype"):
        forward_kinematics_state(arm, arm.q_neutral.to(torch.float64))


# ── correctness at neutral ────────────────────────────────────────────────────


def test_universe_joint_is_identity(arm):
    q = arm.q_neutral
    result = forward_kinematics_state(arm, q)
    T0 = result.joint_pose[0]
    assert T0[:3].norm() < 1e-6, "universe translation should be zero"
    assert abs(float(T0[6]) - 1.0) < 1e-6, "universe quaternion w should be 1"


def test_joint1_at_neutral(arm):
    """Root joint (fixed) at neutral has identity pose."""
    q = arm.q_neutral
    result = forward_kinematics_state(arm, q)
    T1 = result.joint_pose[1]
    # root_joint is fixed, joint_placements[1] is identity
    assert T1[:3].norm() < 1e-5


def test_joint2_translation_at_neutral(arm):
    """Joint 2 (j1 revolute) should be at z=0.1 at neutral (q=0)."""
    q = arm.q_neutral
    result = forward_kinematics_state(arm, q)
    T2 = result.joint_pose[2]
    assert abs(float(T2[2]) - 0.1) < 1e-5, f"Expected z=0.1, got {float(T2[2])}"


def test_revolute_z_rotation(arm):
    """Rotating joint 2 by π/2 around Z should change X→Y direction."""
    q = arm.q_neutral.clone()
    q[0] = math.pi / 2  # rotate j1 by 90 degrees
    result = forward_kinematics_state(arm, q)
    T2 = result.joint_pose[2]
    # Rotation part: qz should be ~sin(π/4), qw should be ~cos(π/4)
    qz = float(T2[5])
    qw = float(T2[6])
    assert abs(qz - math.sin(math.pi / 4)) < 1e-5, f"Expected qz≈{math.sin(math.pi / 4)}, got {qz}"
    assert abs(qw - math.cos(math.pi / 4)) < 1e-5


# ── frame placements ─────────────────────────────────────────────────────────


def test_body_frame_is_at_joint(arm):
    """body_universe frame should be at identity."""
    q = arm.q_neutral
    state = forward_kinematics_state(arm, q)
    f_id = arm.frame_id("body_universe")
    T_f = state.frame_pose[f_id]
    assert T_f[:3].norm() < 1e-6


# ── panda smoke test ─────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def panda_model():
    pytest.importorskip("robot_descriptions")
    from robot_descriptions import panda_description  # noqa: PLC0415
    from better_robot.io import load  # noqa: PLC0415

    return load(panda_description.URDF_PATH)
