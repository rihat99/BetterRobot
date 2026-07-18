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
from better_robot.kinematics.forward import (
    frame_placements_raw,
    forward_kinematics,
    forward_kinematics_raw,
)


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


def test_fk_raw_shapes(arm):
    q = arm.q_neutral
    result = forward_kinematics_raw(arm.structure, arm.values, q)
    assert result.joint_pose_world.shape == (arm.njoints, 7)
    assert result.joint_pose_local.shape == (arm.njoints, 7)


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_fk_rejects_unsupported_low_precision(arm, dtype):
    low_precision_model = arm.to(dtype=dtype)
    with pytest.raises(DtypeMismatchError, match="unsupported"):
        forward_kinematics(low_precision_model, low_precision_model.q_neutral)


def test_fk_rejects_model_query_dtype_mismatch(arm):
    with pytest.raises(DtypeMismatchError, match="model.dtype"):
        forward_kinematics(arm, arm.q_neutral.to(torch.float64))


def test_fk_data_shapes(arm):
    q = arm.q_neutral
    data = forward_kinematics(arm, q)
    assert data.joint_pose_world.shape == (arm.njoints, 7)
    assert data.joint_pose_local.shape == (arm.njoints, 7)


def test_fk_batched_shapes(arm):
    B = 4
    q = arm.q_neutral.unsqueeze(0).expand(B, -1).clone()
    result = forward_kinematics_raw(arm.structure, arm.values, q)
    assert result.joint_pose_world.shape == (B, arm.njoints, 7)
    assert result.joint_pose_local.shape == (B, arm.njoints, 7)


# ── correctness at neutral ────────────────────────────────────────────────────


def test_universe_joint_is_identity(arm):
    q = arm.q_neutral
    result = forward_kinematics_raw(arm.structure, arm.values, q)
    T0 = result.joint_pose_world[0]
    assert T0[:3].norm() < 1e-6, "universe translation should be zero"
    assert abs(float(T0[6]) - 1.0) < 1e-6, "universe quaternion w should be 1"


def test_joint1_at_neutral(arm):
    """Root joint (fixed) at neutral has identity pose."""
    q = arm.q_neutral
    result = forward_kinematics_raw(arm.structure, arm.values, q)
    T1 = result.joint_pose_world[1]
    # root_joint is fixed, joint_placements[1] is identity
    assert T1[:3].norm() < 1e-5


def test_joint2_translation_at_neutral(arm):
    """Joint 2 (j1 revolute) should be at z=0.1 at neutral (q=0)."""
    q = arm.q_neutral
    result = forward_kinematics_raw(arm.structure, arm.values, q)
    T2 = result.joint_pose_world[2]
    assert abs(float(T2[2]) - 0.1) < 1e-5, f"Expected z=0.1, got {float(T2[2])}"


def test_revolute_z_rotation(arm):
    """Rotating joint 2 by π/2 around Z should change X→Y direction."""
    q = arm.q_neutral.clone()
    q[0] = math.pi / 2  # rotate j1 by 90 degrees
    result = forward_kinematics_raw(arm.structure, arm.values, q)
    T2 = result.joint_pose_world[2]
    # Rotation part: qz should be ~sin(π/4), qw should be ~cos(π/4)
    qz = float(T2[5])
    qw = float(T2[6])
    assert abs(qz - math.sin(math.pi / 4)) < 1e-5, f"Expected qz≈{math.sin(math.pi / 4)}, got {qz}"
    assert abs(qw - math.cos(math.pi / 4)) < 1e-5


# ── frame placements ─────────────────────────────────────────────────────────


def test_update_frame_placements_shape(arm):
    q = arm.q_neutral
    data = forward_kinematics(arm, q, compute_frames=True)
    assert data.frame_pose_world is not None
    assert data.frame_pose_world.shape == (arm.nframes, 7)


def test_frame_placements_raw_matches_public_fk(arm):
    fk_result = forward_kinematics_raw(arm.structure, arm.values, arm.q_neutral)
    frame_result = frame_placements_raw(
        arm.structure,
        arm.values,
        fk_result.joint_pose_world,
    )
    data = forward_kinematics(arm, arm.q_neutral, compute_frames=True)

    torch.testing.assert_close(frame_result.frame_pose_world, data.frame_pose_world)


def test_body_frame_is_at_joint(arm):
    """body_universe frame should be at identity."""
    q = arm.q_neutral
    data = forward_kinematics(arm, q, compute_frames=True)
    f_id = arm.frame_id("body_universe")
    T_f = data.frame_pose_world[f_id]
    assert T_f[:3].norm() < 1e-6


# ── panda smoke test ─────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def panda_model():
    pytest.importorskip("robot_descriptions")
    from robot_descriptions import panda_description  # noqa: PLC0415
    from better_robot.io import load  # noqa: PLC0415

    return load(panda_description.URDF_PATH)


def test_panda_fk_neutral_shape(panda_model):
    q = panda_model.q_neutral
    data = forward_kinematics(panda_model, q, compute_frames=True)
    assert data.joint_pose_world.shape == (panda_model.njoints, 7)
    assert data.frame_pose_world.shape == (panda_model.nframes, 7)


def test_panda_fk_universe_is_identity(panda_model):
    q = panda_model.q_neutral
    result = forward_kinematics_raw(panda_model.structure, panda_model.values, q)
    T0 = result.joint_pose_world[0]
    assert T0[:3].norm() < 1e-6
    assert abs(float(T0[6]) - 1.0) < 1e-6


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_panda_fk_dtype_preserved(panda_model, dtype):
    model = panda_model.to(dtype=dtype)
    result = forward_kinematics_raw(
        model.structure,
        model.values,
        model.q_neutral,
    )
    assert result.joint_pose_world.dtype == dtype
