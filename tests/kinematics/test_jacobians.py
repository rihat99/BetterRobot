"""Kinematics Jacobians: a joint's equals its coincident frame's, and the point Jacobian equals finite differences of
forward kinematics."""

from __future__ import annotations

import math

import pytest
import torch

from better_robot.exceptions import DtypeMismatchError
from better_robot.io.builder import ModelBuilder
from better_robot.io import build_model
from better_robot.model import JointFreeFlyer, JointRevolute
from better_robot.kinematics import forward_kinematics
from better_robot.kinematics import frame_jacobian, integrate, joint_jacobian, point_jacobian
from tests.support.robots import g1_urdf, one_revolute_arm_ir


def _chain_model():
    b = ModelBuilder("chain")
    b.add_body("root")
    b.add_body("l1")
    b.add_body("l2")
    b.add_joint(
        "j1",
        kind=JointRevolute(axis=(0.0, 0.0, 1.0)),
        parent="root",
        child="l1",
        origin=torch.tensor([0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0]),
        lower=-math.pi,
        upper=math.pi,
    )
    b.add_joint(
        "j2",
        kind=JointRevolute(axis=(0.0, 1.0, 0.0)),
        parent="l1",
        child="l2",
        origin=torch.tensor([0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0]),
        lower=-math.pi,
        upper=math.pi,
    )
    return build_model(b.finalize())


@pytest.fixture(scope="module")
def arm():
    return build_model(one_revolute_arm_ir())


@pytest.fixture(scope="module")
def chain():
    return _chain_model()


@pytest.mark.parametrize("reference", ("world", "local_world_aligned", "local"))
def test_joint_jacobian_matches_coincident_frame(arm, chain, reference):
    """``joint_jacobian`` equals the Jacobian of the joint's body frame.

    Builder body frames sit at their parent joint (identity local placement),
    so the joint path and the (separately pinocchio-anchored) frame path must
    agree in every reference frame.
    """
    for model, frame_name in ((arm, "link1"), (chain, "l2")):
        q = model.q_neutral.clone()
        for index in range(model.nv):
            q[index] = 0.3 * (index + 1)
        state = forward_kinematics(model, q)
        frame_id = model.frame_id(frame_name)
        joint_id = model.frame_parent_joint_ids[frame_id]
        J_joint = joint_jacobian(model, state, joint=joint_id, reference=reference)
        J_frame = frame_jacobian(model, state, frame=frame_id, reference=reference)
        torch.testing.assert_close(J_joint, J_frame)


def test_point_jacobian_matches_finite_differences_on_a_floating_base():
    from better_robot.io import load  # noqa: PLC0415

    model = load(g1_urdf(), root_joint=JointFreeFlyer(), dtype=torch.float64)
    generator = torch.Generator().manual_seed(0)
    q = integrate(
        model, model.q_neutral.expand(3, -1), 0.3 * torch.randn(3, model.nv, generator=generator, dtype=torch.float64)
    )
    frames = ["left_ankle_roll_link", "right_wrist_yaw_link", "left_ankle_roll_link", "pelvis"]
    points = 0.1 * torch.randn(len(frames), 3, generator=generator, dtype=torch.float64)
    ids = [model.frame_id(frame) for frame in frames]

    def positions(configuration):
        transform = forward_kinematics(model, configuration).frame_transforms[..., ids, :, :]
        return (transform[..., :3] @ points.unsqueeze(-1)).squeeze(-1) + transform[..., 3]

    eps = 1e-6
    numeric = torch.stack(
        [
            (positions(integrate(model, q, eps * e)) - positions(integrate(model, q, -eps * e))) / (2 * eps)
            for e in torch.eye(model.nv, dtype=torch.float64)
        ],
        dim=-1,
    )
    state = forward_kinematics(model, q)
    jacobian = point_jacobian(model, state, frames=frames, points=points)
    assert jacobian.shape == (3, len(frames), 3, model.nv)
    torch.testing.assert_close(jacobian, numeric, rtol=0, atol=1e-8)
    with pytest.raises(DtypeMismatchError):
        point_jacobian(model, state, frames=frames, points=points.float())
    with pytest.raises(ValueError, match="empty"):
        point_jacobian(model, state, frames=[], points=points[:0])
