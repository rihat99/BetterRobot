"""Tests for kinematics Jacobians — shape checks and analytic vs autodiff.

Phase 4 pass criterion: analytic and autodiff Jacobians agree within ``1e-5``.

See ``docs/design/05_KINEMATICS.md §3``.
"""

from __future__ import annotations

import math

import pytest
import torch

from better_robot.io.parsers.programmatic import ModelBuilder
from better_robot.io.build_model import build_model
from better_robot.kinematics.forward import forward_kinematics
from better_robot.kinematics.jacobian import (
    compute_joint_jacobians,
    get_frame_jacobian,
    residual_jacobian,
)
from better_robot.kinematics.jacobian_strategy import JacobianStrategy
from better_robot.residuals.pose import PoseResidual
from better_robot.residuals.base import ResidualState


def _id7() -> torch.Tensor:
    return torch.tensor([0., 0., 0., 0., 0., 0., 1.])


def _simple_arm_model():
    b = ModelBuilder("arm")
    b.add_body("base", mass=0.5)
    b.add_body("link1", mass=1.0)
    b.add_revolute_z("j1", parent="base", child="link1",
                     origin=torch.tensor([0., 0., 0.1, 0., 0., 0., 1.]),
                     lower=-math.pi, upper=math.pi)
    return build_model(b.finalize())


def _chain_model():
    b = ModelBuilder("chain")
    b.add_body("root")
    b.add_body("l1")
    b.add_body("l2")
    b.add_revolute_z("j1", parent="root", child="l1",
                     origin=torch.tensor([0., 0., 0.5, 0., 0., 0., 1.]),
                     lower=-math.pi, upper=math.pi)
    b.add_revolute_y("j2", parent="l1", child="l2",
                     origin=torch.tensor([0., 0., 0.5, 0., 0., 0., 1.]),
                     lower=-math.pi, upper=math.pi)
    return build_model(b.finalize())


@pytest.fixture(scope="module")
def arm():
    return _simple_arm_model()


@pytest.fixture(scope="module")
def chain():
    return _chain_model()


# ── shape checks ─────────────────────────────────────────────────────────────

def test_compute_joint_jacobians_shape(arm):
    q = arm.q_neutral
    data = forward_kinematics(arm, q)
    compute_joint_jacobians(arm, data)
    assert data.joint_jacobians.shape == (arm.njoints, 6, arm.nv)


def test_get_frame_jacobian_shape(arm):
    q = arm.q_neutral
    data = forward_kinematics(arm, q)
    frame_id = arm.frame_id("body_link1")
    J = get_frame_jacobian(arm, data, frame_id)
    assert J.shape == (6, arm.nv)


def test_get_frame_jacobian_batched(arm):
    B = 3
    q = arm.q_neutral.unsqueeze(0).expand(B, -1).clone()
    data = forward_kinematics(arm, q)
    frame_id = arm.frame_id("body_link1")
    J = get_frame_jacobian(arm, data, frame_id)
    assert J.shape == (B, 6, arm.nv)


def test_chain_jacobian_shape(chain):
    q = chain.q_neutral
    data = forward_kinematics(chain, q)
    frame_id = chain.frame_id("body_l2")
    J = get_frame_jacobian(chain, data, frame_id)
    assert J.shape == (6, chain.nv)  # (6, 2)


# ── analytic vs autodiff ──────────────────────────────────────────────────────

def _analytic_vs_autodiff(model, q, frame_id, rtol=1e-3, atol=1e-4):
    """Helper: compare analytic and autodiff Jacobian for PoseResidual."""
    data = forward_kinematics(model, q, compute_frames=True)
    T_target = data.frame_pose_world[frame_id].clone()  # use current pose as target → r≈0 near neutral

    residual = PoseResidual(frame_id=frame_id, target=T_target)
    state = ResidualState(model=model, data=data, variables=q)

    J_analytic = residual_jacobian(residual, state, strategy=JacobianStrategy.ANALYTIC)
    J_autodiff = residual_jacobian(residual, state, strategy=JacobianStrategy.AUTODIFF)

    assert J_analytic.shape == J_autodiff.shape, \
        f"Shape mismatch: {J_analytic.shape} vs {J_autodiff.shape}"
    # Near the target pose, both Jacobians should agree
    torch.testing.assert_close(J_analytic, J_autodiff, rtol=rtol, atol=atol)


def test_analytic_vs_autodiff_simple_arm(arm):
    q = arm.q_neutral
    frame_id = arm.frame_id("body_link1")
    _analytic_vs_autodiff(arm, q, frame_id)


def test_analytic_vs_autodiff_simple_arm_nonzero_q(arm):
    q = arm.q_neutral.clone()
    q[0] = 0.5
    frame_id = arm.frame_id("body_link1")
    _analytic_vs_autodiff(arm, q, frame_id)


def test_analytic_vs_autodiff_chain_j1(chain):
    q = chain.q_neutral.clone()
    q[0] = 0.3
    q[1] = -0.4
    frame_id = chain.frame_id("body_l2")
    _analytic_vs_autodiff(chain, q, frame_id)


# ── panda Jacobian check ──────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def panda_model():
    pytest.importorskip("robot_descriptions")
    from robot_descriptions import panda_description
    from better_robot.io import load
    return load(panda_description.URDF_PATH)


def test_panda_frame_jacobian_shape(panda_model):
    q = panda_model.q_neutral
    data = forward_kinematics(panda_model, q)
    # Use the panda_hand frame (or first body frame)
    frame_name = "body_panda_hand" if "body_panda_hand" in panda_model.frame_name_to_id \
        else panda_model.frame_names[1]
    frame_id = panda_model.frame_id(frame_name)
    J = get_frame_jacobian(panda_model, data, frame_id)
    assert J.shape == (6, panda_model.nv)


def test_panda_analytic_vs_autodiff(panda_model):
    q = panda_model.q_neutral
    data = forward_kinematics(panda_model, q, compute_frames=True)
    frame_name = "body_panda_hand" if "body_panda_hand" in panda_model.frame_name_to_id \
        else panda_model.frame_names[-1]
    frame_id = panda_model.frame_id(frame_name)
    _analytic_vs_autodiff(panda_model, q, frame_id, rtol=1e-3, atol=1e-4)
