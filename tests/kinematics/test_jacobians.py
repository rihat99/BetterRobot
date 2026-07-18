"""Tests for kinematics Jacobians — shape checks and analytic vs finite diff.

Phase 4 pass criterion: analytic and finite-difference Jacobians agree.

See ``docs/concepts/kinematics.md §3``.
"""

from __future__ import annotations

import math

import pytest
import torch

from better_robot.io.parsers.programmatic import ModelBuilder
from better_robot.io.build_model import build_model
from better_robot.kinematics.forward import forward_kinematics, forward_kinematics_raw
from better_robot.kinematics.jacobian import (
    compute_joint_jacobians,
    get_frame_jacobian,
    joint_jacobians_raw,
)
from better_robot.residuals.pose import PoseResidual


class _Context(dict):
    def __init__(self, *args, nv: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.nv = nv

    def free_indices(self, variable_name: str) -> torch.Tensor:
        assert variable_name == "q"
        return torch.arange(self.nv)


def _id7() -> torch.Tensor:
    return torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])


def _simple_arm_model():
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


def _chain_model():
    b = ModelBuilder("chain")
    b.add_body("root")
    b.add_body("l1")
    b.add_body("l2")
    b.add_revolute_z(
        "j1",
        parent="root",
        child="l1",
        origin=torch.tensor([0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0]),
        lower=-math.pi,
        upper=math.pi,
    )
    b.add_revolute_y(
        "j2",
        parent="l1",
        child="l2",
        origin=torch.tensor([0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0]),
        lower=-math.pi,
        upper=math.pi,
    )
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


def test_joint_jacobians_raw_matches_workspace_pass(arm):
    q = arm.q_neutral
    fk_result = forward_kinematics_raw(arm.structure, arm.values, q)
    raw_result = joint_jacobians_raw(
        arm.structure,
        q,
        fk_result.joint_pose_world,
    )
    data = forward_kinematics(arm, q)
    compute_joint_jacobians(arm, data)

    torch.testing.assert_close(raw_result.joint_jacobians, data.joint_jacobians)


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


# ── analytic vs finite diff ───────────────────────────────────────────────────


def _analytic_vs_finite_diff(model, q, frame_id, rtol=1e-3, atol=1e-4):
    """Helper: compare analytic and central-FD Jacobians for PoseResidual."""
    data = forward_kinematics(model, q, compute_frames=True)
    T_target = data.frame_pose_world[frame_id].clone()  # use current pose as target → r≈0 near neutral

    residual = PoseResidual(model, frame_id=frame_id, target=T_target)
    ctx = _Context({"q": q, "data": data}, nv=model.nv)
    J_analytic = residual.jacobian_blocks(ctx)["q"]

    eps = 1e-3 if q.dtype == torch.float32 else 1e-7
    columns = []
    for index in range(model.nv):
        delta = torch.zeros(model.nv, dtype=q.dtype, device=q.device)
        delta[index] = eps
        q_plus = model.integrate(q, delta)
        q_minus = model.integrate(q, -delta)
        plus = residual(
            {
                "q": q_plus,
                "data": forward_kinematics(model, q_plus, compute_frames=True),
            }
        )
        minus = residual(
            {
                "q": q_minus,
                "data": forward_kinematics(model, q_minus, compute_frames=True),
            }
        )
        columns.append((plus - minus) / (2.0 * eps))
    J_finite_diff = torch.stack(columns, dim=-1)

    assert J_analytic.shape == J_finite_diff.shape, f"Shape mismatch: {J_analytic.shape} vs {J_finite_diff.shape}"
    # Near the target pose, both Jacobians should agree
    torch.testing.assert_close(J_analytic, J_finite_diff, rtol=rtol, atol=atol)


def test_analytic_vs_finite_diff_simple_arm(arm):
    q = arm.q_neutral
    frame_id = arm.frame_id("body_link1")
    _analytic_vs_finite_diff(arm, q, frame_id)


def test_analytic_vs_finite_diff_simple_arm_nonzero_q(arm):
    q = arm.q_neutral.clone()
    q[0] = 0.5
    frame_id = arm.frame_id("body_link1")
    _analytic_vs_finite_diff(arm, q, frame_id)


def test_analytic_vs_finite_diff_chain_j1(chain):
    q = chain.q_neutral.clone()
    q[0] = 0.3
    q[1] = -0.4
    frame_id = chain.frame_id("body_l2")
    _analytic_vs_finite_diff(chain, q, frame_id)


def test_pose_residual_resolves_frame_name(arm):
    q = arm.q_neutral
    data = forward_kinematics(arm, q, compute_frames=True)
    frame_name = "body_link1"
    target = data.frame_pose_world[arm.frame_id(frame_name)].clone()
    residual = PoseResidual(arm, frame=frame_name, target=target)

    assert residual.frame_id == arm.frame_id(frame_name)
    torch.testing.assert_close(residual({"q": q, "data": data}), torch.zeros(6))


# ── panda Jacobian check ──────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def panda_model():
    pytest.importorskip("robot_descriptions")
    from robot_descriptions import panda_description  # noqa: PLC0415
    from better_robot.io import load  # noqa: PLC0415

    return load(panda_description.URDF_PATH)


def test_panda_frame_jacobian_shape(panda_model):
    q = panda_model.q_neutral
    data = forward_kinematics(panda_model, q)
    # Use the panda_hand frame (or first body frame)
    frame_name = "body_panda_hand" if "body_panda_hand" in panda_model.frame_name_to_id else panda_model.frame_names[1]
    frame_id = panda_model.frame_id(frame_name)
    J = get_frame_jacobian(panda_model, data, frame_id)
    assert J.shape == (6, panda_model.nv)


def test_panda_analytic_vs_finite_diff(panda_model):
    q = panda_model.q_neutral
    frame_name = "body_panda_hand" if "body_panda_hand" in panda_model.frame_name_to_id else panda_model.frame_names[-1]
    frame_id = panda_model.frame_id(frame_name)
    _analytic_vs_finite_diff(panda_model, q, frame_id, rtol=1e-3, atol=1e-4)
