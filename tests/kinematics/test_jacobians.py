"""Tests for kinematics Jacobians — shape checks and analytic vs finite diff.

Phase 4 pass criterion: analytic and finite-difference Jacobians agree.

See ``docs/concepts/kinematics_and_jacobians.md``.
"""

from __future__ import annotations

import math

import pytest
import torch

from better_robot.io.parsers.programmatic import ModelBuilder
from better_robot.io.build_model import build_model
from better_robot.kinematics.forward import forward_kinematics_raw
from better_robot.kinematics.jacobian import (
    frame_jacobian_raw,
    joint_jacobians_raw,
)
from better_robot.kinematics.state import forward_kinematics_state
from better_robot.terms import frame_pose
from better_robot.kinematics import frame_jacobian, joint_jacobian


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
    state = forward_kinematics_state(arm, q)
    assert torch.stack([joint_jacobian(arm, state, j, reference="world") for j in range(arm.njoints)], dim=-3).shape == (arm.njoints, 6, arm.nv)


def test_joint_jacobians_raw_matches_workspace_pass(arm):
    q = arm.q_neutral
    fk_result = forward_kinematics_raw(arm.structure, arm.values, q)
    raw_result = joint_jacobians_raw(
        arm.structure,
        q,
        fk_result.joint_pose_world,
    )
    state = forward_kinematics_state(arm, q)
    torch.testing.assert_close(raw_result.joint_jacobians, torch.stack([joint_jacobian(arm, state, j, reference="world") for j in range(arm.njoints)], dim=-3))


def test_joint_jacobian_plan_rebuilds_after_to(chain):
    """The memoised batched-Jacobian plan follows ``ModelStructure.to``.

    ``joint_jacobians_raw`` caches a static plan on the structure instance;
    :meth:`ModelStructure.to` mints a fresh instance, so a moved model must not
    reuse a stale plan. Mirrors the ``_fk_matrix_plan`` memoisation contract.
    """
    from better_robot.kinematics import _jacobian_columns  # noqa: PLC0415

    q = chain.q_neutral
    poses = forward_kinematics_state(chain, q).joint_pose
    reference = joint_jacobians_raw(chain.structure, q, poses).joint_jacobians
    assert getattr(chain.structure, "_joint_jacobian_plan", None) is not None

    moved = chain.to()
    assert getattr(moved.structure, "_joint_jacobian_plan", None) is None

    moved_poses = forward_kinematics_state(moved, moved.q_neutral).joint_pose
    moved_jacobians = joint_jacobians_raw(moved.structure, moved.q_neutral, moved_poses).joint_jacobians
    plan = _jacobian_columns.get_plan(moved.structure)
    device = moved.structure.idx_vs_full_tensor.device
    assert plan.column_subspace.device == device
    assert plan.support_mask.device == device
    torch.testing.assert_close(moved_jacobians, reference)


@pytest.mark.parametrize("reference", ("world", "local_world_aligned", "local"))
def test_frame_jacobian_raw_matches_workspace_pass(arm, reference):
    q = arm.q_neutral
    frame_id = arm.frame_id("body_link1")
    fk_result = forward_kinematics_raw(arm.structure, arm.values, q)
    actual = frame_jacobian_raw(
        arm.structure,
        arm.values,
        q,
        fk_result.joint_pose_world,
        frame_id,
        reference=reference,
    )
    state = forward_kinematics_state(arm, q)
    expected = frame_jacobian(arm, state, frame_id, reference=reference)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("reference", ("world", "local_world_aligned", "local"))
def test_get_joint_jacobian_matches_coincident_frame(arm, chain, reference):
    """``get_joint_jacobian`` equals the Jacobian of the joint's body frame.

    Builder body frames sit at their parent joint (identity local placement),
    so the joint path and the (separately pinocchio-anchored) frame path must
    agree in every reference frame. This is the only kinematics test that
    exercises ``get_joint_jacobian`` with a ``reference=`` argument.
    """
    for model, frame_name in ((arm, "body_link1"), (chain, "body_l2")):
        q = model.q_neutral.clone()
        for index in range(model.nv):
            q[index] = 0.3 * (index + 1)
        state = forward_kinematics_state(model, q)
        frame_id = model.frame_id(frame_name)
        joint_id = model.structure.frame_parent_joints[frame_id].item()
        J_joint = joint_jacobian(model, state, joint_id, reference=reference)
        J_frame = frame_jacobian(model, state, frame_id, reference=reference)
        torch.testing.assert_close(J_joint, J_frame)


def test_frame_jacobian_raw_compiles_fullgraph(arm):
    q = arm.q_neutral.expand(2, -1).clone()
    poses = forward_kinematics_raw(arm.structure, arm.values, q).joint_pose_world
    frame_id = arm.frame_id("body_link1")

    def evaluate(q_value, joint_poses):
        return frame_jacobian_raw(
            arm.structure,
            arm.values,
            q_value,
            joint_poses,
            frame_id,
        )

    compiled = torch.compile(evaluate, fullgraph=True, backend="eager")
    torch.testing.assert_close(compiled(q, poses), evaluate(q, poses))


def test_get_frame_jacobian_shape(arm):
    q = arm.q_neutral
    state = forward_kinematics_state(arm, q)
    frame_id = arm.frame_id("body_link1")
    J = frame_jacobian(arm, state, frame_id)
    assert J.shape == (6, arm.nv)


def test_get_frame_jacobian_batched(arm):
    B = 3
    q = arm.q_neutral.unsqueeze(0).expand(B, -1).clone()
    state = forward_kinematics_state(arm, q)
    frame_id = arm.frame_id("body_link1")
    J = frame_jacobian(arm, state, frame_id)
    assert J.shape == (B, 6, arm.nv)


def test_chain_jacobian_shape(chain):
    q = chain.q_neutral
    state = forward_kinematics_state(chain, q)
    frame_id = chain.frame_id("body_l2")
    J = frame_jacobian(chain, state, frame_id)
    assert J.shape == (6, chain.nv)  # (6, 2)


# ── analytic vs finite diff ───────────────────────────────────────────────────


def _analytic_vs_finite_diff(model, q, frame_id, rtol=1e-3, atol=1e-4):
    """Helper: the analytic ``jac`` of ``terms.frame_pose`` against a central-FD Jacobian of its residual."""
    target = forward_kinematics_state(model, q).frame_pose_of(frame_id).clone()  # r ≈ 0 near neutral
    term = frame_pose(model, frame_id, target="target")
    J_analytic = term.jac(forward_kinematics_state(model, q), target)["q"]

    eps = 1e-3 if q.dtype == torch.float32 else 1e-7
    columns = []
    for index in range(model.nv):
        delta = torch.zeros(model.nv, dtype=q.dtype, device=q.device)
        delta[index] = eps
        plus = term.fn(forward_kinematics_state(model, model.integrate(q, delta)), target)
        minus = term.fn(forward_kinematics_state(model, model.integrate(q, -delta)), target)
        columns.append((plus - minus) / (2.0 * eps))
    J_finite_diff = torch.stack(columns, dim=-1)

    assert J_analytic.shape == J_finite_diff.shape, f"Shape mismatch: {J_analytic.shape} vs {J_finite_diff.shape}"
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


# ── panda Jacobian check ──────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def panda_model():
    pytest.importorskip("robot_descriptions")
    from robot_descriptions import panda_description  # noqa: PLC0415
    from better_robot.io import load  # noqa: PLC0415

    return load(panda_description.URDF_PATH)


def test_panda_frame_jacobian_shape(panda_model):
    q = panda_model.q_neutral
    state = forward_kinematics_state(panda_model, q)
    # Use the panda_hand frame (or first body frame)
    frame_name = "body_panda_hand" if "body_panda_hand" in panda_model.frame_name_to_id else panda_model.frame_names[1]
    frame_id = panda_model.frame_id(frame_name)
    J = frame_jacobian(panda_model, state, frame_id)
    assert J.shape == (6, panda_model.nv)


def test_panda_analytic_vs_finite_diff(panda_model):
    q = panda_model.q_neutral
    frame_name = "body_panda_hand" if "body_panda_hand" in panda_model.frame_name_to_id else panda_model.frame_names[-1]
    frame_id = panda_model.frame_id(frame_name)
    _analytic_vs_finite_diff(panda_model, q, frame_id, rtol=1e-3, atol=1e-4)
