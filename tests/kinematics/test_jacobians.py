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


def test_joint_jacobian_plan_rebuilds_after_to(chain):
    """The memoised batched-Jacobian plan follows ``ModelStructure.to``.

    The column pass caches a static plan on the structure instance; :meth:`ModelStructure.to`
    mints a fresh instance, so a moved model must not reuse a stale plan.
    """
    from better_robot.kinematics import _jacobian_columns  # noqa: PLC0415

    def world_jacobians(model):
        state = forward_kinematics_state(model, model.q_neutral)
        return torch.stack([joint_jacobian(model, state, joint=j, reference="world") for j in range(model.njoints)], dim=-3)

    reference = world_jacobians(chain)
    assert getattr(chain.structure, "_joint_jacobian_plan", None) is not None

    moved = chain.to()
    assert getattr(moved.structure, "_joint_jacobian_plan", None) is None

    moved_jacobians = world_jacobians(moved)
    plan = _jacobian_columns.get_plan(moved.structure)
    device = moved.structure.idx_vs_full_tensor.device
    assert plan.column_subspace.device == device
    assert plan.support_mask.device == device
    torch.testing.assert_close(moved_jacobians, reference)


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
        J_joint = joint_jacobian(model, state, joint=joint_id, reference=reference)
        J_frame = frame_jacobian(model, state, frame=frame_id, reference=reference)
        torch.testing.assert_close(J_joint, J_frame)


def test_frame_jacobian_compiles_fullgraph(arm):
    q = arm.q_neutral.expand(2, -1).clone()
    state = forward_kinematics_state(arm, q)
    frame_id = arm.frame_id("body_link1")

    def evaluate(kinematic_state):
        return frame_jacobian(arm, kinematic_state, frame=frame_id)

    compiled = torch.compile(evaluate, fullgraph=True, backend="eager")
    torch.testing.assert_close(compiled(state), evaluate(state))


# ── analytic vs finite diff ───────────────────────────────────────────────────


def _analytic_vs_finite_diff(model, q, frame_id, rtol=1e-3, atol=1e-4):
    """Helper: the analytic ``jac`` of ``terms.frame_pose`` against a central-FD Jacobian of its residual."""
    target = forward_kinematics_state(model, q).frame_pose_of(frame_id).clone()  # r ≈ 0 near neutral
    term = frame_pose(model, frame=frame_id, target="target")
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


@pytest.mark.parametrize(
    "q_overrides",
    [
        pytest.param({}, id="neutral_q"),
        pytest.param({0: 0.5}, id="nonzero_q"),
    ],
)
def test_analytic_vs_finite_diff_simple_arm(arm, q_overrides):
    q = arm.q_neutral.clone()
    for index, value in q_overrides.items():
        q[index] = value
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
