"""IK regression tests — ``solve_ik`` on Panda reaches target pose.

Phase 4 pass criterion: ``solve_ik(panda, …)`` reaches the same target
pose as the legacy code on the IK benchmark set.

See ``docs/design/08_TASKS.md §1``.
"""

from __future__ import annotations

import pytest
import torch

from better_robot.io import load
from better_robot.kinematics.forward import forward_kinematics
from better_robot.tasks.ik import IKCostConfig, OptimizerConfig, solve_ik


@pytest.fixture(scope="module")
def panda():
    pytest.importorskip("robot_descriptions")
    from robot_descriptions import panda_description
    return load(panda_description.URDF_PATH)


def _ee_frame(model) -> str:
    """Return a sensible EE frame name for the panda."""
    for candidate in ("body_panda_hand", "body_panda_link8", "body_panda_link7"):
        if candidate in model.frame_name_to_id:
            return candidate
    return model.frame_names[-1]


def _feasible_q(model) -> torch.Tensor:
    """Return q_neutral clamped to joint limits (some robots have limits that exclude 0)."""
    lo = model.lower_pos_limit
    hi = model.upper_pos_limit
    return model.q_neutral.clone().clamp(lo, hi)


# ── basic solve_ik tests ──────────────────────────────────────────────────────

def test_solve_ik_returns_ik_result(panda):
    q = panda.q_neutral
    data = forward_kinematics(panda, q, compute_frames=True)
    frame_name = _ee_frame(panda)
    T_target = data.frame_pose_world[panda.frame_id(frame_name)].clone()

    result = solve_ik(panda, {frame_name: T_target}, initial_q=q)
    assert result.q is not None
    assert result.q.shape == (panda.nq,)


def test_solve_ik_q_shape(panda):
    q = panda.q_neutral
    frame_name = _ee_frame(panda)
    data = forward_kinematics(panda, q, compute_frames=True)
    T_target = data.frame_pose_world[panda.frame_id(frame_name)].clone()
    result = solve_ik(panda, {frame_name: T_target})
    assert result.q.shape == (panda.nq,)


def test_solve_ik_at_neutral_converges(panda):
    """solve_ik(panda, target=FK(q_neutral)) should keep q ≈ q_neutral."""
    q = panda.q_neutral
    data = forward_kinematics(panda, q, compute_frames=True)
    frame_name = _ee_frame(panda)
    T_target = data.frame_pose_world[panda.frame_id(frame_name)].clone()

    result = solve_ik(
        panda,
        {frame_name: T_target},
        initial_q=q,
        cost_cfg=IKCostConfig(limit_weight=0.0, rest_weight=0.0),
        optimizer_cfg=OptimizerConfig(max_iter=30),
    )
    # At the target, pose residual should be small
    data_sol = forward_kinematics(panda, result.q, compute_frames=True)
    T_sol = data_sol.frame_pose_world[panda.frame_id(frame_name)]
    # Position error < 1cm
    pos_err = float((T_sol[:3] - T_target[:3]).norm())
    assert pos_err < 0.01, f"Position error too large: {pos_err:.4f} m"


def test_solve_ik_reachable_target(panda):
    """solve_ik should reach a known reachable target."""
    # Start from a configuration within joint limits (q_neutral may violate limits
    # on some joints like panda_joint4 whose range is [-3.07, -0.07]).
    q_ref = _feasible_q(panda).clone()
    # Offset first joint slightly
    q_ref[0] = 0.3
    data = forward_kinematics(panda, q_ref, compute_frames=True)
    frame_name = _ee_frame(panda)
    T_target = data.frame_pose_world[panda.frame_id(frame_name)].clone()

    result = solve_ik(
        panda,
        {frame_name: T_target},
        cost_cfg=IKCostConfig(limit_weight=0.0, rest_weight=0.0),
        optimizer_cfg=OptimizerConfig(max_iter=100),
    )
    data_sol = forward_kinematics(panda, result.q, compute_frames=True)
    T_sol = data_sol.frame_pose_world[panda.frame_id(frame_name)]
    pos_err = float((T_sol[:3] - T_target[:3]).norm())
    assert pos_err < 0.01, f"Position error: {pos_err:.4f} m"


def test_solve_ik_limits_respected(panda):
    """After solve_ik, q should be within joint limits."""
    # Use a starting configuration within limits so the IK and bound-clamping
    # don't conflict (q_neutral can have joints outside their limits).
    q = _feasible_q(panda)
    data = forward_kinematics(panda, q, compute_frames=True)
    frame_name = _ee_frame(panda)
    T_target = data.frame_pose_world[panda.frame_id(frame_name)].clone()

    result = solve_ik(panda, {frame_name: T_target}, initial_q=q)
    lo = panda.lower_pos_limit
    hi = panda.upper_pos_limit
    # Clamp and check equality (within floating-point tolerance)
    q_clamped = result.q.clamp(lo, hi)
    assert torch.allclose(result.q, q_clamped, atol=1e-5), \
        "Result q is outside joint limits"


def test_ik_result_fk(panda):
    """IKResult.fk() returns Data with correct FK."""
    q = panda.q_neutral
    data = forward_kinematics(panda, q, compute_frames=True)
    frame_name = _ee_frame(panda)
    T_target = data.frame_pose_world[panda.frame_id(frame_name)].clone()

    result = solve_ik(panda, {frame_name: T_target}, initial_q=q)
    fk_data = result.fk()
    assert fk_data.joint_pose_world is not None
    assert fk_data.joint_pose_world.shape == (panda.njoints, 7)


def test_ik_result_frame_pose(panda):
    """IKResult.frame_pose(name) returns a 7-vector."""
    q = panda.q_neutral
    data = forward_kinematics(panda, q, compute_frames=True)
    frame_name = _ee_frame(panda)
    T_target = data.frame_pose_world[panda.frame_id(frame_name)].clone()

    result = solve_ik(panda, {frame_name: T_target}, initial_q=q)
    T = result.frame_pose(frame_name)
    assert T.shape == (7,)


# ── pluggable optimiser smoke tests ──────────────────────────────────────────

@pytest.mark.parametrize("optimizer", ["lm", "gn", "adam", "lbfgs"])
def test_solve_ik_optimizers_reach_target(panda, optimizer):
    """Every optimiser should drive the Panda EE to a reachable target."""
    q_ref = _feasible_q(panda).clone()
    q_ref[0] = 0.3
    data = forward_kinematics(panda, q_ref, compute_frames=True)
    frame_name = _ee_frame(panda)
    T_target = data.frame_pose_world[panda.frame_id(frame_name)].clone()

    # Adam needs more iterations and a higher lr than the defaults; keep the
    # test quick by giving every solver enough budget.
    max_iter = {"lm": 100, "gn": 100, "adam": 600, "lbfgs": 100}[optimizer]
    result = solve_ik(
        panda,
        {frame_name: T_target},
        initial_q=_feasible_q(panda),
        cost_cfg=IKCostConfig(limit_weight=0.0, rest_weight=0.0),
        optimizer_cfg=OptimizerConfig(optimizer=optimizer, max_iter=max_iter),
    )
    data_sol = forward_kinematics(panda, result.q, compute_frames=True)
    T_sol = data_sol.frame_pose_world[panda.frame_id(frame_name)]
    pos_err = float((T_sol[:3] - T_target[:3]).norm())
    assert pos_err < 0.05, f"{optimizer}: position error {pos_err:.4f} m too large"


# ── floating-base (G1) regression ────────────────────────────────────────────

@pytest.fixture(scope="module")
def g1():
    pytest.importorskip("robot_descriptions")
    from robot_descriptions import g1_description
    return load(g1_description.URDF_PATH, free_flyer=True)


def test_g1_is_free_flyer(g1):
    """G1 loaded with free_flyer=True has nq = nv + 1 and JointFreeFlyer at index 1."""
    from better_robot.data_model.joint_models import JointFreeFlyer
    assert g1.nq == g1.nv + 1
    assert isinstance(g1.joint_models[1], JointFreeFlyer)


def test_solve_ik_floating_base_with_limits(g1):
    """solve_ik with default limit_weight must run end-to-end on a free-flyer robot.

    This is the Phase 4 pass criterion for examples/02_g1_floating_ik.py —
    the limits-residual Jacobian has to be in nv space, not nq, otherwise
    CostStack.jacobian fails to concatenate against the pose Jacobian.
    """
    # Neutral with base at standing height
    q0 = g1.q_neutral.clone()
    q0[2] = 0.78   # base z
    q0[6] = 1.0    # qw
    q0[7:] = q0[7:].clamp(g1.lower_pos_limit[7:], g1.upper_pos_limit[7:])

    # FK to get a reachable target for whichever EE frame exists
    data = forward_kinematics(g1, q0, compute_frames=True)
    candidates = [
        "body_left_rubber_hand",
        "body_right_rubber_hand",
        "body_left_ankle_roll_link",
        "body_right_ankle_roll_link",
    ]
    ee = next((c for c in candidates if c in g1.frame_name_to_id), g1.frame_names[-1])
    T_target = data.frame_pose_world[g1.frame_id(ee)].clone()

    result = solve_ik(
        g1,
        {ee: T_target},
        initial_q=q0,
        cost_cfg=IKCostConfig(limit_weight=0.1, rest_weight=0.001),
        optimizer_cfg=OptimizerConfig(max_iter=30),
    )
    assert result.q.shape == (g1.nq,)
    # We gave the solver FK(q0) as the target, so it should reach it.
    data_sol = forward_kinematics(g1, result.q, compute_frames=True)
    T_sol = data_sol.frame_pose_world[g1.frame_id(ee)]
    pos_err = float((T_sol[:3] - T_target[:3]).norm())
    assert pos_err < 0.05, f"G1 floating-base IK pos_err = {pos_err:.4f} m"


def test_joint_position_limit_jacobian_shape_floating_base(g1):
    """JointPositionLimit.jacobian must return (2*nq, nv), not (2*nq, nq)."""
    from better_robot.residuals.base import ResidualState
    from better_robot.residuals.limits import JointPositionLimit

    q = g1.q_neutral.clone()
    q[6] = 1.0
    data = forward_kinematics(g1, q, compute_frames=True)
    state = ResidualState(model=g1, data=data, variables=q)

    res = JointPositionLimit(g1)
    J = res.jacobian(state)
    assert J.shape == (2 * g1.nq, g1.nv), \
        f"expected (2*nq={2*g1.nq}, nv={g1.nv}), got {tuple(J.shape)}"
