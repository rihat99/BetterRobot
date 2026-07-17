"""Tests for phased IK and the retained legacy composite solver.

Pattern from cuRobo (docs/concepts/tasks.md): LM coarse solve → LBFGS refinement.

This suite checks:
1. ``solve_ik(optimizer="lm_then_adam")`` reaches a reachable Panda target.
2. ``LMThenLBFGS.minimize`` returns a ``SolverState``.
3. ``stage2_disabled_items`` correctly reactivates items after refinement.
4. ``LMThenLBFGS`` satisfies the ``Optimizer`` Protocol.
"""

from __future__ import annotations

import pytest
import torch

from better_robot.io import load
from better_robot.kinematics.forward import forward_kinematics
from better_robot.optim.optimizers.base import Optimizer
from better_robot.optim.optimizers.lm_then_lbfgs import LMThenLBFGS
from better_robot.optim.state import SolverState
from better_robot.tasks.ik import IKCostConfig, OptimizerConfig, solve_ik


@pytest.fixture(scope="module")
def panda():
    pytest.importorskip("robot_descriptions")
    from robot_descriptions import panda_description  # noqa: PLC0415

    return load(panda_description.URDF_PATH)


def _ee_frame(model) -> str:
    for candidate in ("body_panda_hand", "body_panda_link8", "body_panda_link7"):
        if candidate in model.frame_name_to_id:
            return candidate
    return model.frame_names[-1]


def _feasible_q(model) -> torch.Tensor:
    return model.q_neutral.clone().clamp(model.lower_pos_limit, model.upper_pos_limit)


def test_lm_then_lbfgs_satisfies_optimizer_protocol() -> None:
    assert isinstance(LMThenLBFGS(), Optimizer)


def test_lm_then_adam_reaches_panda_target(panda) -> None:
    q_ref = _feasible_q(panda).clone()
    q_ref[0] = 0.25
    data = forward_kinematics(panda, q_ref, compute_frames=True)
    frame_name = _ee_frame(panda)
    T_target = data.frame_pose_world[panda.frame_id(frame_name)].clone()

    result = solve_ik(
        panda,
        {frame_name: T_target},
        initial_q=_feasible_q(panda),
        cost_cfg=IKCostConfig(limit_weight=0.0, rest_weight=0.0),
        optimizer_cfg=OptimizerConfig(optimizer="lm_then_adam", max_iter=80),
    )
    data_sol = forward_kinematics(panda, result.q, compute_frames=True)
    T_sol = data_sol.frame_pose_world[panda.frame_id(frame_name)]
    pos_err = float((T_sol[:3] - T_target[:3]).norm())
    assert pos_err < 0.02, f"lm_then_adam position error {pos_err:.4f} m"


def test_lm_then_adam_returns_public_diagnostics(panda) -> None:
    q = _feasible_q(panda)
    data = forward_kinematics(panda, q, compute_frames=True)
    frame_name = _ee_frame(panda)
    T_target = data.frame_pose_world[panda.frame_id(frame_name)].clone()

    result = solve_ik(
        panda,
        {frame_name: T_target},
        initial_q=q,
        optimizer_cfg=OptimizerConfig(optimizer="lm_then_adam", max_iter=20),
    )
    # ``solve_ik`` wraps the raw solver result into ``IKResult``; we can
    # still exercise the underlying state via the composite directly.
    assert hasattr(result, "q")
    assert hasattr(result, "converged")
    assert isinstance(result.iters, int)
    assert isinstance(result.converged, bool)


@pytest.mark.parametrize("optimizer", ["lbfgs", "lm_then_lbfgs"])
def test_solve_ik_fails_honestly_for_deferred_lbfgs(panda, optimizer) -> None:
    q = _feasible_q(panda)
    frame_name = _ee_frame(panda)
    target = forward_kinematics(panda, q, compute_frames=True).frame_pose_world[panda.frame_id(frame_name)]

    with pytest.raises(NotImplementedError, match="L-BFGS|deferred|lm_then_adam"):
        solve_ik(
            panda,
            {frame_name: target},
            initial_q=q,
            optimizer_cfg=OptimizerConfig(optimizer=optimizer),
        )


def test_lm_then_lbfgs_reactivates_disabled_cost_items(panda) -> None:
    """``stage2_disabled_items`` must restore active flags even on non-convergence."""
    q = _feasible_q(panda)
    data = forward_kinematics(panda, q, compute_frames=True)
    frame_name = _ee_frame(panda)
    T_target = data.frame_pose_world[panda.frame_id(frame_name)].clone()

    from better_robot.costs.stack import CostStack  # noqa: PLC0415
    from better_robot.optim.problem import LeastSquaresProblem  # noqa: PLC0415
    from better_robot.residuals.base import ResidualState  # noqa: PLC0415
    from better_robot.residuals.pose import PoseResidual  # noqa: PLC0415
    from better_robot.residuals.regularization import RestResidual  # noqa: PLC0415

    stack = CostStack()
    stack.add("pose", PoseResidual(frame_id=panda.frame_id(frame_name), target=T_target))
    stack.add("rest", RestResidual(panda, panda.q_neutral), weight=0.01)

    def _state_factory(x):
        d = forward_kinematics(panda, x, compute_frames=True)
        return ResidualState(model=panda, data=d, variables=x)

    problem = LeastSquaresProblem(
        cost_stack=stack,
        state_factory=_state_factory,
        x0=q,
        lower=panda.lower_pos_limit,
        upper=panda.upper_pos_limit,
        nv=panda.nv,
        retract=panda.integrate,
    )
    assert stack.items["rest"].active is True

    opt = LMThenLBFGS(stage2_disabled_items=("rest",))
    state = opt.minimize(problem, max_iter=20)
    # The disabled item must be reactivated after the refinement.
    assert stack.items["rest"].active is True
    assert isinstance(state, SolverState)
