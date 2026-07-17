"""Document the projection-only bounded-LM weakness retained through M0."""

from __future__ import annotations

import pytest
import torch

from better_robot.costs.stack import CostStack
from better_robot.io import load
from better_robot.kinematics.forward import forward_kinematics
from better_robot.optim.optimizers.levenberg_marquardt import LevenbergMarquardt
from better_robot.optim.problem import LeastSquaresProblem
from better_robot.residuals.base import ResidualState
from better_robot.residuals.pose import PoseResidual

panda_description = pytest.importorskip("robot_descriptions.panda_description")


def _end_effector_frame(model) -> str:
    for candidate in ("body_panda_hand", "body_panda_link8", "body_panda_link7"):
        if candidate in model.frame_name_to_id:
            return candidate
    raise AssertionError("Panda end-effector frame is missing")


def test_bounded_lm_reports_maxiter_for_known_interior_target() -> None:
    """Keep the current failure honest; M2b's bounded solver flips this test."""
    model = load(panda_description.URDF_PATH)
    lower = model.lower_pos_limit
    upper = model.upper_pos_limit
    q_target = 0.7 * lower + 0.3 * upper
    q_start = model.q_neutral.clone().clamp(lower, upper)
    frame_id = model.frame_id(_end_effector_frame(model))
    target = forward_kinematics(
        model, q_target, compute_frames=True
    ).frame_pose_world[frame_id].clone()

    stack = CostStack()
    stack.add("pose", PoseResidual(frame_id=frame_id, target=target))

    def state_factory(q: torch.Tensor) -> ResidualState:
        data = forward_kinematics(model, q, compute_frames=True)
        return ResidualState(model=model, data=data, variables=q)

    problem = LeastSquaresProblem(
        cost_stack=stack,
        state_factory=state_factory,
        x0=q_start,
        lower=lower,
        upper=upper,
        nv=model.nv,
        retract=model.integrate,
    )

    # Documents the M0 bounded-LM weakness; M2b replaces projection-only LM
    # and should change these assertions to a successful KKT-aware solve.
    state = LevenbergMarquardt().minimize(problem, max_iter=100)

    assert state.converged is False
    assert state.status == "maxiter"
    assert state.residual_norm > 1e-2
    torch.testing.assert_close(state.x, state.x.clamp(lower, upper))
