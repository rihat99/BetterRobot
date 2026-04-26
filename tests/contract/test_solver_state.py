"""Contract tests for ``SolverState`` — the shared optimizer record.

See ``docs/concepts/residuals_and_costs.md §5``.
"""

from __future__ import annotations

import math

import pytest
import torch

from better_robot.costs.stack import CostStack
from better_robot.io.build_model import build_model
from better_robot.io.parsers.programmatic import ModelBuilder
from better_robot.kinematics.forward import forward_kinematics
from better_robot.optim.optimizers.adam import Adam
from better_robot.optim.optimizers.gauss_newton import GaussNewton
from better_robot.optim.optimizers.lbfgs import LBFGS
from better_robot.optim.optimizers.levenberg_marquardt import LevenbergMarquardt
from better_robot.optim.problem import LeastSquaresProblem
from better_robot.optim.state import SolverState
from better_robot.residuals.base import ResidualState
from better_robot.residuals.pose import PoseResidual


@pytest.fixture(scope="module")
def arm_problem():
    b = ModelBuilder("arm")
    b.add_body("base", mass=0.5)
    b.add_body("link1", mass=1.0)
    b.add_revolute_z("j1", parent="base", child="link1",
                     origin=torch.tensor([0., 0., 0.1, 0., 0., 0., 1.]),
                     lower=-math.pi, upper=math.pi)
    model = build_model(b.finalize())

    q0 = model.q_neutral.clone().float()
    frame_id = model.frame_id("body_link1")
    data = forward_kinematics(model, q0, compute_frames=True)
    target = data.frame_pose_world[frame_id].clone()

    stack = CostStack()
    stack.add("pose", PoseResidual(frame_id=frame_id, target=target))

    def _state_factory(x: torch.Tensor) -> ResidualState:
        d = forward_kinematics(model, x, compute_frames=True)
        return ResidualState(model=model, data=d, variables=x)

    return LeastSquaresProblem(
        cost_stack=stack,
        state_factory=_state_factory,
        x0=q0,
        lower=model.lower_pos_limit.float(),
        upper=model.upper_pos_limit.float(),
        jacobian_strategy=None,
        nv=model.nv,
        retract=lambda q, dv: model.integrate(q, dv),
    )


@pytest.mark.parametrize("optimizer_cls", [LevenbergMarquardt, GaussNewton, Adam, LBFGS])
def test_minimize_returns_solver_state(arm_problem, optimizer_cls) -> None:
    """Every optimiser returns a ``SolverState`` with legal fields."""
    opt = optimizer_cls()
    state = opt.minimize(arm_problem, max_iter=5)

    assert isinstance(state, SolverState)
    assert state.status in ("running", "converged", "stalled", "maxiter")
    assert state.iters >= 0
    assert state.x.shape == arm_problem.x0.shape
    assert state.residual.ndim >= 1


def test_solver_state_converged_property(arm_problem) -> None:
    """The ``converged`` property mirrors ``status == 'converged'``."""
    opt = LevenbergMarquardt()
    state = opt.minimize(arm_problem, max_iter=50)
    assert state.converged is (state.status == "converged")


def test_solver_state_from_problem(arm_problem) -> None:
    """``SolverState.from_problem`` evaluates the residual once at ``x0``."""
    state = SolverState.from_problem(arm_problem)
    assert state.iters == 0
    assert state.status == "running"
    assert state.x.shape == arm_problem.x0.shape
    # residual_norm = 0.5 * ||r||² at x0
    assert float(state.residual_norm) == pytest.approx(
        0.5 * float((state.residual @ state.residual).sum())
    )


def test_optimization_result_is_alias_for_solver_state() -> None:
    """The deprecated ``OptimizationResult`` alias still resolves."""
    from better_robot.optim.optimizers.base import OptimizationResult
    assert OptimizationResult is SolverState
