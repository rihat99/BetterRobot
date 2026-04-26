"""``MultiStageOptimizer`` — generic chain of optimisers.

* Stage-wise weight overrides take effect during the stage and are
  rolled back after the stage completes (or raises).
* A stage that raises does not leak ``active`` / ``weight`` mutations
  onto the shared :class:`CostStack`.

See ``docs/design/07_RESIDUALS_COSTS_SOLVERS.md §5``.
"""

from __future__ import annotations

import math

import pytest
import torch

from better_robot.costs.stack import CostStack
from better_robot.io.build_model import build_model
from better_robot.io.parsers.programmatic import ModelBuilder
from better_robot.kinematics.forward import forward_kinematics
from better_robot.optim.optimizers.lbfgs import LBFGS
from better_robot.optim.optimizers.levenberg_marquardt import LevenbergMarquardt
from better_robot.optim.optimizers.multi_stage import MultiStageOptimizer, OptimizerStage
from better_robot.optim.problem import LeastSquaresProblem
from better_robot.optim.state import SolverState
from better_robot.residuals.base import ResidualState
from better_robot.residuals.pose import PoseResidual
from better_robot.residuals.regularization import RestResidual


@pytest.fixture
def problem_with_two_items():
    b = ModelBuilder("arm")
    b.add_body("base", mass=0.5)
    b.add_body("link1", mass=1.0)
    b.add_revolute_z(
        "j1", parent="base", child="link1",
        origin=torch.tensor([0., 0., 0.1, 0., 0., 0., 1.]),
        lower=-math.pi, upper=math.pi,
    )
    model = build_model(b.finalize())
    q0 = model.q_neutral.clone().float()
    fid = model.frame_id("body_link1")
    target = forward_kinematics(model, q0, compute_frames=True).frame_pose_world[fid].clone()
    target[..., 0] += 0.02

    stack = CostStack()
    stack.add("pose", PoseResidual(frame_id=fid, target=target), weight=1.0)
    stack.add("rest", RestResidual(model, model.q_neutral), weight=0.01)

    def _state(x):
        d = forward_kinematics(model, x, compute_frames=True)
        return ResidualState(model=model, data=d, variables=x)

    return stack, LeastSquaresProblem(
        cost_stack=stack,
        state_factory=_state,
        x0=q0,
        lower=model.lower_pos_limit.float(),
        upper=model.upper_pos_limit.float(),
        nv=model.nv,
        retract=lambda q, dv: model.integrate(q, dv),
    )


def test_multi_stage_runs_two_stages_in_order(problem_with_two_items) -> None:
    _, problem = problem_with_two_items
    opt = MultiStageOptimizer(stages=(
        OptimizerStage(optimizer=LevenbergMarquardt(), max_iter=5),
        OptimizerStage(optimizer=LBFGS(), max_iter=5),
    ))
    state = opt.minimize(problem)
    assert isinstance(state, SolverState)
    assert state.iters >= 1


def test_multi_stage_disabled_items_restored(problem_with_two_items) -> None:
    stack, problem = problem_with_two_items
    assert stack.items["rest"].active is True
    opt = MultiStageOptimizer(stages=(
        OptimizerStage(optimizer=LBFGS(), max_iter=3, disabled_items=("rest",)),
    ))
    opt.minimize(problem)
    assert stack.items["rest"].active is True, "stage left rest disabled"


def test_multi_stage_weight_overrides_restored(problem_with_two_items) -> None:
    stack, problem = problem_with_two_items
    original = stack.items["rest"].weight
    opt = MultiStageOptimizer(stages=(
        OptimizerStage(optimizer=LBFGS(), max_iter=3, weight_overrides={"rest": 99.0}),
    ))
    opt.minimize(problem)
    assert stack.items["rest"].weight == original


def test_multi_stage_restores_on_stage_raise(problem_with_two_items) -> None:
    """If a stage's optimiser raises, cost-stack mutations must roll back."""
    stack, problem = problem_with_two_items

    class Boom:
        _ignores_normal_eqn_kwargs = True
        def minimize(self, *args, **kwargs):
            raise RuntimeError("boom")

    opt = MultiStageOptimizer(stages=(
        OptimizerStage(optimizer=Boom(), max_iter=1, disabled_items=("rest",),
                       weight_overrides={"pose": 42.0}),
    ))
    with pytest.raises(RuntimeError, match="boom"):
        opt.minimize(problem)
    assert stack.items["rest"].active is True
    assert stack.items["pose"].weight == 1.0


def test_multi_stage_zero_stages_raises() -> None:
    with pytest.raises(ValueError, match="at least one stage"):
        MultiStageOptimizer(stages=())
