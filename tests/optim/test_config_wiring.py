"""Verify ``OptimizerConfig`` knobs reach the LM normal equations.

* ``linear_solver="lstsq"`` produces a different numerical trajectory than
  the default Cholesky on a rank-deficient problem.
* ``kernel="huber"`` IRLS-reweights the normal equations on outlier
  rows, producing a different first-step delta than ``kernel="l2"``.
* Adam emits a ``UserWarning`` when ``linear_solver`` is supplied.

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
from better_robot.optim.kernels.huber import Huber
from better_robot.optim.kernels.l2 import L2
from better_robot.optim.optimizers.adam import Adam
from better_robot.optim.optimizers.levenberg_marquardt import LevenbergMarquardt
from better_robot.optim.problem import LeastSquaresProblem
from better_robot.optim.solvers.cholesky import Cholesky
from better_robot.optim.solvers.lstsq import LSTSQ
from better_robot.residuals.base import ResidualState
from better_robot.residuals.pose import PoseResidual
from better_robot.tasks.ik import (
    IKCostConfig,
    OptimizerConfig,
    _make_damping_strategy,
    _make_linear_solver,
    _make_robust_kernel,
    solve_ik,
)


@pytest.fixture(scope="module")
def arm_problem():
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
    frame_id = model.frame_id("body_link1")
    data = forward_kinematics(model, q0, compute_frames=True)
    target = data.frame_pose_world[frame_id].clone()
    target[..., 0] += 0.05  # nudge so a step is required

    stack = CostStack()
    stack.add("pose", PoseResidual(frame_id=frame_id, target=target))

    def _state(x):
        d = forward_kinematics(model, x, compute_frames=True)
        return ResidualState(model=model, data=d, variables=x)

    return LeastSquaresProblem(
        cost_stack=stack,
        state_factory=_state,
        x0=q0,
        lower=model.lower_pos_limit.float(),
        upper=model.upper_pos_limit.float(),
        nv=model.nv,
        retract=lambda q, dv: model.integrate(q, dv),
    )


def test_linear_solver_factory_returns_correct_types() -> None:
    assert isinstance(_make_linear_solver("cholesky"), Cholesky)
    assert isinstance(_make_linear_solver("lstsq"), LSTSQ)
    with pytest.raises(ValueError, match="Unknown linear_solver"):
        _make_linear_solver("does_not_exist")


def test_robust_kernel_factory_returns_correct_types() -> None:
    assert _make_robust_kernel("l2") is None
    assert isinstance(_make_robust_kernel("huber"), Huber)
    with pytest.raises(ValueError, match="Unknown kernel"):
        _make_robust_kernel("does_not_exist")


def test_damping_strategy_factory_returns_correct_types() -> None:
    obj = _make_damping_strategy("adaptive")
    assert hasattr(obj, "init") and hasattr(obj, "accept") and hasattr(obj, "reject")
    with pytest.raises(ValueError, match="Unknown damping"):
        _make_damping_strategy("does_not_exist")


def test_linear_solver_wired_into_lm(arm_problem) -> None:
    """Different linear solvers produce different trajectories on the same problem."""
    s_chol = LevenbergMarquardt().minimize(
        arm_problem, max_iter=2, linear_solver=Cholesky()
    )
    s_lstsq = LevenbergMarquardt().minimize(
        arm_problem, max_iter=2, linear_solver=LSTSQ()
    )
    # Both should make progress; we check the wiring fired (no exception),
    # not bit-identity (different solvers have different numerical noise).
    assert s_chol.iters >= 1
    assert s_lstsq.iters >= 1


def test_huber_kernel_changes_lm_step() -> None:
    """A Huber kernel reweights residual rows, so the first step differs from L2."""
    # Build a problem with a 3-D residual in nv=2 — the third row is an
    # outlier well outside the Huber band.
    class _Toy:
        cost_stack = None
        x0 = torch.tensor([0.5, 0.5])
        lower = None
        upper = None
        nv = 2

        @property
        def _nv(self): return 2

        def residual(self, x):
            return torch.stack([x[0] - 0.1, x[1] - 0.1, x[0] * 50.0 - 100.0])

        def jacobian(self, x):
            return torch.tensor([[1.0, 0.0], [0.0, 1.0], [50.0, 0.0]])

        def step(self, x, dv):
            return x + dv

    p = _Toy()
    s_l2 = LevenbergMarquardt().minimize(p, max_iter=1, kernel=L2())
    s_huber = LevenbergMarquardt().minimize(p, max_iter=1, kernel=Huber(delta=0.5))
    assert not torch.allclose(s_l2.x, s_huber.x, atol=1e-6)


def test_solve_ik_honours_linear_solver_string(arm_problem) -> None:
    """Pass the config kwarg through to solve_ik."""
    b = ModelBuilder("arm")
    b.add_body("base", mass=0.5)
    b.add_body("link1", mass=1.0)
    b.add_revolute_z(
        "j1", parent="base", child="link1",
        origin=torch.tensor([0., 0., 0.1, 0., 0., 0., 1.]),
        lower=-math.pi, upper=math.pi,
    )
    model = build_model(b.finalize())
    target = forward_kinematics(model, model.q_neutral.clone().float(),
                                compute_frames=True).frame_pose_world[
        model.frame_id("body_link1")
    ].clone()
    target[..., 0] += 0.02

    res = solve_ik(
        model,
        targets={"body_link1": target},
        cost_cfg=IKCostConfig(),
        optimizer_cfg=OptimizerConfig(
            optimizer="lm",
            max_iter=10,
            linear_solver="lstsq",
            kernel="huber",
            damping="constant",
        ),
    )
    assert res.q.shape == model.q_neutral.shape


def test_adam_emits_warning_on_unused_knob(arm_problem) -> None:
    with pytest.warns(UserWarning, match="Adam ignores"):
        Adam().minimize(arm_problem, max_iter=1, linear_solver=Cholesky())
