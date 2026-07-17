"""Truthfulness regressions for the public inverse-kinematics facade."""

from __future__ import annotations

import math

import pytest
import torch

from better_robot.io.build_model import build_model
from better_robot.io.parsers.programmatic import ModelBuilder
from better_robot.kinematics.forward import forward_kinematics
from better_robot.optim.state import SolverState
from better_robot.tasks.ik import IKCostConfig, OptimizerConfig, solve_ik


def _arm_model():
    builder = ModelBuilder("truthful_ik_arm")
    builder.add_body("base", mass=0.5)
    builder.add_body("link", mass=1.0)
    builder.add_revolute_z(
        "joint",
        parent="base",
        child="link",
        origin=torch.tensor([0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0]),
        lower=-math.pi,
        upper=math.pi,
    )
    return build_model(builder.finalize())


def _neutral_target(model) -> torch.Tensor:
    data = forward_kinematics(model, model.q_neutral, compute_frames=True)
    return data.frame_pose_world[model.frame_id("body_link")].clone()


def test_solve_ik_accepts_batched_initial_configuration() -> None:
    model = _arm_model()
    q_batch = model.q_neutral.expand(4, -1).clone()

    result = solve_ik(
        model,
        {"body_link": _neutral_target(model)},
        initial_q=q_batch,
    )

    assert result.q.shape == (4, model.nq)
    assert isinstance(result.iters, torch.Tensor) and result.iters.shape == (4,)
    assert isinstance(result.converged, torch.Tensor)
    assert result.converged.shape == (4,)
    assert bool(result.converged.all())


def test_solver_state_rejects_batched_residual_backstop() -> None:
    class BatchedProblem:
        x0 = torch.zeros(2)

        @staticmethod
        def residual(x: torch.Tensor) -> torch.Tensor:
            return torch.zeros(3, 2, dtype=x.dtype)

    with pytest.raises(NotImplementedError, match="Batched|batched|M2b"):
        SolverState.from_problem(BatchedProblem())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_solve_ik_preserves_working_dtype(dtype: torch.dtype) -> None:
    model = _arm_model().to(dtype=dtype)
    q0 = model.q_neutral.clone()
    target = _neutral_target(model)

    result = solve_ik(
        model,
        {"body_link": target},
        initial_q=q0,
        cost_cfg=IKCostConfig(limit_weight=0.0, rest_weight=0.0),
        optimizer_cfg=OptimizerConfig(max_iter=2),
    )

    assert result.q.dtype == dtype
    assert result.fk().joint_pose_world.dtype == dtype
