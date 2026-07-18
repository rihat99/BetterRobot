"""Verify the live ``OptimizerConfig`` factories and ``solve_ik`` wiring.

See ``docs/concepts/solver_stack.md §5``.
"""

from __future__ import annotations

import math

import pytest
import torch

from better_robot.io.build_model import build_model
from better_robot.io.parsers.programmatic import ModelBuilder
from better_robot.kinematics.forward import forward_kinematics
from better_robot.optim.kernels.huber import Huber
from better_robot.optim.kernels.l2 import L2
from better_robot.optim.solvers.cholesky import Cholesky
from better_robot.optim.solvers.lstsq import LSTSQ
from better_robot.tasks.ik import (
    IKCostConfig,
    OptimizerConfig,
    _make_linear_solver,
    _make_robust_kernel,
    solve_ik,
)


def test_linear_solver_factory_returns_correct_types() -> None:
    assert isinstance(_make_linear_solver("cholesky"), Cholesky)
    assert isinstance(_make_linear_solver("lstsq"), LSTSQ)
    with pytest.raises(ValueError, match="Unknown linear_solver"):
        _make_linear_solver("cg")
    with pytest.raises(ValueError, match="Unknown linear_solver"):
        _make_linear_solver("does_not_exist")


def test_robust_kernel_factory_returns_correct_types() -> None:
    assert isinstance(_make_robust_kernel("l2"), L2)
    assert isinstance(_make_robust_kernel("huber"), Huber)
    with pytest.raises(ValueError, match="Unknown kernel"):
        _make_robust_kernel("does_not_exist")


def test_solve_ik_honours_linear_solver_string() -> None:
    """Pass the config kwarg through to solve_ik."""
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
    model = build_model(b.finalize())
    target = (
        forward_kinematics(model, model.q_neutral.clone().float(), compute_frames=True)
        .frame_pose_world[model.frame_id("body_link1")]
        .clone()
    )
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
