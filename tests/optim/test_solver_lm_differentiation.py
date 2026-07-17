"""Manifold KKT and explicit unrolled-step differentiation regressions."""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

import torch

from better_robot.optim.blocks import (
    LevenbergMarquardt,
    LMStatus,
    Problem,
    ResidualItem,
    SO3Manifold,
    VarSpec,
)


class _PeriodicSO3Residual:
    name = "periodic_so3"
    reads = ("rotation",)
    dim = 1

    def __call__(self, ctx: Mapping[str, Any]) -> torch.Tensor:
        rotation = ctx["rotation"]
        return 1.0 + 4.0 * math.pi * rotation[..., :1]

    def jacobian_blocks(self, ctx: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        rotation = ctx["rotation"]
        row = rotation.new_tensor([2.0 * math.pi, 0.0, 0.0])
        return {"rotation": row.expand(*rotation.shape[:-1], 1, 3)}


class _QuadraticResidual:
    name = "quadratic"
    reads = ("x",)
    dim = 1

    def __call__(self, ctx: Mapping[str, Any]) -> torch.Tensor:
        return ctx["x"].square() - 2.0


def test_unbounded_so3_kkt_uses_raw_gradient_without_periodic_wrap() -> None:
    problem = Problem(
        vars=(VarSpec("rotation", (4,), manifold=SO3Manifold()),),
        residuals=(ResidualItem("periodic_so3", _PeriodicSO3Residual()),),
    )
    values = {"rotation": torch.tensor([0.0, 0.0, 0.0, 1.0])}

    state = LevenbergMarquardt(jacobian_strategy="analytic").init_state(values, problem)

    torch.testing.assert_close(state.grad_norm, torch.tensor(2.0 * math.pi))
    torch.testing.assert_close(state.projected_grad_norm, state.grad_norm)
    assert int(state.status) == int(LMStatus.RUNNING)


def _one_quadratic_step(value: torch.Tensor, *, create_graph: bool) -> torch.Tensor:
    problem = Problem(
        vars=(VarSpec("x", (1,)),),
        residuals=(ResidualItem("quadratic", _QuadraticResidual()),),
    )
    solver = LevenbergMarquardt(max_iter=1, jacobian_strategy="jacrev")
    values = {"x": value}
    state = solver.init_state(values, problem, create_graph=create_graph)
    next_values, _ = solver.update(values, state, problem, create_graph=create_graph)
    return next_values["x"]


def test_explicit_unrolled_step_preserves_the_exact_iteration_derivative() -> None:
    x = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)

    y = _one_quadratic_step(x, create_graph=True)
    derivative = torch.autograd.grad(y.sum(), x)[0]

    eps = 1e-5
    plus = _one_quadratic_step(torch.tensor([1.0 + eps], dtype=torch.float64), create_graph=False)
    minus = _one_quadratic_step(torch.tensor([1.0 - eps], dtype=torch.float64), create_graph=False)
    finite_difference = (plus - minus) / (2.0 * eps)

    assert y.requires_grad
    torch.testing.assert_close(derivative, finite_difference, atol=2e-7, rtol=2e-6)
    assert derivative < -0.49


def test_detached_run_never_appears_to_support_unrolled_backpropagation() -> None:
    x = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
    problem = Problem(
        vars=(VarSpec("x", (1,)),),
        residuals=(ResidualItem("quadratic", _QuadraticResidual()),),
    )

    values, state = LevenbergMarquardt(max_iter=2, jacobian_strategy="jacrev").run(
        {"x": x},
        problem,
    )

    assert not values["x"].requires_grad
    assert all(not tensor.requires_grad for tensor in state)
