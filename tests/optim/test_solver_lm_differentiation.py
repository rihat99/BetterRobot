"""Manifold KKT and explicit unrolled-step differentiation regressions."""

from __future__ import annotations

import math

import torch

from better_robot.optim import (
    LevenbergMarquardt,
    OptimizerStatus,
    Problem,
    Residual,
    SO3Variable,
    Variable,
)


class _PeriodicSO3Residual(Residual):
    def __init__(self, rotation: SO3Variable) -> None:
        self.rotation = rotation
        super().__init__(rotation, dim=1, name="periodic_so3")

    def error(self) -> torch.Tensor:
        return 1.0 + 4.0 * math.pi * self.rotation.tensor[..., :1]

    def jacobian(self) -> tuple[torch.Tensor, ...]:
        rotation = self.rotation.tensor
        row = rotation.new_tensor([2.0 * math.pi, 0.0, 0.0])
        return (row.expand(*rotation.shape[:-1], 1, 3),)


class _QuadraticResidual(Residual):
    def __init__(self, x: Variable) -> None:
        self.x = x
        super().__init__(x, dim=1, name="quadratic")

    def error(self) -> torch.Tensor:
        return self.x.tensor.square() - 2.0


def test_unbounded_so3_kkt_uses_raw_gradient_without_periodic_wrap() -> None:
    rotation = SO3Variable(torch.tensor([0.0, 0.0, 0.0, 1.0]), name="rotation")
    problem = Problem([_PeriodicSO3Residual(rotation)])
    optimizer = LevenbergMarquardt(problem, jacobian_strategy="analytic")

    state = optimizer._init_state({"rotation": rotation.tensor}, problem)

    torch.testing.assert_close(state.grad_norm, torch.tensor(2.0 * math.pi))
    torch.testing.assert_close(state.projected_grad_norm, state.grad_norm)
    assert int(state.status) == int(OptimizerStatus.RUNNING)


def _one_quadratic_step(value: torch.Tensor, *, create_graph: bool) -> torch.Tensor:
    x = Variable(value, name="x")
    problem = Problem([_QuadraticResidual(x)])
    optimizer = LevenbergMarquardt(problem, max_iterations=1, jacobian_strategy="jacrev")
    values = {"x": value}
    state = optimizer._init_state(values, problem, create_graph=create_graph)
    next_values, _ = optimizer._update(values, state, problem, create_graph=create_graph)
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


def test_detached_optimize_never_appears_to_support_unrolled_backpropagation() -> None:
    x = Variable(torch.tensor([1.0], dtype=torch.float64, requires_grad=True), name="x")
    problem = Problem([_QuadraticResidual(x)])

    info = LevenbergMarquardt(problem, max_iterations=2, jacobian_strategy="jacrev").optimize()

    assert not x.tensor.requires_grad
    assert all(not tensor.requires_grad for tensor in (info.status, info.iterations, info.cost))
