"""Implicit differentiation through graph-carrying static variables."""

from __future__ import annotations

import pytest
import torch

import better_robot.optim.implicit as implicit_module
from better_robot.optim import LevenbergMarquardt, Problem, Variable, residual
from better_robot.optim.implicit import ImplicitDifferentiationError


def test_implicit_module_exports_only_public_configuration_and_error() -> None:
    assert implicit_module.__all__ == ["ImplicitDiffConfig", "ImplicitDifferentiationError"]


def test_implicit_backward_routes_solution_gradient_to_static_target() -> None:
    target_tensor = torch.tensor([1.25, -0.75], dtype=torch.float64, requires_grad=True)
    solution = Variable(torch.zeros(2, dtype=torch.float64), name="x")
    target = Variable(target_tensor, name="target", trainable=False)

    @residual(solution, target, dim=2)
    def difference(value: torch.Tensor, desired: torch.Tensor) -> torch.Tensor:
        return value - desired

    optimizer = LevenbergMarquardt(Problem([difference]), max_iterations=20, tolerance=1e-9)
    info = optimizer.optimize(differentiate="implicit")
    loss = solution.tensor.square().sum()
    gradient = torch.autograd.grad(loss, target_tensor)[0]

    assert bool(info.converged)
    torch.testing.assert_close(solution.tensor, target_tensor)
    torch.testing.assert_close(gradient, 2.0 * target_tensor)


def test_implicit_backward_handles_independent_batch_elements() -> None:
    target_tensor = torch.tensor([[1.0], [-2.0], [0.5]], dtype=torch.float64, requires_grad=True)
    solution = Variable(torch.zeros_like(target_tensor), name="x", batch_ndim=1)
    target = Variable(target_tensor, name="target", trainable=False, batch_ndim=1)

    @residual(solution, target, dim=1)
    def difference(value: torch.Tensor, desired: torch.Tensor) -> torch.Tensor:
        return value - desired

    LevenbergMarquardt(Problem([difference]), max_iterations=20, tolerance=1e-9).optimize(differentiate="implicit")
    gradient = torch.autograd.grad(solution.tensor.sum(), target_tensor)[0]

    torch.testing.assert_close(gradient, torch.ones_like(target_tensor))


def test_disconnected_graph_carrying_static_variable_fails_honestly() -> None:
    target_tensor = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
    solution = Variable(torch.zeros(1, dtype=torch.float64), name="x")
    target = Variable(target_tensor, name="unused", trainable=False)

    @residual(solution, target, dim=1)
    def objective(value: torch.Tensor, unused: torch.Tensor) -> torch.Tensor:
        del unused
        return value - 2.0

    LevenbergMarquardt(Problem([objective]), max_iterations=20, tolerance=1e-9).optimize(differentiate="implicit")

    with pytest.raises(ImplicitDifferentiationError, match="disconnected"):
        torch.autograd.grad(solution.tensor.sum(), target_tensor)
