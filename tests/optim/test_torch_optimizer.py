"""Retraction regressions for the matrix-free TorchOptimizer adapter."""

from __future__ import annotations

import pytest
import torch

from better_robot.lie import so3
from better_robot.optim import Bounds, Problem, SO3Variable, TorchOptimizer, Variable, residual


def _target_problem(target: torch.Tensor, *, mask=None, bounds=None):
    value = Variable(torch.zeros_like(target), name="x", mask=mask, bounds=bounds)

    @residual(value, dim=target.shape[-1])
    def target_error(current: torch.Tensor) -> torch.Tensor:
        return current - target

    return value, Problem([target_error])


@pytest.mark.parametrize(
    ("optimizer_cls", "options", "iterations"),
    (
        (torch.optim.Adam, {"lr": 0.08}, 250),
        (torch.optim.SGD, {"lr": 0.2}, 80),
    ),
)
def test_generic_torch_optimizer_classes_solve_without_jacobians(
    optimizer_cls,
    options,
    iterations: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = torch.tensor([0.5, -0.25])
    value, problem = _target_problem(target)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("first-order optimization assembled a Jacobian")

    monkeypatch.setattr(Problem, "jacobian_blocks", forbidden)
    monkeypatch.setattr(Problem, "dense_jacobian", forbidden)
    info = TorchOptimizer(
        problem,
        optimizer_cls,
        max_iterations=iterations,
        tolerance=1e-6,
        **options,
    ).optimize()
    assert bool(info.converged)
    torch.testing.assert_close(value.tensor, target, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(info.cost, problem.objective())


def test_retraction_enforces_masks_bounds_and_group_projection() -> None:
    target = torch.tensor([0.9, 0.8, -0.9])
    bounds = Bounds(torch.tensor([-0.2, -1.0, -0.3]), torch.tensor([0.2, 1.0, 0.3]))
    value, problem = _target_problem(target, mask=torch.tensor([True, False, True]), bounds=bounds)
    TorchOptimizer(problem, torch.optim.SGD, lr=0.2, max_iterations=12).optimize()
    torch.testing.assert_close(value.tensor, torch.tensor([0.2, 0.0, -0.3]), rtol=0.0, atol=1e-7)

    rotation_target = torch.tensor([0.2, -0.1, 0.15])
    rotation = SO3Variable(torch.tensor([0.0, 0.0, 0.0, 1.0]), name="rotation")

    @residual(rotation, dim=3)
    def rotation_error(current: torch.Tensor) -> torch.Tensor:
        return so3.log(current) - rotation_target

    info = TorchOptimizer(
        Problem([rotation_error]),
        torch.optim.Adam,
        lr=0.08,
        max_iterations=250,
        tolerance=1e-6,
    ).optimize()
    assert bool(info.converged)
    torch.testing.assert_close(rotation.tensor.norm(), torch.tensor(1.0), atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(so3.log(rotation.tensor), rotation_target, atol=2e-5, rtol=2e-5)
