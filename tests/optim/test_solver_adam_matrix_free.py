"""Tests for the thin, matrix-free ``torch.optim`` adapter."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import pytest
import torch

from better_robot.lie import so3
from better_robot.optim import Bounds, Problem, ResidualItem, SO3Manifold, VarSpec
from better_robot.optim.first_order import FirstOrderResult, run_first_order


class _TargetResidual:
    name = "target"
    reads = ("x", "target")

    def __init__(self, dim: int) -> None:
        self.dim = dim

    def __call__(self, ctx: Mapping[str, Any]) -> torch.Tensor:
        return ctx["x"] - ctx["target"]


class _RotationResidual:
    name = "rotation"
    reads = ("rotation", "target")
    dim = 3

    def __call__(self, ctx: Mapping[str, Any]) -> torch.Tensor:
        return so3.log(ctx["rotation"]) - ctx["target"]


def _target_problem(
    target: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    bounds: Bounds | None = None,
) -> Problem:
    return Problem(
        vars=(VarSpec("x", (target.shape[-1],), mask=mask, bounds=bounds),),
        residuals=(ResidualItem("target", _TargetResidual(target.shape[-1])),),
        parameters={"target": target},
    )


def _adam(parameters: Iterable[torch.Tensor]) -> torch.optim.Optimizer:
    return torch.optim.Adam(parameters, lr=0.08)


def _sgd(parameters: Iterable[torch.Tensor]) -> torch.optim.Optimizer:
    return torch.optim.SGD(parameters, lr=0.2)


@pytest.mark.parametrize("factory,max_iter", [(_adam, 250), (_sgd, 80)])
def test_generic_torch_optimizer_factories_solve_without_jacobians(
    factory,
    max_iter: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = torch.tensor([0.5, -0.25])
    problem = _target_problem(target)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("first-order optimization assembled a Jacobian")

    monkeypatch.setattr(Problem, "jacobian_blocks", forbidden)
    monkeypatch.setattr(Problem, "dense_jacobian", forbidden)
    values, result = run_first_order(
        {"x": torch.zeros_like(target)},
        problem,
        factory,
        max_iter=max_iter,
        tolerance=1e-6,
    )

    assert isinstance(result, FirstOrderResult)
    assert bool(result.converged)
    assert result.step.dtype == torch.int64
    torch.testing.assert_close(values["x"], target, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(result.cost, problem.objective(values))


def test_optimizer_and_leaf_buffers_persist_while_batch_elements_stop_independently() -> None:
    target = torch.tensor([[0.0, 0.0], [0.5, -0.25]])
    problem = _target_problem(target)
    optimizers: list[torch.optim.Optimizer] = []
    parameter_ids: list[tuple[int, ...]] = []

    def factory(parameters: Iterable[torch.Tensor]) -> torch.optim.Optimizer:
        leaves = tuple(parameters)
        parameter_ids.append(tuple(id(parameter) for parameter in leaves))
        optimizer = torch.optim.Adam(leaves, lr=0.08)
        optimizers.append(optimizer)
        return optimizer

    values, result = run_first_order(
        {"x": torch.zeros_like(target)},
        problem,
        factory,
        max_iter=250,
        tolerance=1e-6,
    )

    assert len(optimizers) == len(parameter_ids) == 1
    assert optimizers[0].state
    assert result.step.shape == result.converged.shape == result.cost.shape == (2,)
    assert result.step[0] == 0
    assert result.step[1] > 1
    assert result.converged.tolist() == [True, True]
    torch.testing.assert_close(values["x"][0], torch.zeros(2), rtol=0.0, atol=0.0)
    torch.testing.assert_close(values["x"][1], target[1], atol=2e-5, rtol=2e-5)


def test_retraction_enforces_masks_bounds_and_manifold_projection() -> None:
    target = torch.tensor([0.9, 0.8, -0.9])
    bounds = Bounds(torch.tensor([-0.2, -1.0, -0.3]), torch.tensor([0.2, 1.0, 0.3]))
    masked_problem = _target_problem(target, mask=torch.tensor([True, False, True]), bounds=bounds)
    masked_values, _ = run_first_order(
        {"x": torch.zeros_like(target)},
        masked_problem,
        _sgd,
        max_iter=12,
        tolerance=0.0,
    )
    torch.testing.assert_close(masked_values["x"], torch.tensor([0.2, 0.0, -0.3]), rtol=0.0, atol=1e-7)

    rotation_target = torch.tensor([0.2, -0.1, 0.15])
    rotation_problem = Problem(
        vars=(VarSpec("rotation", (4,), manifold=SO3Manifold()),),
        residuals=(ResidualItem("rotation", _RotationResidual()),),
        parameters={"target": rotation_target},
    )
    identity = torch.tensor([0.0, 0.0, 0.0, 1.0])
    rotation_values, rotation_result = run_first_order(
        {"rotation": identity},
        rotation_problem,
        _adam,
        max_iter=250,
        tolerance=1e-6,
    )
    assert bool(rotation_result.converged)
    torch.testing.assert_close(rotation_values["rotation"].norm(), torch.tensor(1.0), atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(so3.log(rotation_values["rotation"]), rotation_target, atol=2e-5, rtol=2e-5)
