"""Focused contracts for the optimizer interface and torch adapter."""

from __future__ import annotations

from dataclasses import fields
from inspect import signature

import pytest
import torch

from better_robot.optim.optimizers import Optimizer, OptimizerInfo, OptimizerStatus, TorchOptimizer
from better_robot.optim.problem import Problem
from better_robot.optim.variables import Variable
from better_robot.residuals.base import residual


def _target_problem(initial: torch.Tensor, target: torch.Tensor) -> tuple[Problem, Variable]:
    value = Variable(initial.clone(), name="value", batch_ndim=max(0, initial.ndim - 1))

    @residual(value, dim=initial.shape[-1], name="target")
    def target_error(current: torch.Tensor) -> torch.Tensor:
        return current - target

    return Problem([target_error]), value


def test_optimizer_base_defaults_match_public_contract() -> None:
    parameters = signature(Optimizer).parameters
    assert parameters["max_iterations"].default == 50
    assert parameters["tolerance"].default == 1e-8


def test_optimizer_status_and_info_have_one_shared_minimal_contract() -> None:
    assert [member.value for member in OptimizerStatus] == [0, 1, 2, 3, 4]
    info = OptimizerInfo(
        status=torch.tensor([OptimizerStatus.RUNNING, OptimizerStatus.CONVERGED, OptimizerStatus.STALLED_AT_BOUNDS]),
        iterations=torch.tensor([0, 2, 3]),
        cost=torch.tensor([2.0, 1.0, 0.0]),
    )

    assert [field.name for field in fields(info)] == ["status", "iterations", "cost"]
    assert info.converged.tolist() == [False, True, True]


def test_optimizer_and_buffers_persist_while_batch_elements_stop_independently() -> None:
    target = torch.tensor([[0.0, 0.0], [0.5, -0.25]])
    problem, value = _target_problem(torch.zeros_like(target), target)
    instances: list[torch.optim.Optimizer] = []
    parameter_ids: list[tuple[int, ...]] = []

    def factory(parameters, **kwargs):
        leaves = tuple(parameters)
        parameter_ids.append(tuple(id(parameter) for parameter in leaves))
        instance = torch.optim.Adam(leaves, **kwargs)
        instances.append(instance)
        return instance

    optimizer = TorchOptimizer(
        problem,
        factory,
        lr=0.08,
        max_iterations=250,
        tolerance=1e-6,
    )
    info = optimizer.optimize()

    assert len(instances) == len(parameter_ids) == 1
    assert instances[0].state
    assert info.iterations.shape == info.status.shape == info.cost.shape == (2,)
    assert info.iterations[0] == 0
    assert info.iterations[1] > 1
    assert info.converged.tolist() == [True, True]
    torch.testing.assert_close(value.tensor[0], torch.zeros(2), rtol=0.0, atol=0.0)
    torch.testing.assert_close(value.tensor[1], target[1], atol=2e-5, rtol=2e-5)


def test_lbfgs_uses_a_closure_over_the_summed_objective() -> None:
    target = torch.tensor([[0.5, -0.25], [-0.2, 0.4]])
    problem, value = _target_problem(torch.zeros_like(target), target)
    optimizer = TorchOptimizer(
        problem,
        torch.optim.LBFGS,
        lr=1.0,
        max_iter=20,
        line_search_fn="strong_wolfe",
        max_iterations=3,
        tolerance=1e-7,
    )
    info = optimizer.optimize()

    assert bool(info.converged.all())
    torch.testing.assert_close(value.tensor, target, atol=1e-6, rtol=1e-6)


def test_reset_clears_algorithm_state_but_keeps_the_current_value() -> None:
    target = torch.tensor([0.5])
    problem, value = _target_problem(torch.zeros_like(target), target)
    optimizer = TorchOptimizer(problem, torch.optim.SGD, lr=0.2, max_iterations=10)
    optimizer.step()
    current = value.tensor
    first_instance = optimizer._optimizer

    optimizer.reset()

    assert value.tensor is current
    optimizer.step()
    assert optimizer._optimizer is not first_instance


def test_problem_update_rebuilds_batch_shaped_state() -> None:
    problem, value = _target_problem(torch.zeros(1), torch.ones(1))
    optimizer = TorchOptimizer(problem, torch.optim.SGD, lr=0.2, max_iterations=2)
    optimizer.step()
    first_instance = optimizer._optimizer

    problem.update({"value": torch.zeros(3, 1)})
    info = optimizer.step()

    assert optimizer._optimizer is not first_instance
    assert info.status.shape == (3,)
    assert value.tensor.shape == (3, 1)


def test_torch_optimizer_rejects_differentiation_and_invalid_controls() -> None:
    problem, _ = _target_problem(torch.zeros(1), torch.ones(1))
    with pytest.raises(ValueError, match="non-negative int"):
        TorchOptimizer(problem, torch.optim.Adam, max_iterations=-1)
    with pytest.raises(ValueError, match="finite non-negative"):
        TorchOptimizer(problem, torch.optim.Adam, tolerance=float("nan"))

    optimizer = TorchOptimizer(problem, torch.optim.Adam)
    with pytest.raises(ValueError, match="does not support differentiation"):
        optimizer.optimize(differentiate="implicit")
