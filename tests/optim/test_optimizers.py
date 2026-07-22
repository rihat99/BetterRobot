"""Focused contracts for the optimizer interface and torch adapter."""

from __future__ import annotations

from dataclasses import fields
from inspect import signature

import pytest
import torch

from better_robot.optim import LevenbergMarquardt
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


def _updatable_target_problem(initial: torch.Tensor, target: torch.Tensor) -> tuple[Problem, Variable]:
    value = Variable(initial.clone(), name="value")
    target_value = Variable(target.clone(), name="target_value", trainable=False)

    @residual(value, target_value, dim=initial.shape[-1], name="target")
    def target_error(current: torch.Tensor, desired: torch.Tensor) -> torch.Tensor:
        return current - desired

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
        scheduler=lambda inner: torch.optim.lr_scheduler.StepLR(inner, step_size=1, gamma=0.5),
    )
    info = optimizer.optimize()

    assert bool(info.converged.all())
    assert optimizer._scheduler is not None
    assert optimizer._scheduler.last_epoch == 1
    assert optimizer._scheduler.get_last_lr() == [0.5]
    optimizer.step()
    assert optimizer._scheduler.last_epoch == 1
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
    optimizer = TorchOptimizer(
        problem,
        torch.optim.SGD,
        lr=0.2,
        max_iterations=2,
        scheduler=lambda inner: torch.optim.lr_scheduler.StepLR(inner, step_size=1),
    )
    optimizer.step()
    first_instance = optimizer._optimizer
    first_scheduler = optimizer._scheduler

    problem.update({"value": torch.zeros(3, 1)})
    info = optimizer.step()

    assert optimizer._optimizer is not first_instance
    assert optimizer._scheduler is not first_scheduler
    assert info.status.shape == (3,)
    assert value.tensor.shape == (3, 1)


def test_same_layout_update_preserves_adam_moments_and_scheduler() -> None:
    problem, _value = _updatable_target_problem(torch.zeros(1), torch.ones(1))
    optimizer = TorchOptimizer(
        problem,
        torch.optim.Adam,
        lr=0.1,
        max_iterations=2,
        scheduler=lambda inner: torch.optim.lr_scheduler.StepLR(inner, step_size=1, gamma=0.5),
    )
    optimizer.step()
    assert optimizer._optimizer is not None and optimizer._scheduler is not None
    first_optimizer = optimizer._optimizer
    first_scheduler = optimizer._scheduler
    parameter = next(iter(optimizer._buffers.values()))
    moments = {name: value.clone() for name, value in first_optimizer.state[parameter].items()}
    scheduler_state = first_scheduler.state_dict()

    problem.update({"target_value": torch.tensor([-1.0])})
    optimizer.resume()

    assert optimizer._optimizer is first_optimizer
    assert optimizer._scheduler is first_scheduler
    assert optimizer._scheduler.state_dict() == scheduler_state
    for name, expected in moments.items():
        torch.testing.assert_close(first_optimizer.state[parameter][name], expected)
    expected_cost = problem.objective()
    optimizer.step()
    torch.testing.assert_close(optimizer._cost, expected_cost)
    assert optimizer._scheduler.last_epoch == scheduler_state["last_epoch"] + 1


def test_scheduler_does_not_step_when_the_gradient_is_already_converged() -> None:
    problem, _value = _target_problem(torch.ones(1), torch.ones(1))
    optimizer = TorchOptimizer(
        problem,
        torch.optim.SGD,
        lr=0.1,
        tolerance=0.0,
        scheduler=lambda inner: torch.optim.lr_scheduler.StepLR(inner, step_size=1),
    )
    assert optimizer._scheduler is not None
    last_epoch = optimizer._scheduler.last_epoch

    info = optimizer.step()

    assert bool(info.converged)
    assert optimizer._scheduler.last_epoch == last_epoch


def test_resume_after_maxiter_keeps_counts_and_continues_to_convergence() -> None:
    problem, value = _target_problem(torch.zeros(1), torch.ones(1))
    optimizer = TorchOptimizer(
        problem,
        torch.optim.SGD,
        lr=0.5,
        max_iterations=2,
        tolerance=0.13,
    )

    first = optimizer.optimize()
    assert int(first.status) == OptimizerStatus.MAXITER
    assert int(first.iterations) == 2

    optimizer.resume()
    second = optimizer.optimize()

    assert bool(second.converged)
    assert int(second.iterations) == 3
    torch.testing.assert_close(value.tensor, torch.tensor([0.875]))


def test_resume_after_enabled_flip_optimizes_the_new_objective() -> None:
    value = Variable(torch.zeros(1), name="value")

    @residual(value, dim=1, name="positive")
    def positive(current: torch.Tensor) -> torch.Tensor:
        return current - 1.0

    @residual(value, dim=1, name="negative", enabled=False)
    def negative(current: torch.Tensor) -> torch.Tensor:
        return current + 1.0

    optimizer = TorchOptimizer(
        Problem([positive, negative]),
        torch.optim.SGD,
        lr=0.5,
        max_iterations=20,
        tolerance=1e-4,
    )
    first = optimizer.optimize()
    positive.enabled = False
    negative.enabled = True
    optimizer.resume()
    second = optimizer.optimize()

    assert bool(first.converged) and bool(second.converged)
    assert int(second.iterations) > int(first.iterations)
    torch.testing.assert_close(value.tensor, torch.tensor([-1.0]), atol=1e-4, rtol=0.0)


def test_lm_resume_keeps_values_and_counts_but_reinitializes_damping() -> None:
    problem, value = _updatable_target_problem(torch.zeros(1), torch.ones(1))
    optimizer = LevenbergMarquardt(problem, max_iterations=1, tolerance=0.0)
    first = optimizer.optimize()
    assert optimizer._state is not None
    optimizer._state = optimizer._state._replace(
        mu=torch.full_like(optimizer._state.mu, optimizer.mu_max),
        increase_factor=torch.full_like(optimizer._state.increase_factor, optimizer.increase_factor_max),
        gain_ratio=torch.ones_like(optimizer._state.gain_ratio),
    )
    problem.update({"target_value": torch.tensor([-1.0])})
    current = value.tensor
    expected = optimizer._init_state(problem._trainable_values(), problem)

    optimizer.resume()

    assert optimizer._state is not None
    assert value.tensor is current
    assert int(optimizer._state.status) == OptimizerStatus.RUNNING
    assert int(optimizer._state.iterations) == int(first.iterations)
    torch.testing.assert_close(optimizer._state.mu, expected.mu)
    torch.testing.assert_close(optimizer._state.increase_factor, expected.increase_factor)
    torch.testing.assert_close(optimizer._state.gain_ratio, expected.gain_ratio)
    torch.testing.assert_close(optimizer._state.cost, expected.cost)


def test_zero_iteration_optimize_refreshes_layout_after_update() -> None:
    problem, _ = _target_problem(torch.zeros(1), torch.ones(1))
    optimizer = TorchOptimizer(problem, torch.optim.SGD, lr=0.2, max_iterations=0)

    problem.update({"value": torch.zeros(3, 1)})
    info = optimizer.optimize()

    assert info.status.shape == info.iterations.shape == info.cost.shape == (3,)
    assert bool((info.status == OptimizerStatus.MAXITER).all())
    torch.testing.assert_close(info.cost, problem.objective())


def test_torch_optimizer_rejects_differentiation_and_invalid_controls() -> None:
    problem, _ = _target_problem(torch.zeros(1), torch.ones(1))
    with pytest.raises(ValueError, match="non-negative int"):
        TorchOptimizer(problem, torch.optim.Adam, max_iterations=-1)
    with pytest.raises(ValueError, match="finite non-negative"):
        TorchOptimizer(problem, torch.optim.Adam, tolerance=float("nan"))
    with pytest.raises(TypeError, match="scheduler must be callable"):
        TorchOptimizer(problem, torch.optim.Adam, scheduler=object())
    with pytest.raises(TypeError, match="scheduler must return"):
        TorchOptimizer(problem, torch.optim.Adam, scheduler=lambda _optimizer: object())

    optimizer = TorchOptimizer(problem, torch.optim.Adam)
    with pytest.raises(ValueError, match="does not support differentiation"):
        optimizer.optimize(differentiate="implicit")
