"""TorchOptimizer behavior over object-referenced problems."""

from __future__ import annotations

import torch

from better_robot.optim.optimizers import OptimizerStatus, TorchOptimizer
from better_robot.optim.problem import Problem
from better_robot.optim.variables import Variable
from better_robot.residuals.base import residual


def _quadratic_problem(value: torch.Tensor, target: torch.Tensor, *, batch_ndim: int = 0):
    variable = Variable(value, name="x", batch_ndim=batch_ndim)

    @residual(variable, dim=value.shape[-1])
    def difference(current: torch.Tensor) -> torch.Tensor:
        return current - target

    return variable, Problem([difference])


def test_adam_optimizes_batched_elements_and_reports_one_info_type() -> None:
    target = torch.tensor([[1.0, -2.0], [-0.5, 0.25], [2.0, 1.5]])
    variable, problem = _quadratic_problem(torch.zeros_like(target), target, batch_ndim=1)
    optimizer = TorchOptimizer(
        problem,
        torch.optim.Adam,
        lr=0.15,
        max_iterations=250,
        tolerance=2e-4,
    )

    info = optimizer.optimize()

    torch.testing.assert_close(variable.tensor, target, rtol=2e-3, atol=2e-3)
    assert info.status.shape == (3,)
    assert info.iterations.shape == (3,)
    assert info.cost.shape == (3,)
    assert bool(info.converged.all())


def test_step_updates_the_referenced_variable_and_reset_clears_iterations() -> None:
    variable, problem = _quadratic_problem(torch.tensor([0.0]), torch.tensor([1.0]))
    optimizer = TorchOptimizer(problem, torch.optim.SGD, lr=0.25, max_iterations=20)

    first = optimizer.step()

    assert variable.tensor.item() > 0.0
    assert first.iterations.item() == 1
    optimizer.reset()
    reset = optimizer._initial_info()
    assert reset.iterations.item() == 0
    assert reset.status.item() == OptimizerStatus.RUNNING


def test_problem_update_resets_batch_state_before_the_next_step() -> None:
    variable, problem = _quadratic_problem(torch.tensor([0.0]), torch.tensor([1.0]))
    optimizer = TorchOptimizer(problem, torch.optim.Adam, lr=0.1, max_iterations=20)
    optimizer.step()

    problem.update({"x": torch.zeros(4, 1)})
    info = optimizer.step()

    assert variable.tensor.shape == (4, 1)
    assert info.status.shape == (4,)
    torch.testing.assert_close(info.iterations, torch.ones(4, dtype=torch.int64))


def test_lbfgs_closure_path_solves_a_small_quadratic() -> None:
    variable, problem = _quadratic_problem(torch.tensor([4.0, -3.0]), torch.tensor([1.0, 2.0]))
    optimizer = TorchOptimizer(
        problem,
        torch.optim.LBFGS,
        lr=1.0,
        max_iter=10,
        line_search_fn="strong_wolfe",
        max_iterations=5,
        tolerance=1e-7,
    )

    info = optimizer.optimize()

    torch.testing.assert_close(variable.tensor, torch.tensor([1.0, 2.0]), rtol=1e-5, atol=1e-5)
    assert info.cost.item() < 1e-10


def test_torch_optimizer_rejects_differentiation_mode() -> None:
    _variable, problem = _quadratic_problem(torch.tensor([0.0]), torch.tensor([1.0]))
    optimizer = TorchOptimizer(problem, torch.optim.SGD, lr=0.1)

    try:
        optimizer.optimize(differentiate="implicit")
    except ValueError as exc:
        assert "does not support differentiation" in str(exc)
    else:
        raise AssertionError("TorchOptimizer must reject differentiation")
