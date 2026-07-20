"""Object-owned Levenberg--Marquardt behavior contracts."""

from __future__ import annotations

import torch

from better_robot.optim import Bounds, LevenbergMarquardt, Problem, Variable, residual
from better_robot.optim.optimizers import OptimizerStatus


def _difference_problem(initial: torch.Tensor, target: torch.Tensor, *, bounds=None, batch_ndim: int = 0):
    variable = Variable(initial, name="x", bounds=bounds, batch_ndim=batch_ndim)

    @residual(variable, dim=initial.shape[-1])
    def difference(value: torch.Tensor) -> torch.Tensor:
        return value - target

    return variable, Problem([difference])


def test_target_line_fit_example_runs_as_written() -> None:
    x = torch.tensor([0.0, 1.0, 2.0, 3.0])
    y = torch.tensor([1.0, 3.0, 5.0, 7.0])
    theta = Variable(torch.zeros(2), name="theta")

    @residual(theta, dim=4)
    def line_fit(value: torch.Tensor) -> torch.Tensor:
        slope, intercept = value[..., 0:1], value[..., 1:2]
        return slope * x + intercept - y

    problem = Problem([line_fit])
    info = LevenbergMarquardt(problem, max_iterations=20).optimize()

    torch.testing.assert_close(theta.tensor, torch.tensor([2.0, 1.0]), rtol=1e-4, atol=1e-4)
    assert info.status.item() == OptimizerStatus.CONVERGED
    assert bool(info.converged)


def test_step_updates_referenced_values_and_returns_minimal_info() -> None:
    variable, problem = _difference_problem(torch.tensor([0.0]), torch.tensor([2.0]))
    optimizer = LevenbergMarquardt(problem, max_iterations=10)

    before = variable.tensor.clone()
    info = optimizer.step()

    assert variable.tensor.item() > before.item()
    assert info.iterations.item() == 1
    assert set(info.__dataclass_fields__) == {"status", "iterations", "cost"}


def test_batched_elements_converge_independently() -> None:
    target = torch.tensor([[1.0, -2.0], [0.5, 0.25], [-1.0, 3.0]])
    variable, problem = _difference_problem(torch.zeros_like(target), target, batch_ndim=1)

    info = LevenbergMarquardt(problem, max_iterations=20).optimize()

    torch.testing.assert_close(variable.tensor, target, rtol=2e-5, atol=2e-5)
    assert info.status.shape == (3,)
    assert bool(info.converged.all())


def test_projected_bound_solution_reports_stalled_at_bounds() -> None:
    bounds = Bounds(torch.tensor([-0.5]), torch.tensor([0.5]))
    variable, problem = _difference_problem(torch.tensor([0.0]), torch.tensor([2.0]), bounds=bounds)

    info = LevenbergMarquardt(problem, max_iterations=20).optimize()

    torch.testing.assert_close(variable.tensor, torch.tensor([0.5]))
    assert info.status.item() == OptimizerStatus.STALLED_AT_BOUNDS
    assert bool(info.converged)


def test_problem_update_refreshes_status_and_solves_new_batch() -> None:
    variable, problem = _difference_problem(torch.tensor([0.0]), torch.tensor([1.0]))
    optimizer = LevenbergMarquardt(problem, max_iterations=20)
    optimizer.optimize()

    problem.update({"x": torch.zeros(4, 1)})
    info = optimizer.optimize()

    assert variable.tensor.shape == (4, 1)
    torch.testing.assert_close(variable.tensor, torch.ones(4, 1), rtol=2e-5, atol=2e-5)
    assert bool(info.converged.all())
