"""Object-owned Levenberg--Marquardt behavior contracts."""

from __future__ import annotations

import torch

from better_robot.optim import LevenbergMarquardt, Problem, Variable, residual


def _difference_problem(initial: torch.Tensor, target: torch.Tensor, *, bounds=None, batch_ndim: int = 0):
    variable = Variable(initial, name="x", bounds=bounds, batch_ndim=batch_ndim)

    @residual(variable, dim=initial.shape[-1])
    def difference(value: torch.Tensor) -> torch.Tensor:
        return value - target

    return variable, Problem([difference])


def test_step_updates_referenced_values_and_returns_minimal_info() -> None:
    variable, problem = _difference_problem(torch.tensor([0.0]), torch.tensor([2.0]))
    optimizer = LevenbergMarquardt(problem, max_iterations=10)

    before = variable.tensor.clone()
    info = optimizer.step()

    assert variable.tensor.item() > before.item()
    assert info.iterations.item() == 1
    assert set(info.__dataclass_fields__) == {"status", "iterations", "cost"}


def test_problem_update_refreshes_status_and_solves_new_batch() -> None:
    variable, problem = _difference_problem(torch.tensor([0.0]), torch.tensor([1.0]))
    optimizer = LevenbergMarquardt(problem, max_iterations=20)
    optimizer.optimize()

    problem.update({"x": torch.zeros(4, 1)})
    info = optimizer.optimize()

    assert variable.tensor.shape == (4, 1)
    torch.testing.assert_close(variable.tensor, torch.ones(4, 1), rtol=2e-5, atol=2e-5)
    assert bool(info.converged.all())
