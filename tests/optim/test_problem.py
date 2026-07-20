"""Object-graph Problem contracts for the optimization API."""

from __future__ import annotations

import pytest
import torch

from better_robot.optim.problem import Problem
from better_robot.optim.variables import Variable
from better_robot.residuals.base import Residual, ScaleWeight, residual


class _TwoVariableResidual(Residual):
    def __init__(self, x: Variable, y: Variable, *, name: str = "pair") -> None:
        self.x = x
        self.y = y
        super().__init__(x, y, dim=2, name=name)

    def error(self) -> torch.Tensor:
        return self.x.tensor + 2.0 * self.y.tensor


class _AnalyticResidual(Residual):
    def __init__(self, x: Variable) -> None:
        self.x = x
        super().__init__(x, dim=2, weight=2.0, name="analytic")

    def error(self) -> torch.Tensor:
        return self.x.tensor.square()

    def jacobian(self) -> tuple[torch.Tensor, ...]:
        return (torch.diag_embed(2.0 * self.x.tensor),)


class _PublicHookVariable(Variable):
    def project(self, value: torch.Tensor) -> torch.Tensor:
        return value.clamp(max=4.0)

    def retract(self, delta: torch.Tensor) -> torch.Tensor:
        return super().retract(2.0 * delta)

    def difference(self, other: torch.Tensor) -> torch.Tensor:
        return (self.tensor - other) / 2.0


def test_problem_routes_all_geometry_through_public_variable_hooks() -> None:
    x = _PublicHookVariable(torch.tensor([1.0]), name="x")

    @residual(x, dim=1)
    def identity(value: torch.Tensor) -> torch.Tensor:
        return value

    problem = Problem([identity])
    original = x.tensor

    torch.testing.assert_close(problem.gradient()["x"], torch.tensor([2.0]))
    for strategy in ("jacrev", "jacfwd", "finite_difference"):
        torch.testing.assert_close(
            problem.dense_jacobian(strategy=strategy),
            torch.tensor([[2.0]]),
            rtol=3e-3,
            atol=3e-4,
        )
    torch.testing.assert_close(
        problem.retract({"x": torch.tensor([3.0])}, {"x": torch.tensor([1.0])})["x"],
        torch.tensor([4.0]),
    )
    torch.testing.assert_close(
        problem.difference({"x": torch.tensor([1.0])}, {"x": torch.tensor([5.0])})["x"],
        torch.tensor([2.0]),
    )
    assert x.tensor is original


@pytest.mark.parametrize("strategy", ("jacrev", "jacfwd", "finite_difference"))
def test_all_ad_strategies_match_analytic_blocks(strategy: str) -> None:
    x = Variable(torch.tensor([0.25, -0.5]), name="x")
    problem = Problem([_AnalyticResidual(x)])

    actual = problem.dense_jacobian(strategy=strategy)
    expected = 2.0 * torch.diag(2.0 * x.tensor)

    torch.testing.assert_close(actual, expected, rtol=3e-3, atol=3e-4)


def test_analytic_weight_is_applied_once_to_error_and_rows() -> None:
    x = Variable(torch.tensor([2.0, 3.0]), name="x")
    item = _AnalyticResidual(x)
    problem = Problem([item])

    torch.testing.assert_close(problem.error(), torch.tensor([8.0, 18.0]))
    torch.testing.assert_close(problem.dense_jacobian(), torch.diag(torch.tensor([8.0, 12.0])))


def test_update_validates_atomically_and_refreshes_current_values() -> None:
    x = Variable(torch.zeros(2), name="x")
    y = Variable(torch.zeros(2), name="y")
    problem = Problem([_TwoVariableResidual(x, y)])
    problem.error()
    original_x = x.tensor

    with pytest.raises(ValueError, match="share batch shape"):
        problem.update({"x": torch.zeros(3, 2)})

    assert x.tensor is original_x
    problem.update({"x": torch.ones(4, 2), "y": torch.full((4, 2), 2.0)})
    assert problem.error().shape == (4, 2)
    torch.testing.assert_close(problem.error(), torch.full((4, 2), 5.0))


def test_duplicate_names_are_rejected_when_problem_freezes() -> None:
    first = Variable(torch.zeros(1), name="same")
    second = Variable(torch.zeros(1), name="same")
    problem = Problem([_TwoVariableResidual(first, second)])

    with pytest.raises(ValueError, match="duplicate variable name 'same'"):
        problem.error()


def test_python_zero_weight_skips_residual_but_tensor_zero_evaluates() -> None:
    x = Variable(torch.ones(1), name="x")
    calls = 0

    @residual(x, dim=1, weight=0.0)
    def counted(value: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return value

    problem = Problem([counted])
    torch.testing.assert_close(problem.error(), torch.zeros(1))
    assert calls == 0

    counted.weight = ScaleWeight(torch.tensor(0.0))
    torch.testing.assert_close(problem.error(), torch.zeros(1))
    assert calls == 1


def test_trial_exception_restores_exact_original_tensor() -> None:
    x = Variable(torch.tensor([2.0]), name="x")
    original = x.tensor

    @residual(x, dim=1)
    def exploding(value: torch.Tensor) -> torch.Tensor:
        if value is not original:
            raise RuntimeError("candidate failed")
        return value

    problem = Problem([exploding])
    problem.error()
    with pytest.raises(RuntimeError, match="candidate failed"):
        problem.jacobian_blocks(strategy="finite_difference")

    assert x.tensor is original
    torch.testing.assert_close(problem.error(), original)


def test_problem_freezes_at_first_use() -> None:
    x = Variable(torch.ones(1), name="x")

    @residual(x, dim=1)
    def identity(value: torch.Tensor) -> torch.Tensor:
        return value

    problem = Problem([identity])
    assert not problem.frozen
    problem.error()
    assert problem.frozen
    with pytest.raises(RuntimeError, match="before the Problem is frozen"):
        problem.add_residual(identity)
