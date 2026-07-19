"""Per-variable physical tangent step limits for object-owned LM."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from better_robot.optim import LevenbergMarquardt, Problem, Residual, Variable


class _TargetResidual(Residual):
    def __init__(self, x: Variable, target: Variable, *extra: Variable) -> None:
        self.x = x
        self.target = target
        self.extra = extra
        super().__init__(x, target, *extra, dim=1, name="target")

    def error(self) -> torch.Tensor:
        anchor = sum((variable.tensor.sum() * 0.0 for variable in self.extra), self.x.tensor.new_zeros(()))
        return self.x.tensor - self.target.tensor + anchor


@dataclass(frozen=True)
class _ZeroLinearSolver:
    """Make the projected-gradient safeguard win over the zero LM proposal."""

    def solve(
        self,
        A: torch.Tensor,
        b: torch.Tensor,
        ridge: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        del A, ridge
        return torch.zeros_like(b)


def _problem(*extra: Variable) -> Problem:
    x = Variable(torch.zeros(1, dtype=torch.float64), name="x")
    target = Variable(torch.tensor([10.0], dtype=torch.float64), name="target", trainable=False)
    return Problem([_TargetResidual(x, target, *extra)])


@pytest.mark.parametrize(
    "limits",
    (
        pytest.param([("x", 1.0)], id="mutable-container"),
        pytest.param((("x", 1.0, 2.0),), id="malformed-entry"),
        pytest.param((("", 1.0),), id="empty-name"),
    ),
)
def test_block_step_limit_structure_is_immutable_and_well_formed(limits: object) -> None:
    with pytest.raises(TypeError, match="block_step_limits"):
        LevenbergMarquardt(_problem(), block_step_limits=limits)  # type: ignore[arg-type]


@pytest.mark.parametrize("limit", (0.0, -1.0, float("inf"), float("nan"), True, torch.tensor(1.0)))
def test_block_step_limit_norm_must_be_a_positive_finite_python_number(limit: object) -> None:
    with pytest.raises(ValueError, match="finite positive Python number"):
        LevenbergMarquardt(_problem(), block_step_limits=(("x", limit),))  # type: ignore[arg-type]


def test_block_step_limit_names_are_unique_and_belong_to_problem() -> None:
    with pytest.raises(ValueError, match="duplicate.*x"):
        LevenbergMarquardt(_problem(), block_step_limits=(("x", 1.0), ("x", 2.0)))

    unknown_problem = _problem()
    unknown = LevenbergMarquardt(unknown_problem, block_step_limits=(("unknown", 1.0),))
    with pytest.raises(ValueError, match="unknown variable names.*unknown"):
        unknown._init_state({"x": torch.zeros(1, dtype=torch.float64)}, unknown_problem)

    fixed = Variable(
        torch.zeros(1, dtype=torch.float64),
        name="fixed",
        mask=torch.tensor([False]),
    )
    fixed_problem = _problem(fixed)
    fixed_solver = LevenbergMarquardt(fixed_problem, block_step_limits=(("fixed", 1.0),))
    with pytest.raises(ValueError, match="at least one free tangent.*fixed"):
        fixed_solver._init_state({"x": torch.zeros(1, dtype=torch.float64), "fixed": fixed.tensor}, fixed_problem)


def test_normal_equation_step_is_limited_before_retraction() -> None:
    problem = _problem()
    values = {"x": torch.zeros(1, dtype=torch.float64)}
    optimizer = LevenbergMarquardt(problem, block_step_limits=(("x", 0.2),))

    next_values, next_state = optimizer._update(values, optimizer._init_state(values, problem), problem)

    torch.testing.assert_close(next_values["x"], torch.tensor([0.2], dtype=torch.float64))
    torch.testing.assert_close(next_state.step_norm, torch.tensor(0.2, dtype=torch.float64))


def test_projected_gradient_safeguard_step_is_limited_before_retraction() -> None:
    problem = _problem()
    values = {"x": torch.zeros(1, dtype=torch.float64)}
    optimizer = LevenbergMarquardt(
        problem,
        solver=_ZeroLinearSolver(),
        block_step_limits=(("x", 0.1),),
    )

    next_values, next_state = optimizer._update(values, optimizer._init_state(values, problem), problem)

    torch.testing.assert_close(next_values["x"], torch.tensor([0.1], dtype=torch.float64))
    torch.testing.assert_close(next_state.step_norm, torch.tensor(0.1, dtype=torch.float64))
