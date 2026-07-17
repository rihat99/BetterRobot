"""Per-variable physical tangent step limits for named-block LM."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pytest
import torch

from better_robot.optim import LevenbergMarquardt, Problem, ResidualItem, VarSpec


class _TargetResidual:
    name = "target"
    reads = ("x", "target")
    dim = 1

    def __call__(self, ctx: Mapping[str, Any]) -> torch.Tensor:
        return ctx["x"] - ctx["target"]

    def jacobian_blocks(self, ctx: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        return {"x": torch.ones_like(ctx["x"]).unsqueeze(-1)}


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


def _problem() -> Problem:
    return Problem(
        vars=(VarSpec("x", (1,)),),
        residuals=(ResidualItem("target", _TargetResidual()),),
        parameters={"target": torch.tensor([10.0], dtype=torch.float64)},
    )


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
        LevenbergMarquardt(block_step_limits=limits)  # type: ignore[arg-type]


@pytest.mark.parametrize("limit", (0.0, -1.0, float("inf"), float("nan"), True, torch.tensor(1.0)))
def test_block_step_limit_norm_must_be_a_positive_finite_python_number(limit: object) -> None:
    with pytest.raises(ValueError, match="finite positive Python number"):
        LevenbergMarquardt(block_step_limits=(("x", limit),))  # type: ignore[arg-type]


def test_block_step_limit_names_are_unique_and_belong_to_problem() -> None:
    with pytest.raises(ValueError, match="duplicate.*x"):
        LevenbergMarquardt(block_step_limits=(("x", 1.0), ("x", 2.0)))

    solver = LevenbergMarquardt(block_step_limits=(("unknown", 1.0),))
    with pytest.raises(ValueError, match="unknown variable names.*unknown"):
        solver.init_state({"x": torch.zeros(1, dtype=torch.float64)}, _problem())

    fixed_problem = Problem(
        vars=(
            VarSpec("x", (1,)),
            VarSpec("fixed", (1,), mask=torch.tensor([False])),
        ),
        residuals=(ResidualItem("target", _TargetResidual()),),
        parameters={"target": torch.tensor([10.0], dtype=torch.float64)},
    )
    fixed_solver = LevenbergMarquardt(block_step_limits=(("fixed", 1.0),))
    fixed_values = {
        "x": torch.zeros(1, dtype=torch.float64),
        "fixed": torch.zeros(1, dtype=torch.float64),
    }
    with pytest.raises(ValueError, match="at least one free tangent.*fixed"):
        fixed_solver.init_state(fixed_values, fixed_problem)


def test_normal_equation_step_is_limited_before_retraction() -> None:
    problem = _problem()
    values = {"x": torch.zeros(1, dtype=torch.float64)}
    solver = LevenbergMarquardt(block_step_limits=(("x", 0.2),))

    next_values, next_state = solver.update(values, solver.init_state(values, problem), problem)

    torch.testing.assert_close(next_values["x"], torch.tensor([0.2], dtype=torch.float64))
    torch.testing.assert_close(next_state.step_norm, torch.tensor(0.2, dtype=torch.float64))


def test_projected_gradient_safeguard_step_is_limited_before_retraction() -> None:
    problem = _problem()
    values = {"x": torch.zeros(1, dtype=torch.float64)}
    solver = LevenbergMarquardt(
        linear_solver=_ZeroLinearSolver(),
        block_step_limits=(("x", 0.1),),
    )

    next_values, next_state = solver.update(values, solver.init_state(values, problem), problem)

    torch.testing.assert_close(next_values["x"], torch.tensor([0.1], dtype=torch.float64))
    torch.testing.assert_close(next_state.step_norm, torch.tensor(0.1, dtype=torch.float64))
