"""Functional phase rebuilding, staging, and failure-isolation tests."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pytest
import torch

from better_robot.optim import (
    Adam,
    AdamState,
    LevenbergMarquardt,
    LMState,
    Phase,
    Problem,
    ResidualItem,
    VarSpec,
    run_phases,
)


class _LinearResidual:
    name = "linear"
    reads = ("x", "target")
    dim = 2

    def __call__(self, ctx: Mapping[str, Any]) -> torch.Tensor:
        return ctx["x"] - ctx["target"]

    def jacobian_blocks(self, ctx: Any) -> dict[str, torch.Tensor]:
        x = ctx["x"]
        identity = torch.eye(2, dtype=x.dtype, device=x.device)
        free = ctx.free_indices("x").to(device=x.device)
        block = identity.index_select(-1, free)
        return {"x": block.expand(*x.shape[:-1], 2, free.numel())}


@dataclass
class _CountingProvider:
    calls: int = 0
    name: str = "double_provider"
    inputs: tuple[str, ...] = ("x",)
    outputs: tuple[str, ...] = ("double_x",)

    def __call__(self, ctx: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        self.calls += 1
        return {"double_x": 2.0 * ctx["x"]}


class _LazyResidual:
    name = "lazy"
    reads = ("double_x",)
    dim = 2

    def __call__(self, ctx: Mapping[str, Any]) -> torch.Tensor:
        return ctx["double_x"]


@dataclass(frozen=True)
class _RaisingSolver:
    max_iter: int = 1

    def init_state(self, values, problem):
        del values, problem
        return torch.tensor(0)

    def update(self, values, state, problem):
        del problem
        return values, state

    def run(self, values, problem, state=None):
        del values, problem, state
        raise RuntimeError("phase boom")


def _problem(
    target: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    provider: _CountingProvider | None = None,
) -> Problem:
    residuals = [ResidualItem("linear", _LinearResidual())]
    providers = ()
    if provider is not None:
        residuals.append(ResidualItem("lazy", _LazyResidual()))
        providers = (provider,)
    return Problem(
        vars=(VarSpec("x", (2,), mask=mask),),
        residuals=tuple(residuals),
        providers=providers,
        parameters={"target": target},
    )


def test_phase_masks_are_functional_and_cannot_unfreeze_base_mask() -> None:
    base_mask = torch.tensor([True, False])
    problem = _problem(torch.tensor([1.0, 2.0]), mask=base_mask)
    values = {"x": torch.zeros(2)}
    solver = LevenbergMarquardt(gtol=1e-7)
    phases = (
        Phase(
            "none_free",
            iters=0,
            optimizer=solver,
            mask_overrides={"x": torch.tensor([False, True])},
        ),
        Phase(
            "try_unfreeze",
            iters=8,
            optimizer=solver,
            mask_overrides={"x": torch.tensor([True, True])},
        ),
    )

    result = run_phases(problem, values, phases)

    torch.testing.assert_close(result.values["x"], torch.tensor([1.0, 0.0]), atol=2e-5, rtol=2e-5)
    assert result.states[0] is None
    assert problem.vars[0].mask is base_mask
    torch.testing.assert_close(values["x"], torch.zeros(2), rtol=0.0, atol=0.0)


def test_zero_weight_skips_residual_and_its_lazy_provider() -> None:
    provider = _CountingProvider()
    problem = _problem(torch.tensor([0.5, -0.25]), provider=provider)
    phase = Phase(
        "without_lazy_term",
        iters=8,
        optimizer=LevenbergMarquardt(gtol=1e-7),
        weight_overrides={"lazy": 0.0},
    )

    result = run_phases(problem, {"x": torch.zeros(2)}, (phase,))

    torch.testing.assert_close(result.values["x"], torch.tensor([0.5, -0.25]), atol=2e-5, rtol=2e-5)
    assert provider.calls == 0
    assert problem.residuals[1].weight == 1.0


def test_original_problem_is_unchanged_when_phase_solver_raises() -> None:
    mask = torch.tensor([True, False])
    problem = _problem(torch.ones(2), mask=mask)
    original_vars = problem.vars
    original_residuals = problem.residuals
    phase = Phase(
        "raising",
        iters=1,
        optimizer=_RaisingSolver(),
        weight_overrides={"linear": 0.25},
        mask_overrides={"x": torch.tensor([False, True])},
    )

    with pytest.raises(RuntimeError, match="phase boom"):
        run_phases(problem, {"x": torch.zeros(2)}, (phase,))

    assert problem.vars == original_vars
    assert problem.residuals == original_residuals
    assert problem.vars[0].mask is mask
    assert problem.residuals[0].weight == 1.0


def test_on_start_runs_once_even_for_zero_iterations() -> None:
    calls: list[str] = []
    phase = Phase(
        "hook_only",
        iters=0,
        optimizer=LevenbergMarquardt(),
        on_start=lambda: calls.append("started"),
    )

    result = run_phases(_problem(torch.ones(2)), {"x": torch.zeros(2)}, (phase,))

    assert calls == ["started"]
    assert result.states == (None,)
    torch.testing.assert_close(result.values["x"], torch.zeros(2), rtol=0.0, atol=0.0)


def test_phase_override_names_and_mask_shapes_fail_clearly() -> None:
    problem = _problem(torch.ones(2))
    values = {"x": torch.zeros(2)}
    solver = LevenbergMarquardt()

    with pytest.raises(ValueError, match="unknown weight overrides"):
        run_phases(
            problem,
            values,
            (Phase("bad_weight", 0, solver, weight_overrides={"missing": 0.0}),),
        )
    with pytest.raises(ValueError, match="must have tangent shape"):
        run_phases(
            problem,
            values,
            (Phase("bad_mask", 0, solver, mask_overrides={"x": torch.ones(3)}),),
        )


def test_adam_mask_transition_uses_fresh_reduced_moments() -> None:
    problem = _problem(torch.tensor([0.8, -0.6]))
    initial = {"x": torch.zeros(2)}
    first = Phase(
        "first_coordinate",
        iters=3,
        optimizer=Adam(lr=0.08, tol=0.0),
        mask_overrides={"x": torch.tensor([True, False])},
    )
    second = Phase(
        "both_coordinates",
        iters=4,
        optimizer=Adam(lr=0.08, tol=0.0),
        mask_overrides={"x": torch.tensor([True, True])},
    )

    staged = run_phases(problem, initial, (first, second))
    first_only = run_phases(problem, initial, (first,))
    independently_refined = run_phases(problem, first_only.values, (second,))

    first_state, second_state = staged.states
    assert isinstance(first_state, AdamState)
    assert isinstance(second_state, AdamState)
    assert first_state.m["x"].shape == (1,)
    assert second_state.m["x"].shape == (2,)
    assert int(first_state.step) == first.iters
    assert int(second_state.step) == second.iters
    torch.testing.assert_close(staged.values, independently_refined.values)
    torch.testing.assert_close(second_state.m, independently_refined.states[0].m)
    torch.testing.assert_close(second_state.v, independently_refined.states[0].v)


def test_mixed_lm_then_adam_phases_share_problem_and_converge() -> None:
    target = torch.tensor([0.75, -0.4])
    problem = _problem(target)
    initial = {"x": torch.zeros(2)}
    result = run_phases(
        problem,
        initial,
        (
            Phase(
                "lm_seed",
                iters=1,
                optimizer=LevenbergMarquardt(gtol=0.0, xtol=0.0, ftol=0.0),
            ),
            Phase(
                "adam_refine",
                iters=80,
                optimizer=Adam(lr=0.02, tol=1e-7),
            ),
        ),
    )

    assert isinstance(result.states[0], LMState)
    assert isinstance(result.states[1], AdamState)
    assert torch.linalg.vector_norm(result.values["x"] - target) < 2e-4
    assert problem.residuals[0].weight == 1.0


def test_batched_phase_run_matches_sequential_elements() -> None:
    target = torch.tensor(
        [[0.7, -0.2], [-0.5, 0.8], [0.25, -0.9]],
        dtype=torch.float64,
    )
    phases = (
        Phase(
            "x_only",
            iters=5,
            optimizer=Adam(lr=0.06, tol=0.0),
            mask_overrides={"x": torch.tensor([True, False])},
        ),
        Phase(
            "all",
            iters=8,
            optimizer=Adam(lr=0.04, tol=0.0),
        ),
    )
    batched = run_phases(_problem(target), {"x": torch.zeros_like(target)}, phases)

    sequential = []
    for row in target:
        result = run_phases(_problem(row), {"x": torch.zeros_like(row)}, phases)
        sequential.append(result.values["x"])

    torch.testing.assert_close(
        batched.values["x"],
        torch.stack(sequential),
        rtol=2e-6,
        atol=2e-6,
    )
