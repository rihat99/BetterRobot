"""Madsen--Nielsen damping, scaling, and robust-gain regressions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pytest
import torch

from better_robot.optim.blocks import (
    Bounds,
    LMStatus,
    LevenbergMarquardt,
    Problem,
    ResidualItem,
    VarSpec,
)
from better_robot.optim.kernels import Huber, Tukey


class _ExponentialResidual:
    name = "exponential"
    reads = ("x", "target")
    dim = 1

    def __call__(self, ctx: Mapping[str, Any]) -> torch.Tensor:
        return ctx["x"].exp() - ctx["target"]

    def jacobian_blocks(self, ctx: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        return {"x": ctx["x"].exp().unsqueeze(-1)}


class _MixedUnitResidual:
    name = "mixed_unit"
    reads = ("meters", "millimeters", "target")
    dim = 2

    def __call__(self, ctx: Mapping[str, Any]) -> torch.Tensor:
        return torch.stack(
            (
                ctx["meters"][..., 0] - ctx["target"][..., 0],
                1_000.0 * (ctx["millimeters"][..., 0] - ctx["target"][..., 1]),
            ),
            dim=-1,
        )

    def jacobian_blocks(self, ctx: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        target = ctx["target"]
        zeros = torch.zeros_like(target[..., 0])
        ones = torch.ones_like(zeros)
        return {
            "meters": torch.stack((ones, zeros), dim=-1).unsqueeze(-1),
            "millimeters": torch.stack((zeros, 1_000.0 * ones), dim=-1).unsqueeze(-1),
        }


class _PointFitResidual:
    name = "points"
    reads = ("x", "points")
    dim = 6

    def __call__(self, ctx: Mapping[str, Any]) -> torch.Tensor:
        return (ctx["x"].unsqueeze(-2) - ctx["points"]).reshape(*ctx["x"].shape[:-1], 6)

    def jacobian_blocks(self, ctx: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        x = ctx["x"]
        identity = torch.eye(2, dtype=x.dtype, device=x.device)
        block = identity.expand(*x.shape[:-1], 3, 2, 2)
        return {"x": block.reshape(*x.shape[:-1], 6, 2)}


class _TukeyDirectionReversalResidual:
    """A smooth residual whose trial lowers raw L2 but raises grouped Tukey."""

    name = "tukey_reversal"
    reads = ("x",)
    dim = 4

    def __init__(self) -> None:
        self.seen: list[torch.Tensor] = []

    def __call__(self, ctx: Mapping[str, Any]) -> torch.Tensor:
        x = ctx["x"][..., 0]
        self.seen.append(x.detach().clone())
        return torch.stack(
            (
                10.0 - 230.0 * x.square(),
                torch.zeros_like(x),
                0.2 + x + 20.0 * x.square(),
                torch.zeros_like(x),
            ),
            dim=-1,
        )

    def jacobian_blocks(self, ctx: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        x = ctx["x"][..., 0]
        derivative = torch.stack(
            (
                -460.0 * x,
                torch.zeros_like(x),
                1.0 + 40.0 * x,
                torch.zeros_like(x),
            ),
            dim=-1,
        )
        return {"x": derivative.unsqueeze(-1)}


class _OffsetPairResidual:
    reads = ("x",)
    dim = 2

    def __init__(self, name: str, target: float) -> None:
        self.name = name
        self.target = target

    def __call__(self, ctx: Mapping[str, Any]) -> torch.Tensor:
        x = ctx["x"][..., 0]
        return torch.stack((x - self.target, torch.zeros_like(x)), dim=-1)

    def jacobian_blocks(self, ctx: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        x = ctx["x"][..., 0]
        return {"x": torch.stack((torch.ones_like(x), torch.zeros_like(x)), dim=-1).unsqueeze(-1)}


class _LinearResidual:
    name = "linear"
    reads = ("x", "target")
    dim = 1

    def __call__(self, ctx: Mapping[str, Any]) -> torch.Tensor:
        return ctx["x"] - ctx["target"]

    def jacobian_blocks(self, ctx: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        return {"x": torch.ones_like(ctx["x"]).unsqueeze(-1)}


@dataclass(frozen=True)
class _ThresholdSolver:
    """Return a finite step only once LM has attempted the damping cap."""

    minimum_diagonal: float

    def solve(
        self,
        A: torch.Tensor,
        b: torch.Tensor,
        ridge: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        del ridge
        solved = torch.linalg.solve(A, b)
        ok = A.diagonal(dim1=-2, dim2=-1).amin(dim=-1) >= self.minimum_diagonal
        return torch.where(ok.unsqueeze(-1), solved, torch.full_like(solved, torch.nan))


def _grouped_cost(residual: torch.Tensor, kernel: object, group_size: int) -> torch.Tensor:
    groups = residual.reshape(-1, group_size)
    return kernel.rho(groups.square().sum(dim=-1)).sum()


def test_madsen_nielsen_damping_is_per_element_and_uses_exact_multipliers() -> None:
    target = torch.tensor([[1.1], [2.25], [10.0]], dtype=torch.float64)
    problem = Problem(
        vars=(VarSpec("x", (1,)),),
        residuals=(ResidualItem("exponential", _ExponentialResidual()),),
        parameters={"target": target},
    )
    solver = LevenbergMarquardt(max_iter=3)
    values = {"x": torch.zeros_like(target)}
    state = solver.init_state(values, problem)

    values_next, state_next = solver.update(values, state, problem)

    assert state_next.gain_ratio[0] > 0.9
    assert 0.0 < state_next.gain_ratio[1] < 0.1
    assert state_next.gain_ratio[2] < 0.0
    accepted_multiplier = torch.maximum(
        torch.full_like(state_next.gain_ratio[:2], 1.0 / 3.0),
        1.0 - (2.0 * state_next.gain_ratio[:2] - 1.0).pow(3),
    )
    torch.testing.assert_close(
        state_next.mu[:2],
        state.mu[:2] * accepted_multiplier,
        rtol=1e-14,
        atol=0.0,
    )
    torch.testing.assert_close(state_next.increase_factor[:2], torch.full((2,), 2.0, dtype=torch.float64))
    torch.testing.assert_close(state_next.mu[2], state.mu[2] * 2.0, rtol=0.0, atol=0.0)
    torch.testing.assert_close(state_next.increase_factor[2], torch.tensor(4.0, dtype=torch.float64))
    assert torch.all(values_next["x"][:2] != values["x"][:2])
    assert torch.equal(values_next["x"][2], values["x"][2])

    _, state_twice = solver.update(values_next, state_next, problem)

    torch.testing.assert_close(state_twice.mu[2], state_next.mu[2] * 4.0, rtol=0.0, atol=0.0)
    torch.testing.assert_close(state_twice.increase_factor[2], torch.tensor(8.0, dtype=torch.float64))


def test_rejected_damping_and_increase_factor_clamp_independently() -> None:
    target = torch.tensor([[1.1], [10.0]], dtype=torch.float64)
    problem = Problem(
        vars=(VarSpec("x", (1,)),),
        residuals=(ResidualItem("exponential", _ExponentialResidual()),),
        parameters={"target": target},
    )
    solver = LevenbergMarquardt(mu_max=16.0, increase_factor_max=32.0)
    values = {"x": torch.zeros_like(target)}
    state = solver.init_state(values, problem)._replace(
        mu=torch.tensor([1e-4, 8.0], dtype=torch.float64),
        increase_factor=torch.tensor([2.0, 32.0], dtype=torch.float64),
    )

    _, state_next = solver.update(values, state, problem)

    assert state_next.gain_ratio[0] > 0.0
    assert state_next.gain_ratio[1] < 0.0
    assert state_next.mu[0] < state.mu[0]
    torch.testing.assert_close(state_next.mu[1], torch.tensor(16.0, dtype=torch.float64))
    torch.testing.assert_close(state_next.increase_factor[1], torch.tensor(32.0, dtype=torch.float64))


def test_zero_mu_min_is_rejected_instead_of_deadlocking_factorization_retries() -> None:
    with pytest.raises(ValueError, match="0 < mu_min <= mu_max"):
        LevenbergMarquardt(mu_min=0.0)


def test_factorization_gets_one_attempt_at_mu_cap_before_failure() -> None:
    dtype = torch.float64
    problem = _linear_problem(torch.ones(1, dtype=dtype))
    solver = LevenbergMarquardt(
        mu_max=10.0,
        linear_solver=_ThresholdSolver(minimum_diagonal=11.0),
    )
    values = {"x": torch.zeros(1, dtype=dtype)}
    state = solver.init_state(values, problem)._replace(
        mu=torch.tensor(5.0, dtype=dtype),
        increase_factor=torch.tensor(2.0, dtype=dtype),
    )

    unchanged, at_cap = solver.update(values, state, problem)

    torch.testing.assert_close(unchanged["x"], values["x"], rtol=0.0, atol=0.0)
    torch.testing.assert_close(at_cap.mu, torch.tensor(10.0, dtype=dtype))
    assert LMStatus(int(at_cap.status)) is LMStatus.RUNNING

    moved, recovered = solver.update(unchanged, at_cap, problem)

    assert moved["x"].item() > 0.0
    assert bool(recovered.factorization_ok)
    assert LMStatus(int(recovered.status)) is not LMStatus.FAILED


@pytest.mark.parametrize(
    ("xtol", "ftol"),
    [(2.0, 0.0), (0.0, 2.0)],
)
def test_accepted_step_or_relative_decrease_tolerance_terminates(
    xtol: float,
    ftol: float,
) -> None:
    dtype = torch.float64
    problem = Problem(
        vars=(VarSpec("x", (1,)),),
        residuals=(ResidualItem("exponential", _ExponentialResidual()),),
        parameters={"target": torch.tensor([2.0], dtype=dtype)},
    )
    solver = LevenbergMarquardt(max_iter=1, gtol=0.0, xtol=xtol, ftol=ftol)

    values, state = solver.run({"x": torch.zeros(1, dtype=dtype)}, problem)

    assert values["x"].item() > 0.0
    assert state.step_norm > 0.0
    assert state.relative_decrease > 0.0
    assert state.grad_norm > 1.0  # This is a numerical stop, not a KKT stop.
    assert LMStatus(int(state.status)) is LMStatus.CONVERGED
    assert bool(state.converged)
    assert not bool(state.implicit_valid)


def test_bounded_problem_never_reports_tolerance_only_convergence() -> None:
    dtype = torch.float64
    problem = Problem(
        vars=(
            VarSpec(
                "x",
                (1,),
                bounds=Bounds(
                    lower=torch.tensor([-10.0], dtype=dtype),
                    upper=torch.tensor([10.0], dtype=dtype),
                ),
            ),
        ),
        residuals=(ResidualItem("exponential", _ExponentialResidual()),),
        parameters={"target": torch.tensor([2.0], dtype=dtype)},
    )
    solver = LevenbergMarquardt(max_iter=1, gtol=1e-8, xtol=2.0, ftol=2.0)

    _values, state = solver.run({"x": torch.zeros(1, dtype=dtype)}, problem)

    assert state.projected_grad_norm > solver.gtol
    assert LMStatus(int(state.status)) is LMStatus.MAXITER
    assert not bool(state.converged)
    assert not bool(state.implicit_valid)


def test_rejected_zero_step_does_not_trigger_step_or_decrease_tolerance() -> None:
    dtype = torch.float64
    problem = Problem(
        vars=(VarSpec("x", (1,)),),
        residuals=(ResidualItem("exponential", _ExponentialResidual()),),
        parameters={"target": torch.tensor([10.0], dtype=dtype)},
    )
    solver = LevenbergMarquardt(gtol=0.0, xtol=100.0, ftol=100.0)
    values = {"x": torch.zeros(1, dtype=dtype)}
    state = solver.init_state(values, problem)

    next_values, next_state = solver.update(values, state, problem)

    torch.testing.assert_close(next_values["x"], values["x"], rtol=0.0, atol=0.0)
    torch.testing.assert_close(next_state.step_norm, torch.tensor(0.0, dtype=dtype))
    torch.testing.assert_close(next_state.relative_decrease, torch.tensor(0.0, dtype=dtype))
    assert LMStatus(int(next_state.status)) is LMStatus.RUNNING


def test_finalize_does_not_preserve_tolerance_status_for_changed_values() -> None:
    dtype = torch.float64
    problem = Problem(
        vars=(VarSpec("x", (1,)),),
        residuals=(ResidualItem("exponential", _ExponentialResidual()),),
        parameters={"target": torch.tensor([2.0], dtype=dtype)},
    )
    solver = LevenbergMarquardt(gtol=0.0, xtol=2.0, ftol=0.0)
    initial = {"x": torch.zeros(1, dtype=dtype)}
    state = solver.init_state(initial, problem)
    _accepted_values, tolerance_state = solver.update(initial, state, problem)
    assert LMStatus(int(tolerance_state.status)) is LMStatus.CONVERGED

    _, refreshed = solver.finalize(initial, tolerance_state, problem)

    assert LMStatus(int(refreshed.status)) is LMStatus.RUNNING
    assert not bool(refreshed.converged)
    assert not bool(refreshed.implicit_valid)


def _mixed_unit_problem(*, scaled: bool, dtype: torch.dtype) -> Problem:
    return Problem(
        vars=(
            VarSpec(
                "meters",
                (1,),
                scale=torch.tensor([1.0], dtype=dtype) if scaled else None,
            ),
            VarSpec(
                "millimeters",
                (1,),
                scale=torch.tensor([1e-3], dtype=dtype) if scaled else None,
            ),
        ),
        residuals=(ResidualItem("mixed_unit", _MixedUnitResidual()),),
        parameters={"target": torch.ones(2, dtype=dtype)},
    )


def test_varspec_scale_sets_scaled_mu_and_balances_mixed_unit_step() -> None:
    dtype = torch.float64
    values = {
        "meters": torch.zeros(1, dtype=dtype),
        "millimeters": torch.zeros(1, dtype=dtype),
    }
    solver = LevenbergMarquardt(max_iter=1)
    unscaled_problem = _mixed_unit_problem(scaled=False, dtype=dtype)
    scaled_problem = _mixed_unit_problem(scaled=True, dtype=dtype)
    unscaled_state = solver.init_state(values, unscaled_problem)
    scaled_state = solver.init_state(values, scaled_problem)

    torch.testing.assert_close(unscaled_state.scale, torch.ones(2, dtype=dtype))
    torch.testing.assert_close(scaled_state.scale, torch.tensor([1.0, 1e-3], dtype=dtype))
    torch.testing.assert_close(unscaled_state.mu, torch.tensor(100.0, dtype=dtype))
    torch.testing.assert_close(scaled_state.mu, torch.tensor(1e-4, dtype=dtype))

    unscaled_values, unscaled_next = solver.update(values, unscaled_state, unscaled_problem)
    scaled_values, scaled_next = solver.update(values, scaled_state, scaled_problem)

    assert unscaled_values["meters"][0] < 0.02
    assert scaled_values["meters"][0] > 0.99
    assert scaled_values["millimeters"][0] > 0.99
    assert scaled_next.cost < unscaled_next.cost * 0.02


def test_grouped_huber_gain_accepts_when_raw_l2_increases() -> None:
    dtype = torch.float64
    points = torch.tensor([[0.0, 0.0], [0.0, 0.0], [100.0, 0.0]], dtype=dtype)
    start = points.mean(dim=0)
    kernel = Huber(delta=1.0)
    problem = Problem(
        vars=(VarSpec("x", (2,)),),
        residuals=(ResidualItem("points", _PointFitResidual(), group_size=2),),
        parameters={"points": points},
    )
    solver = LevenbergMarquardt(kernel=kernel)
    values = {"x": start.clone()}
    state = solver.init_state(values, problem)
    current_raw = 0.5 * state.residual.square().sum()
    current_robust = _grouped_cost(state.residual, kernel, 2)
    expected_group_weight = kernel.weight(state.residual.reshape(3, 2).square().sum(dim=-1))

    values_next, state_next = solver.update(values, state, problem)

    candidate_raw = 0.5 * state_next.residual.square().sum()
    candidate_robust = _grouped_cost(state_next.residual, kernel, 2)
    torch.testing.assert_close(
        state.robust_weights,
        expected_group_weight.repeat_interleave(2),
    )
    torch.testing.assert_close(state.cost, current_robust)
    torch.testing.assert_close(state_next.cost, candidate_robust)
    assert state_next.gain_ratio > 0.0
    assert not torch.equal(values_next["x"], values["x"])
    assert candidate_raw > current_raw
    assert candidate_robust < current_robust


def test_grouped_tukey_gain_rejects_when_raw_l2_decreases() -> None:
    dtype = torch.float64
    residual = _TukeyDirectionReversalResidual()
    kernel = Tukey(c=1.0)
    problem = Problem(
        vars=(VarSpec("x", (1,)),),
        residuals=(ResidualItem("tukey_reversal", residual, group_size=2),),
    )
    solver = LevenbergMarquardt(kernel=kernel)
    values = {"x": torch.zeros(1, dtype=dtype)}
    state = solver.init_state(values, problem)
    current_raw = 0.5 * state.residual.square().sum()
    current_robust = _grouped_cost(state.residual, kernel, 2)
    residual.seen.clear()

    values_next, state_next = solver.update(values, state, problem)

    trial_x = residual.seen[-1]
    trial_residual = residual({"x": trial_x.unsqueeze(-1)})
    trial_raw = 0.5 * trial_residual.square().sum()
    trial_robust = _grouped_cost(trial_residual, kernel, 2)
    assert not torch.equal(trial_x, values["x"][..., 0])
    assert trial_raw < current_raw
    assert trial_robust > current_robust
    assert state_next.gain_ratio < 0.0
    torch.testing.assert_close(values_next["x"], values["x"], rtol=0.0, atol=0.0)
    torch.testing.assert_close(state_next.cost, state.cost, rtol=0.0, atol=0.0)


def test_item_kernel_overrides_solver_default_per_semantic_group() -> None:
    dtype = torch.float64
    override_kernel = Huber(delta=1.0)
    default_kernel = Tukey(c=1.0)
    problem = Problem(
        vars=(VarSpec("x", (1,)),),
        residuals=(
            ResidualItem(
                "override",
                _OffsetPairResidual("override", 2.0),
                kernel=override_kernel,
                group_size=2,
            ),
            ResidualItem(
                "default",
                _OffsetPairResidual("default", 2.0),
                group_size=2,
            ),
        ),
    )
    solver = LevenbergMarquardt(kernel=default_kernel)
    values = {"x": torch.zeros(1, dtype=dtype)}

    state = solver.init_state(values, problem)

    squared_norm = torch.tensor(4.0, dtype=dtype)
    expected_cost = override_kernel.rho(squared_norm) + default_kernel.rho(squared_norm)
    expected_weights = torch.tensor([0.5, 0.5, 0.0, 0.0], dtype=dtype)
    torch.testing.assert_close(state.cost, expected_cost)
    torch.testing.assert_close(state.robust_weights, expected_weights)

    _, state_next = solver.update(values, state, problem)

    assert state_next.gain_ratio > 0.0
    assert state_next.cost < state.cost
    assert torch.all(state_next.robust_weights > 0.99)


def _linear_problem(target: torch.Tensor) -> Problem:
    return Problem(
        vars=(VarSpec("x", (1,)),),
        residuals=(ResidualItem("linear", _LinearResidual()),),
        parameters={"target": target},
    )


def test_warm_start_retains_damping_but_refreshes_changed_target_artifacts() -> None:
    dtype = torch.float64
    solver = LevenbergMarquardt(max_iter=0)
    values = {"x": torch.zeros(1, dtype=dtype)}
    _, old_state = solver.run(values, _linear_problem(torch.ones(1, dtype=dtype)))
    old_state = old_state._replace(
        mu=torch.tensor(0.123, dtype=dtype),
        increase_factor=torch.tensor(8.0, dtype=dtype),
    )

    _, refreshed = solver.run(
        values,
        _linear_problem(torch.full((1,), 2.0, dtype=dtype)),
        state=old_state,
    )

    torch.testing.assert_close(refreshed.mu, old_state.mu, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        refreshed.increase_factor,
        old_state.increase_factor,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(refreshed.residual, torch.tensor([-2.0], dtype=dtype))
    torch.testing.assert_close(refreshed.cost, torch.tensor(2.0, dtype=dtype))
    torch.testing.assert_close(refreshed.iterations, torch.tensor(0, dtype=torch.int64))


def test_warm_start_rejects_damping_dtype_mismatch_at_the_boundary() -> None:
    solver = LevenbergMarquardt(max_iter=0)
    old_values = {"x": torch.zeros(1, dtype=torch.float64)}
    _, old_state = solver.run(
        old_values,
        _linear_problem(torch.ones(1, dtype=torch.float64)),
    )
    new_values = {"x": torch.zeros(1, dtype=torch.float32)}

    with pytest.raises(ValueError, match="warm-start damping tensors.*dtype and device"):
        solver.run(
            new_values,
            _linear_problem(torch.ones(1, dtype=torch.float32)),
            state=old_state,
        )
