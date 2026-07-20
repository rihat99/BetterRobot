"""Madsen--Nielsen damping and robust-gain regressions."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from better_robot.optim import (
    Bounds,
    LevenbergMarquardt,
    OptimizerStatus,
    Problem,
    Residual,
    Variable,
)
from better_robot.optim.kernels import Huber, Tukey


class _ExponentialResidual(Residual):
    def __init__(self, x: Variable, target: Variable) -> None:
        self.x = x
        self.target = target
        super().__init__(x, target, dim=1, name="exponential")

    def error(self) -> torch.Tensor:
        return self.x.tensor.exp() - self.target.tensor

    def jacobian(self) -> tuple[torch.Tensor, ...]:
        return (self.x.tensor.exp().unsqueeze(-1),)


class _PointFitResidual(Residual):
    def __init__(self, x: Variable, points: Variable, *, kernel: Huber) -> None:
        self.x = x
        self.points = points
        super().__init__(x, points, dim=6, kernel=kernel, group_size=2, name="points")

    def error(self) -> torch.Tensor:
        return (self.x.tensor.unsqueeze(-2) - self.points.tensor).reshape(*self.x.tensor.shape[:-1], 6)

    def jacobian(self) -> tuple[torch.Tensor, ...]:
        x = self.x.tensor
        identity = torch.eye(2, dtype=x.dtype, device=x.device)
        block = identity.expand(*x.shape[:-1], 3, 2, 2)
        return (block.reshape(*x.shape[:-1], 6, 2),)


class _TukeyDirectionReversalResidual(Residual):
    """A smooth residual whose trial lowers raw L2 but raises grouped Tukey."""

    def __init__(self, x: Variable, *, kernel: Tukey) -> None:
        self.x = x
        self.seen: list[torch.Tensor] = []
        super().__init__(x, dim=4, kernel=kernel, group_size=2, name="tukey_reversal")

    def evaluate(self, value: torch.Tensor) -> torch.Tensor:
        x = value[..., 0]
        return torch.stack(
            (
                10.0 - 230.0 * x.square(),
                torch.zeros_like(x),
                0.2 + x + 20.0 * x.square(),
                torch.zeros_like(x),
            ),
            dim=-1,
        )

    def error(self) -> torch.Tensor:
        self.seen.append(self.x.tensor[..., 0].detach().clone())
        return self.evaluate(self.x.tensor)

    def jacobian(self) -> tuple[torch.Tensor, ...]:
        x = self.x.tensor[..., 0]
        derivative = torch.stack(
            (
                -460.0 * x,
                torch.zeros_like(x),
                1.0 + 40.0 * x,
                torch.zeros_like(x),
            ),
            dim=-1,
        )
        return (derivative.unsqueeze(-1),)


class _OffsetPairResidual(Residual):
    def __init__(self, x: Variable, name: str, target: float, *, kernel: object) -> None:
        self.x = x
        self.target = target
        super().__init__(x, dim=2, kernel=kernel, group_size=2, name=name)

    def error(self) -> torch.Tensor:
        x = self.x.tensor[..., 0]
        return torch.stack((x - self.target, torch.zeros_like(x)), dim=-1)

    def jacobian(self) -> tuple[torch.Tensor, ...]:
        x = self.x.tensor[..., 0]
        block = torch.stack((torch.ones_like(x), torch.zeros_like(x)), dim=-1).unsqueeze(-1)
        return (block,)


class _LinearResidual(Residual):
    def __init__(self, x: Variable, target: Variable) -> None:
        self.x = x
        self.target = target
        super().__init__(x, target, dim=1, name="linear")

    def error(self) -> torch.Tensor:
        return self.x.tensor - self.target.tensor

    def jacobian(self) -> tuple[torch.Tensor, ...]:
        return (torch.ones_like(self.x.tensor).unsqueeze(-1),)


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


def _exponential_problem(
    target_tensor: torch.Tensor,
    *,
    initial: torch.Tensor | None = None,
    bounds: Bounds | None = None,
) -> tuple[Variable, Variable, Problem]:
    value = torch.zeros_like(target_tensor) if initial is None else initial
    batch_ndim = max(value.ndim - 1, 0)
    x = Variable(value, name="x", bounds=bounds, batch_ndim=batch_ndim)
    target = Variable(
        target_tensor,
        name="target",
        trainable=False,
        batch_ndim=max(target_tensor.ndim - 1, 0),
    )
    return x, target, Problem([_ExponentialResidual(x, target)])


def _linear_problem(target_tensor: torch.Tensor, *, initial: torch.Tensor | None = None):
    value = torch.zeros_like(target_tensor) if initial is None else initial
    x = Variable(value, name="x", batch_ndim=max(value.ndim - 1, 0))
    target = Variable(
        target_tensor,
        name="target",
        trainable=False,
        batch_ndim=max(target_tensor.ndim - 1, 0),
    )
    return x, target, Problem([_LinearResidual(x, target)])


def test_madsen_nielsen_damping_is_per_element_and_uses_exact_multipliers() -> None:
    target_tensor = torch.tensor([[1.1], [2.25], [10.0]], dtype=torch.float64)
    x, _target, problem = _exponential_problem(target_tensor)
    optimizer = LevenbergMarquardt(problem, max_iterations=3)
    values = {"x": x.tensor}
    state = optimizer._init_state(values, problem)

    values_next, state_next = optimizer._update(values, state, problem)

    assert state_next.gain_ratio[0] > 0.9
    assert 0.0 < state_next.gain_ratio[1] < 0.1
    assert state_next.gain_ratio[2] < 0.0
    accepted_multiplier = torch.maximum(
        torch.full_like(state_next.gain_ratio[:2], 1.0 / 3.0),
        1.0 - (2.0 * state_next.gain_ratio[:2] - 1.0).pow(3),
    )
    torch.testing.assert_close(state_next.mu[:2], state.mu[:2] * accepted_multiplier, rtol=1e-14, atol=0.0)
    torch.testing.assert_close(state_next.increase_factor[:2], torch.full((2,), 2.0, dtype=torch.float64))
    torch.testing.assert_close(state_next.mu[2], state.mu[2] * 2.0, rtol=0.0, atol=0.0)
    torch.testing.assert_close(state_next.increase_factor[2], torch.tensor(4.0, dtype=torch.float64))
    assert torch.all(values_next["x"][:2] != values["x"][:2])
    assert torch.equal(values_next["x"][2], values["x"][2])

    _, state_twice = optimizer._update(values_next, state_next, problem)
    torch.testing.assert_close(state_twice.mu[2], state_next.mu[2] * 4.0, rtol=0.0, atol=0.0)
    torch.testing.assert_close(state_twice.increase_factor[2], torch.tensor(8.0, dtype=torch.float64))


def test_rejected_damping_and_increase_factor_clamp_independently() -> None:
    target_tensor = torch.tensor([[1.1], [50.0]], dtype=torch.float64)
    x, _target, problem = _exponential_problem(target_tensor)
    optimizer = LevenbergMarquardt(problem, mu_max=16.0, increase_factor_max=32.0)
    values = {"x": x.tensor}
    state = optimizer._init_state(values, problem)._replace(
        mu=torch.tensor([1e-4, 8.0], dtype=torch.float64),
        increase_factor=torch.tensor([2.0, 32.0], dtype=torch.float64),
    )

    _, state_next = optimizer._update(values, state, problem)

    assert state_next.gain_ratio[0] > 0.0
    assert state_next.gain_ratio[1] < 0.0
    assert state_next.mu[0] < state.mu[0]
    torch.testing.assert_close(state_next.mu[1], torch.tensor(16.0, dtype=torch.float64))
    torch.testing.assert_close(state_next.increase_factor[1], torch.tensor(32.0, dtype=torch.float64))


def test_zero_mu_min_is_rejected_instead_of_deadlocking_factorization_retries() -> None:
    _x, _target, problem = _linear_problem(torch.ones(1))
    with pytest.raises(ValueError, match="0 < mu_min <= mu_max"):
        LevenbergMarquardt(problem, mu_min=0.0)


def test_factorization_gets_one_attempt_at_mu_cap_before_failure() -> None:
    dtype = torch.float64
    x, _target, problem = _linear_problem(torch.ones(1, dtype=dtype))
    optimizer = LevenbergMarquardt(
        problem,
        mu_max=10.0,
        solver=_ThresholdSolver(minimum_diagonal=11.0),
    )
    values = {"x": x.tensor}
    state = optimizer._init_state(values, problem)._replace(
        mu=torch.tensor(5.0, dtype=dtype),
        increase_factor=torch.tensor(2.0, dtype=dtype),
    )

    unchanged, at_cap = optimizer._update(values, state, problem)

    torch.testing.assert_close(unchanged["x"], values["x"], rtol=0.0, atol=0.0)
    torch.testing.assert_close(at_cap.mu, torch.tensor(10.0, dtype=dtype))
    assert OptimizerStatus(int(at_cap.status)) is OptimizerStatus.RUNNING

    moved, recovered = optimizer._update(unchanged, at_cap, problem)
    assert moved["x"].item() > 0.0
    assert bool(recovered.factorization_ok)
    assert OptimizerStatus(int(recovered.status)) is not OptimizerStatus.FAILED


@pytest.mark.parametrize(("xtol", "ftol"), [(2.0, 0.0), (0.0, 2.0)])
def test_accepted_step_or_relative_decrease_tolerance_terminates(xtol: float, ftol: float) -> None:
    dtype = torch.float64
    x, _target, problem = _exponential_problem(torch.tensor([2.0], dtype=dtype))
    optimizer = LevenbergMarquardt(
        problem,
        max_iterations=1,
        tolerance=0.0,
        step_tolerance=xtol,
        relative_tolerance=ftol,
    )

    info = optimizer.optimize()
    assert optimizer._state is not None
    state = optimizer._state

    assert x.tensor.item() > 0.0
    assert state.step_norm > 0.0
    assert state.relative_decrease > 0.0
    assert state.grad_norm > 1.0
    assert OptimizerStatus(int(info.status)) is OptimizerStatus.CONVERGED
    assert bool(info.converged)
    assert not bool(state.implicit_valid)


def test_bounded_problem_never_reports_tolerance_only_convergence() -> None:
    dtype = torch.float64
    bounds = Bounds(
        lower=torch.tensor([-10.0], dtype=dtype),
        upper=torch.tensor([10.0], dtype=dtype),
    )
    _x, _target, problem = _exponential_problem(torch.tensor([2.0], dtype=dtype), bounds=bounds)
    optimizer = LevenbergMarquardt(
        problem,
        max_iterations=1,
        tolerance=1e-8,
        step_tolerance=2.0,
        relative_tolerance=2.0,
    )

    info = optimizer.optimize()
    assert optimizer._state is not None
    state = optimizer._state

    assert state.projected_grad_norm > optimizer.gtol
    assert OptimizerStatus(int(info.status)) is OptimizerStatus.MAXITER
    assert not bool(info.converged)
    assert not bool(state.implicit_valid)


def test_rejected_zero_step_does_not_trigger_step_or_decrease_tolerance() -> None:
    dtype = torch.float64
    x, _target, problem = _exponential_problem(torch.tensor([10.0], dtype=dtype))
    optimizer = LevenbergMarquardt(
        problem,
        tolerance=0.0,
        step_tolerance=100.0,
        relative_tolerance=100.0,
    )
    values = {"x": x.tensor}
    state = optimizer._init_state(values, problem)

    next_values, next_state = optimizer._update(values, state, problem)

    torch.testing.assert_close(next_values["x"], values["x"], rtol=0.0, atol=0.0)
    torch.testing.assert_close(next_state.step_norm, torch.tensor(0.0, dtype=dtype))
    torch.testing.assert_close(next_state.relative_decrease, torch.tensor(0.0, dtype=dtype))
    assert OptimizerStatus(int(next_state.status)) is OptimizerStatus.RUNNING


def test_finalize_does_not_preserve_tolerance_status_for_changed_values() -> None:
    dtype = torch.float64
    x, _target, problem = _exponential_problem(torch.tensor([2.0], dtype=dtype))
    optimizer = LevenbergMarquardt(problem, tolerance=0.0, step_tolerance=2.0, relative_tolerance=0.0)
    initial = {"x": x.tensor}
    state = optimizer._init_state(initial, problem)
    _accepted_values, tolerance_state = optimizer._update(initial, state, problem)
    assert OptimizerStatus(int(tolerance_state.status)) is OptimizerStatus.CONVERGED

    _, refreshed = optimizer._finalize(initial, tolerance_state, problem)

    assert OptimizerStatus(int(refreshed.status)) is OptimizerStatus.RUNNING
    assert not bool(refreshed.converged)
    assert not bool(refreshed.implicit_valid)


def test_grouped_huber_gain_accepts_when_raw_l2_increases() -> None:
    dtype = torch.float64
    points_tensor = torch.tensor([[0.0, 0.0], [0.0, 0.0], [100.0, 0.0]], dtype=dtype)
    start = points_tensor.mean(dim=0)
    kernel = Huber(delta=1.0)
    x = Variable(start.clone(), name="x")
    points = Variable(points_tensor, name="points", trainable=False)
    problem = Problem([_PointFitResidual(x, points, kernel=kernel)])
    optimizer = LevenbergMarquardt(problem)
    values = {"x": x.tensor}
    state = optimizer._init_state(values, problem)
    current_raw = 0.5 * state.residual.square().sum()
    current_robust = _grouped_cost(state.residual, kernel, 2)
    expected_group_weight = kernel.weight(state.residual.reshape(3, 2).square().sum(dim=-1))

    values_next, state_next = optimizer._update(values, state, problem)

    candidate_raw = 0.5 * state_next.residual.square().sum()
    candidate_robust = _grouped_cost(state_next.residual, kernel, 2)
    torch.testing.assert_close(state.robust_weights, expected_group_weight.repeat_interleave(2))
    torch.testing.assert_close(state.cost, current_robust)
    torch.testing.assert_close(state_next.cost, candidate_robust)
    assert state_next.gain_ratio > 0.0
    assert not torch.equal(values_next["x"], values["x"])
    assert candidate_raw > current_raw
    assert candidate_robust < current_robust


def test_grouped_tukey_gain_rejects_when_raw_l2_decreases() -> None:
    dtype = torch.float64
    kernel = Tukey(c=1.0)
    x = Variable(torch.zeros(1, dtype=dtype), name="x")
    residual = _TukeyDirectionReversalResidual(x, kernel=kernel)
    problem = Problem([residual])
    optimizer = LevenbergMarquardt(problem)
    values = {"x": x.tensor}
    state = optimizer._init_state(values, problem)
    current_raw = 0.5 * state.residual.square().sum()
    current_robust = _grouped_cost(state.residual, kernel, 2)
    residual.seen.clear()

    values_next, state_next = optimizer._update(values, state, problem)

    trial_x = residual.seen[-1]
    trial_residual = residual.evaluate(trial_x.unsqueeze(-1))
    trial_raw = 0.5 * trial_residual.square().sum()
    trial_robust = _grouped_cost(trial_residual, kernel, 2)
    assert not torch.equal(trial_x, values["x"][..., 0])
    assert trial_raw < current_raw
    assert trial_robust > current_robust
    assert state_next.gain_ratio < 0.0
    torch.testing.assert_close(values_next["x"], values["x"], rtol=0.0, atol=0.0)
    torch.testing.assert_close(state_next.cost, state.cost, rtol=0.0, atol=0.0)


def test_item_kernels_apply_per_semantic_group() -> None:
    dtype = torch.float64
    override_kernel = Huber(delta=1.0)
    default_kernel = Tukey(c=1.0)
    x = Variable(torch.zeros(1, dtype=dtype), name="x")
    problem = Problem(
        [
            _OffsetPairResidual(x, "override", 2.0, kernel=override_kernel),
            _OffsetPairResidual(x, "default", 2.0, kernel=default_kernel),
        ]
    )
    optimizer = LevenbergMarquardt(problem)
    values = {"x": x.tensor}
    state = optimizer._init_state(values, problem)

    squared_norm = torch.tensor(4.0, dtype=dtype)
    expected_cost = override_kernel.rho(squared_norm) + default_kernel.rho(squared_norm)
    expected_weights = torch.tensor([0.5, 0.5, 0.0, 0.0], dtype=dtype)
    torch.testing.assert_close(state.cost, expected_cost)
    torch.testing.assert_close(state.robust_weights, expected_weights)

    _, state_next = optimizer._update(values, state, problem)

    assert state_next.gain_ratio > 0.0
    assert state_next.cost < state.cost
    assert torch.all(state_next.robust_weights > 0.99)


def test_initial_nonfinite_model_reaches_failed_status() -> None:
    initial = torch.tensor([float("nan")])
    x, _target, problem = _linear_problem(torch.ones(1), initial=initial)
    info = LevenbergMarquardt(problem, max_iterations=2).optimize()

    assert torch.isnan(x.tensor).all()
    assert OptimizerStatus(int(info.status)) is OptimizerStatus.FAILED


def test_warm_start_retains_damping_but_refreshes_changed_target_artifacts() -> None:
    dtype = torch.float64
    x, target, problem = _linear_problem(torch.ones(1, dtype=dtype))
    optimizer = LevenbergMarquardt(problem, max_iterations=0)
    optimizer.optimize()
    assert optimizer._state is not None
    old_state = optimizer._state._replace(
        mu=torch.tensor(0.123, dtype=dtype),
        increase_factor=torch.tensor(8.0, dtype=dtype),
    )
    optimizer._state = old_state

    problem.update({"x": x.tensor, "target": torch.full_like(target.tensor, 2.0)})
    optimizer.optimize()
    assert optimizer._state is not None
    refreshed = optimizer._state

    torch.testing.assert_close(refreshed.mu, old_state.mu, rtol=0.0, atol=0.0)
    torch.testing.assert_close(refreshed.increase_factor, old_state.increase_factor, rtol=0.0, atol=0.0)
    torch.testing.assert_close(refreshed.residual, torch.tensor([-2.0], dtype=dtype))
    torch.testing.assert_close(refreshed.cost, torch.tensor(2.0, dtype=dtype))
    torch.testing.assert_close(refreshed.iterations, torch.tensor(0, dtype=torch.int64))
