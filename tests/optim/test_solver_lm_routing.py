"""Dense and banded routing for object-referenced LM problems."""

from __future__ import annotations

import warnings

import pytest
import torch

from better_robot.optim import (
    AutodiffFallbackWarning,
    LevenbergMarquardt,
    LinearizationReason,
    Problem,
    Residual,
    TemporalPattern,
    Variable,
)
from better_robot.optim.solvers import BandedCholesky, Cholesky


def _matches(variable: Variable | str, expected: Variable) -> bool:
    return variable is expected or variable == expected.name


class _TrajectoryResidual(Residual):
    def __init__(self, x: Variable, horizon: int, width: int, *, name: str) -> None:
        self.x = x
        self.horizon = horizon
        self.width = width
        super().__init__(x, dim=horizon * width, name=name)


class _DiagonalTrajectoryResidual(_TrajectoryResidual):
    def __init__(self, x: Variable, horizon: int, width: int) -> None:
        super().__init__(x, horizon, width, name="trajectory_target")

    def error(self) -> torch.Tensor:
        x = self.x.tensor
        return (x - 1.0).reshape(*x.shape[:-2], self.dim)

    def temporal_structure(self, variable: Variable | str) -> TemporalPattern | None:
        if not _matches(variable, self.x):
            return None
        return TemporalPattern(self.horizon, self.width, 0, (0,))

    def temporal_jacobian_blocks(self, variable: Variable | str) -> dict[int, torch.Tensor]:
        if not _matches(variable, self.x):
            return {}
        x = self.x.tensor
        identity = torch.eye(self.width, dtype=x.dtype, device=x.device)
        block = identity.expand(*x.shape[:-2], self.horizon, self.width, self.width)
        return {0: block + x.sum(dim=(-2, -1))[..., None, None, None] * 0.0}


class _DeclaredWithoutBlocks(_TrajectoryResidual):
    def __init__(self, x: Variable, horizon: int, width: int) -> None:
        super().__init__(x, horizon, width, name="operator_only")

    def error(self) -> torch.Tensor:
        x = self.x.tensor
        return (2.0 * x - 1.0).reshape(*x.shape[:-2], self.dim)

    def temporal_structure(self, variable: Variable | str) -> TemporalPattern | None:
        if not _matches(variable, self.x):
            return None
        return TemporalPattern(self.horizon, self.width, 0, (0,))


class _UndeclaredTrajectoryResidual(_TrajectoryResidual):
    def __init__(self, x: Variable, horizon: int, width: int) -> None:
        super().__init__(x, horizon, width, name="undeclared")

    def error(self) -> torch.Tensor:
        x = self.x.tensor
        return x.reshape(*x.shape[:-2], self.dim)


def _problem(
    residual_type: type[_TrajectoryResidual],
    horizon: int,
    width: int,
    *,
    dtype: torch.dtype = torch.float64,
) -> tuple[Variable, Problem]:
    x = Variable(
        torch.zeros(horizon, width, dtype=dtype),
        name="x",
        time_axis=0,
    )
    return x, Problem([residual_type(x, horizon, width)])


def test_auto_banded_matches_forced_dense() -> None:
    values = torch.zeros(2, 6, 3, dtype=torch.float64)
    outputs: dict[str, torch.Tensor] = {}
    costs: dict[str, torch.Tensor] = {}
    statuses: dict[str, torch.Tensor] = {}
    for mode in ("dense", "structured", "auto"):
        x, problem = _problem(_DiagonalTrajectoryResidual, 6, 3)
        problem.update({"x": values.clone()})
        optimizer = LevenbergMarquardt(problem, max_iterations=4, linearization=mode)
        info = optimizer.optimize()
        outputs[mode] = x.tensor.clone()
        costs[mode] = info.cost
        statuses[mode] = info.status

    _x, auto_problem = _problem(_DiagonalTrajectoryResidual, 6, 3)
    assert LevenbergMarquardt(auto_problem).resolve_linearization(auto_problem).used == "banded"
    for mode in ("structured", "auto"):
        torch.testing.assert_close(outputs[mode], outputs["dense"], atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(costs[mode], costs["dense"], atol=1e-12, rtol=1e-12)
        assert torch.equal(statuses[mode], statuses["dense"])


def test_structured_route_never_calls_dense_jacobian(monkeypatch: pytest.MonkeyPatch) -> None:
    _x, problem = _problem(_DiagonalTrajectoryResidual, 5, 2)

    def fail_dense(*_args, **_kwargs):
        raise AssertionError("structured route materialized the dense Jacobian")

    monkeypatch.setattr(Problem, "dense_jacobian", fail_dense)
    values = {"x": torch.zeros(5, 2, dtype=torch.float64)}
    optimizer = LevenbergMarquardt(problem, linearization="structured")
    state = optimizer._init_state(values, problem)

    assert torch.isfinite(state.cost)


def test_missing_numeric_blocks_fall_back_to_dense() -> None:
    x, problem = _problem(_DeclaredWithoutBlocks, 4, 2, dtype=torch.float32)
    optimizer = LevenbergMarquardt(problem, max_iterations=4, jacobian_strategy="jacrev")
    analysis = problem.temporal_analysis

    assert not analysis.direct_eligible
    assert analysis.reason is LinearizationReason.MISSING_TEMPORAL_BLOCKS
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", AutodiffFallbackWarning)
        assert optimizer.resolve_linearization(problem).used == "dense"
        assert optimizer.resolve_linearization(problem).used == "dense"
    fallback = [item for item in caught if issubclass(item.category, AutodiffFallbackWarning)]
    assert len(fallback) == 1
    assert "_DeclaredWithoutBlocks residual 'operator_only'" in str(fallback[0].message)
    assert "linearization='auto' is using the dense route" in str(fallback[0].message)
    with pytest.raises(ValueError, match="missing_temporal_blocks"):
        LevenbergMarquardt(problem, linearization="structured").resolve_linearization(problem)

    info = optimizer.optimize()
    torch.testing.assert_close(x.tensor, torch.full_like(x.tensor, 0.5), atol=1e-8, rtol=1e-8)
    assert torch.isfinite(info.cost)


def test_explicit_dense_temporal_route_does_not_warn() -> None:
    _x, problem = _problem(_DeclaredWithoutBlocks, 4, 2, dtype=torch.float32)

    with warnings.catch_warnings():
        warnings.simplefilter("error", AutodiffFallbackWarning)
        decision = LevenbergMarquardt(problem, linearization="dense").resolve_linearization(problem)

    assert decision.reason is LinearizationReason.FORCED_DENSE


def test_missing_blocks_with_banded_only_solver_errors_without_warning() -> None:
    _x, problem = _problem(_DeclaredWithoutBlocks, 4, 2, dtype=torch.float32)

    with warnings.catch_warnings():
        warnings.simplefilter("error", AutodiffFallbackWarning)
        with pytest.raises(ValueError, match="incompatible_solver"):
            LevenbergMarquardt(problem, solver=BandedCholesky()).resolve_linearization(problem)


def test_undeclared_problem_falls_back_with_stable_reason() -> None:
    _x, undeclared = _problem(_UndeclaredTrajectoryResidual, 4, 2)
    optimizer = LevenbergMarquardt(undeclared)
    decision = optimizer.resolve_linearization(undeclared)
    assert decision.used == "dense"
    assert decision.reason is LinearizationReason.UNDECLARED_TEMPORAL_RESIDUAL
    with pytest.raises(ValueError, match="undeclared_temporal_residual"):
        LevenbergMarquardt(undeclared, linearization="structured").resolve_linearization(undeclared)


def test_explicit_solver_compatibility_is_not_silently_ignored() -> None:
    _x, problem = _problem(_DiagonalTrajectoryResidual, 4, 2)

    assert LevenbergMarquardt(problem, solver=Cholesky()).resolve_linearization(problem).used == "dense"
    assert LevenbergMarquardt(problem, solver=BandedCholesky()).resolve_linearization(problem).used == "banded"
    with pytest.raises(ValueError, match="incompatible_solver"):
        LevenbergMarquardt(
            problem,
            linearization="dense",
            solver=BandedCholesky(),
        ).resolve_linearization(problem)
