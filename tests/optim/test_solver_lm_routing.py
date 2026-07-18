"""Dense and banded routing for named-block LM."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from better_robot.optim import (
    LevenbergMarquardt,
    LinearizationReason,
    Problem,
    ResidualItem,
    TemporalPattern,
    VarSpec,
)
from better_robot.optim.solvers import BandedCholesky, Cholesky


@dataclass(frozen=True)
class _DiagonalTrajectoryResidual:
    horizon: int
    width: int
    name: str = "trajectory_target"
    reads: tuple[str, ...] = ("x",)

    @property
    def dim(self) -> int:
        return self.horizon * self.width

    def __call__(self, ctx) -> torch.Tensor:
        x = ctx["x"]
        return (x - 1.0).reshape(*x.shape[:-2], self.dim)

    def temporal_structure(self, variable_name: str) -> TemporalPattern | None:
        if variable_name != "x":
            return None
        return TemporalPattern(
            rows=self.horizon,
            row_width=self.width,
            row_origin=0,
            offsets=(0,),
        )

    def temporal_jacobian_blocks(self, ctx, variable_name: str):
        x = ctx["x"]
        local = ctx.temporal_free_indices(variable_name).to(device=x.device)
        identity = torch.eye(self.width, dtype=x.dtype, device=x.device).index_select(-1, local)
        block = identity.expand(*x.shape[:-2], self.horizon, self.width, local.numel())
        # Keep the exact constant block usable by create_graph=True.
        return {0: block + x.sum(dim=(-2, -1))[..., None, None, None] * 0.0}


@dataclass(frozen=True)
class _DeclaredWithoutBlocks:
    horizon: int
    width: int
    name: str = "operator_only"
    reads: tuple[str, ...] = ("x",)

    @property
    def dim(self) -> int:
        return self.horizon * self.width

    def __call__(self, ctx) -> torch.Tensor:
        x = ctx["x"]
        return (2.0 * x - 1.0).reshape(*x.shape[:-2], self.dim)

    def temporal_structure(self, variable_name: str) -> TemporalPattern | None:
        if variable_name != "x":
            return None
        return TemporalPattern(self.horizon, self.width, 0, (0,))


@dataclass(frozen=True)
class _UndeclaredTrajectoryResidual:
    horizon: int
    width: int
    name: str = "undeclared"
    reads: tuple[str, ...] = ("x",)

    @property
    def dim(self) -> int:
        return self.horizon * self.width

    def __call__(self, ctx) -> torch.Tensor:
        x = ctx["x"]
        return x.reshape(*x.shape[:-2], self.dim)


def _problem(
    residual,
    *,
    mask: torch.Tensor | None = None,
) -> Problem:
    return Problem(
        vars=(
            VarSpec(
                "x",
                (residual.horizon, residual.width),
                mask=mask,
                time_axis=0,
            ),
        ),
        residuals=(ResidualItem(residual.name, residual),),
    )


def test_auto_banded_matches_forced_dense() -> None:
    residual = _DiagonalTrajectoryResidual(6, 3)
    problem = _problem(residual)
    values = {"x": torch.zeros(2, 6, 3, dtype=torch.float64)}

    outputs = {}
    states = {}
    for mode in ("dense", "structured", "auto"):
        solver = LevenbergMarquardt(max_iter=4, linearization=mode)
        outputs[mode], states[mode] = solver.run(values, problem)

    assert LevenbergMarquardt().resolve_linearization(problem).used == "banded"
    for mode in ("structured", "auto"):
        torch.testing.assert_close(outputs[mode]["x"], outputs["dense"]["x"], atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(states[mode].cost, states["dense"].cost, atol=1e-12, rtol=1e-12)
        assert torch.equal(states[mode].status, states["dense"].status)


def test_structured_route_never_calls_dense_jacobian(monkeypatch: pytest.MonkeyPatch) -> None:
    residual = _DiagonalTrajectoryResidual(5, 2)
    problem = _problem(residual)

    def fail_dense(*_args, **_kwargs):
        raise AssertionError("structured route materialized the dense Jacobian")

    monkeypatch.setattr(Problem, "dense_jacobian", fail_dense)
    values = {"x": torch.zeros(5, 2, dtype=torch.float64)}
    state = LevenbergMarquardt(linearization="structured").init_state(values, problem)

    assert torch.isfinite(state.cost)


def test_missing_numeric_blocks_fall_back_to_dense() -> None:
    residual = _DeclaredWithoutBlocks(4, 2)
    problem = _problem(residual)
    analysis = problem.temporal_analysis

    assert not analysis.direct_eligible
    assert analysis.reason is LinearizationReason.MISSING_TEMPORAL_BLOCKS
    assert LevenbergMarquardt().resolve_linearization(problem).used == "dense"
    with pytest.raises(ValueError, match="missing_temporal_blocks"):
        LevenbergMarquardt(linearization="structured").resolve_linearization(problem)

    values = {"x": torch.zeros(4, 2, dtype=torch.float64)}
    solved, state = LevenbergMarquardt(max_iter=4).run(values, problem)
    torch.testing.assert_close(solved["x"], torch.full_like(solved["x"], 0.5), atol=1e-8, rtol=1e-8)
    assert torch.isfinite(state.cost)


def test_undeclared_or_nonseparable_problem_falls_back_with_stable_reason() -> None:
    undeclared = _problem(_UndeclaredTrajectoryResidual(4, 2))
    decision = LevenbergMarquardt().resolve_linearization(undeclared)
    assert decision.used == "dense"
    assert decision.reason is LinearizationReason.UNDECLARED_TEMPORAL_RESIDUAL
    with pytest.raises(ValueError, match="undeclared_temporal_residual"):
        LevenbergMarquardt(linearization="structured").resolve_linearization(undeclared)

    mask = torch.tensor([1, 1, 1, 0, 1, 1, 1, 1], dtype=torch.bool)
    nonseparable = _problem(_DiagonalTrajectoryResidual(4, 2), mask=mask)
    decision = LevenbergMarquardt().resolve_linearization(nonseparable)
    assert decision.used == "dense"
    assert decision.reason is LinearizationReason.NONSEPARABLE_MASK


def test_explicit_solver_compatibility_is_not_silently_ignored() -> None:
    problem = _problem(_DiagonalTrajectoryResidual(4, 2))

    assert LevenbergMarquardt(linear_solver=Cholesky()).resolve_linearization(problem).used == "dense"
    assert LevenbergMarquardt(linear_solver=BandedCholesky()).resolve_linearization(problem).used == "banded"
    with pytest.raises(ValueError, match="incompatible_solver"):
        LevenbergMarquardt(
            linearization="dense",
            linear_solver=BandedCholesky(),
        ).resolve_linearization(problem)
