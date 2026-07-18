"""Temporal declaration, lower-band assembly, and operator parity tests."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pytest
import torch

from better_robot.optim import (
    Problem,
    ResidualItem,
    TemporalPattern,
    VarSpec,
)
from better_robot.optim.temporal import BlockBandedMatrix, LinearizationReason


@dataclass(frozen=True)
class _DifferenceResidual:
    horizon: int
    width: int
    name: str = "difference"
    reads: tuple[str, ...] = ("x",)

    @property
    def dim(self) -> int:
        return (self.horizon - 1) * self.width

    def __call__(self, ctx: Mapping[str, Any]) -> torch.Tensor:
        value = ctx["x"]
        return (value[..., 1:, :] - value[..., :-1, :]).reshape(*value.shape[:-2], self.dim)

    def temporal_structure(self, variable_name: str) -> TemporalPattern | None:
        if variable_name != "x":
            return None
        return TemporalPattern(self.horizon - 1, self.width, 0, (0, 1))

    def temporal_jacobian_blocks(
        self,
        ctx: Mapping[str, Any],
        variable_name: str,
    ) -> Mapping[int, torch.Tensor]:
        assert variable_name == "x"
        value = ctx["x"]
        indices = ctx.temporal_free_indices("x").to(device=value.device)
        identity = torch.eye(self.width, dtype=value.dtype, device=value.device).index_select(-1, indices)
        block = identity.expand(*value.shape[:-2], self.horizon - 1, self.width, indices.numel())
        anchor = value.sum(dim=(-2, -1)) * 0.0
        block = block + anchor[..., None, None, None]
        return {0: -block, 1: block}


@dataclass(frozen=True)
class _DeclaredOnlyResidual:
    horizon: int
    name: str = "declared_only"
    reads: tuple[str, ...] = ("x",)

    @property
    def dim(self) -> int:
        return self.horizon

    def __call__(self, ctx: Mapping[str, Any]) -> torch.Tensor:
        return ctx["x"][..., :, 0]

    def temporal_structure(self, variable_name: str) -> TemporalPattern | None:
        return TemporalPattern(self.horizon, 1, 0, (0,)) if variable_name == "x" else None


@dataclass(frozen=True)
class _UndeclaredResidual:
    horizon: int
    name: str = "undeclared"
    reads: tuple[str, ...] = ("x",)

    @property
    def dim(self) -> int:
        return self.horizon

    def __call__(self, ctx: Mapping[str, Any]) -> torch.Tensor:
        return ctx["x"][..., :, 0]


def _problem(*, horizon: int = 5, width: int = 3, mask: torch.Tensor | None = None) -> Problem:
    residual = _DifferenceResidual(horizon, width)
    return Problem(
        vars=(VarSpec("x", (horizon, width), mask=mask, time_axis=0),),
        residuals=(ResidualItem(residual.name, residual, weight=1.7),),
    )


def test_temporal_pattern_and_time_axis_validate_static_contracts() -> None:
    pattern = TemporalPattern(3, 2, 1, (-1, 0, 1))
    assert pattern.hessian_offsets == (-2, -1, 0, 1, 2)
    with pytest.raises(ValueError, match="sorted and unique"):
        TemporalPattern(2, 1, 0, (1, 0))
    with pytest.raises(ValueError, match="non-empty"):
        TemporalPattern(2, 1, 0, ())
    with pytest.raises(TypeError, match="time_axis"):
        VarSpec("x", (3, 2), time_axis=True)
    with pytest.raises(ValueError, match="leading, separable time axis"):
        VarSpec("x", (3, 2), time_axis=1)
    with pytest.raises(ValueError, match="leading, separable time axis"):
        VarSpec("x", (), time_axis=0)


def test_cached_analysis_keeps_zero_weight_and_requires_numeric_blocks() -> None:
    horizon = 4
    declared = _DeclaredOnlyResidual(horizon)
    declared_problem = Problem(
        vars=(VarSpec("x", (horizon, 2), time_axis=0),),
        residuals=(ResidualItem(declared.name, declared),),
    )
    analysis = declared_problem.temporal_analysis
    assert not analysis.direct_eligible
    assert analysis.reason is LinearizationReason.MISSING_TEMPORAL_BLOCKS

    undeclared = _UndeclaredResidual(horizon)
    zero_problem = Problem(
        vars=(VarSpec("x", (horizon, 2), time_axis=0),),
        residuals=(ResidualItem(undeclared.name, undeclared, weight=0.0),),
    )
    assert not zero_problem.temporal_analysis.direct_eligible
    assert zero_problem.temporal_analysis.reason is LinearizationReason.UNDECLARED_TEMPORAL_RESIDUAL


def test_nonseparable_mask_falls_back_but_separable_mask_exposes_local_indices() -> None:
    horizon, width = 4, 3
    separable = torch.tensor([True, False, True]).repeat(horizon)
    problem = _problem(horizon=horizon, width=width, mask=separable)
    assert problem.temporal_analysis.direct_eligible
    assert problem.temporal_analysis.reduced_width == 2
    torch.testing.assert_close(problem.vars[0].temporal_free_indices, torch.tensor([0, 2]))

    nonseparable = separable.clone()
    nonseparable[width + 1] = True
    fallback = _problem(horizon=horizon, width=width, mask=nonseparable)
    assert not fallback.temporal_analysis.direct_eligible
    assert fallback.temporal_analysis.reason is LinearizationReason.NONSEPARABLE_MASK


@pytest.mark.parametrize("batch_shape", [(), (2,)])
def test_structured_normal_matches_dense_jacobian_and_flat_operators(batch_shape: tuple[int, ...]) -> None:
    torch.manual_seed(11)
    horizon, width = 5, 3
    problem = _problem(horizon=horizon, width=width)
    values = {"x": torch.randn(*batch_shape, horizon, width)}
    residual = problem.residual(values)
    row_scale = torch.linspace(0.3, 1.1, problem.dim_total).expand(*batch_shape, problem.dim_total).clone()

    structured = problem.structured_normal(values, residual=residual, row_scale=row_scale)
    dense_j = problem.dense_jacobian(values) * row_scale.unsqueeze(-1)
    weighted_residual = residual * row_scale
    dense_gradient = (dense_j.mT @ weighted_residual.unsqueeze(-1)).squeeze(-1)
    dense_normal = dense_j.mT @ dense_j

    assert structured.normal.bandwidth == 1
    assert structured.normal.bands.shape == (*batch_shape, horizon, 2, width, width)
    torch.testing.assert_close(structured.normal.densify(), dense_normal)
    torch.testing.assert_close(structured.gradient, dense_gradient)
    torch.testing.assert_close(structured.normal_diagonal, dense_normal.diagonal(dim1=-2, dim2=-1))
    assert bool(structured.finite.all())

    vector = torch.randn(*batch_shape, horizon * width)
    cotangent = torch.randn(*batch_shape, problem.dim_total)
    torch.testing.assert_close(structured.jvp(vector), (dense_j @ vector.unsqueeze(-1)).squeeze(-1))
    torch.testing.assert_close(structured.vjp(cotangent), (dense_j.mT @ cotangent.unsqueeze(-1)).squeeze(-1))
    torch.testing.assert_close(
        structured.normal_matvec(vector),
        (dense_normal @ vector.unsqueeze(-1)).squeeze(-1),
    )
    torch.testing.assert_close(structured.normal.matvec(vector), structured.normal_matvec(vector))


def test_block_banded_scaled_restricted_matches_dense_oracle() -> None:
    torch.manual_seed(3)
    raw = torch.randn(2, 4, 2, 3, 3)
    bands = BlockBandedMatrix(raw, bandwidth=1)
    scale = torch.linspace(0.5, 1.5, 12)
    movable = torch.randint(0, 2, (2, 12)).to(dtype=raw.dtype)
    diagonal = torch.linspace(0.1, 0.4, 12).expand(2, 12)
    transformed = bands.scaled_restricted(scale, movable, diagonal)

    factor = scale * movable
    expected = factor.unsqueeze(-1) * bands.densify() * factor.unsqueeze(-2)
    expected = expected + torch.diag_embed(diagonal)
    torch.testing.assert_close(transformed.densify(), expected)
