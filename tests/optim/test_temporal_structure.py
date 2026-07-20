"""Temporal declaration, lower-band assembly, and operator parity tests."""

from __future__ import annotations

from collections.abc import Mapping

import pytest
import torch

from better_robot.optim import Problem, Residual, TemporalPattern, Variable
from better_robot.optim.temporal import BlockBandedMatrix, LinearizationReason


class _DifferenceResidual(Residual):
    def __init__(self, x: Variable, *, weight: float = 1.7) -> None:
        self.x = x
        self.horizon = x.time_length
        self.width = x.shape[-1]
        super().__init__(
            x,
            dim=(self.horizon - 1) * self.width,
            weight=weight,
            name="difference",
        )

    def error(self) -> torch.Tensor:
        value = self.x.tensor
        return (value[..., 1:, :] - value[..., :-1, :]).reshape(*value.shape[:-2], self.dim)

    def temporal_structure(self, variable: Variable | str) -> TemporalPattern | None:
        if variable is not self.x and variable != self.x.name:
            return None
        return TemporalPattern(self.horizon - 1, self.width, 0, (0, 1))

    def temporal_jacobian_blocks(self, variable: Variable | str) -> Mapping[int, torch.Tensor]:
        if variable is not self.x and variable != self.x.name:
            return {}
        value = self.x.tensor
        identity = torch.eye(self.width, dtype=value.dtype, device=value.device)
        block = identity.expand(*value.shape[:-2], self.horizon - 1, self.width, self.width)
        anchor = value.sum(dim=(-2, -1)) * 0.0
        block = block + anchor[..., None, None, None]
        return {0: -block, 1: block}


class _DeclaredOnlyResidual(Residual):
    def __init__(self, x: Variable) -> None:
        self.x = x
        self.horizon = x.time_length
        super().__init__(x, dim=self.horizon, name="declared_only")

    def error(self) -> torch.Tensor:
        return self.x.tensor[..., :, 0]

    def temporal_structure(self, variable: Variable | str) -> TemporalPattern | None:
        if variable is not self.x and variable != self.x.name:
            return None
        return TemporalPattern(self.horizon, 1, 0, (0,))


class _UndeclaredResidual(Residual):
    def __init__(self, x: Variable, *, weight: float = 1.0) -> None:
        self.x = x
        self.horizon = x.time_length
        super().__init__(x, dim=self.horizon, weight=weight, name="undeclared")

    def error(self) -> torch.Tensor:
        return self.x.tensor[..., :, 0]


def _problem(*, horizon: int = 5, width: int = 3) -> Problem:
    x = Variable(torch.zeros(horizon, width), name="x", time_axis=0)
    problem = Problem([_DifferenceResidual(x)])
    problem.error()
    return problem


def test_temporal_pattern_and_time_axis_validate_static_contracts() -> None:
    pattern = TemporalPattern(3, 2, 1, (-1, 0, 1))
    assert pattern.hessian_offsets == (-2, -1, 0, 1, 2)
    with pytest.raises(ValueError, match="sorted and unique"):
        TemporalPattern(2, 1, 0, (1, 0))
    with pytest.raises(ValueError, match="non-empty"):
        TemporalPattern(2, 1, 0, ())
    with pytest.raises(TypeError, match="time_axis"):
        Variable(torch.zeros(3, 2), name="x", time_axis=True)
    with pytest.raises(ValueError, match="leading event axis 0"):
        Variable(torch.zeros(3, 2), name="x", time_axis=1)
    with pytest.raises(ValueError, match="non-scalar event shape"):
        Variable(torch.tensor(0.0), name="x", time_axis=0)


def test_cached_analysis_keeps_zero_weight_and_requires_numeric_blocks() -> None:
    horizon = 4
    declared_x = Variable(torch.zeros(horizon, 2), name="x", time_axis=0)
    declared_problem = Problem([_DeclaredOnlyResidual(declared_x)])
    declared_problem.error()
    analysis = declared_problem.temporal_analysis
    assert not analysis.direct_eligible
    assert analysis.reason is LinearizationReason.MISSING_TEMPORAL_BLOCKS

    undeclared_x = Variable(torch.zeros(horizon, 2), name="x", time_axis=0)
    zero_problem = Problem([_UndeclaredResidual(undeclared_x, weight=0.0)])
    zero_problem.error()
    assert not zero_problem.temporal_analysis.direct_eligible
    assert zero_problem.temporal_analysis.reason is LinearizationReason.UNDECLARED_TEMPORAL_RESIDUAL


@pytest.mark.parametrize("batch_shape", [(), (2,)])
def test_structured_normal_matches_dense_jacobian_and_flat_operators(batch_shape: tuple[int, ...]) -> None:
    torch.manual_seed(11)
    horizon, width = 5, 3
    problem = _problem(horizon=horizon, width=width)
    problem.update({"x": torch.randn(*batch_shape, horizon, width)})
    residual = problem.error()
    row_scale = torch.linspace(0.3, 1.1, problem.dim_total).expand(*batch_shape, problem.dim_total).clone()

    structured = problem.structured_normal(residual=residual, row_scale=row_scale)
    dense_j = problem.dense_jacobian() * row_scale.unsqueeze(-1)
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


def test_block_banded_restricted_matches_dense_oracle() -> None:
    torch.manual_seed(3)
    raw = torch.randn(2, 4, 2, 3, 3)
    bands = BlockBandedMatrix(raw, bandwidth=1)
    movable = torch.randint(0, 2, (2, 12)).to(dtype=raw.dtype)
    diagonal = torch.linspace(0.1, 0.4, 12).expand(2, 12)
    transformed = bands.restricted(movable, diagonal)

    expected = movable.unsqueeze(-1) * bands.densify() * movable.unsqueeze(-2)
    expected = expected + torch.diag_embed(diagonal)
    torch.testing.assert_close(transformed.densify(), expected)
