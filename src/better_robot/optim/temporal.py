"""Temporal analysis and direct block-banded normal assembly."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from numbers import Real
from typing import TYPE_CHECKING

import torch

from ..residuals.structure import TemporalPattern

if TYPE_CHECKING:
    from .problem import Problem, Weight
    from .variables import Values, VarSpec


class LinearizationReason(str, Enum):
    FORCED_DENSE = "forced_dense"
    ELIGIBLE_BANDED = "eligible_banded"
    NO_TIME_VARIABLE = "no_time_variable"
    MULTIPLE_OPTIMIZED_VARIABLES = "multiple_optimized_variables"
    NONSEPARABLE_MASK = "nonseparable_mask"
    UNDECLARED_TEMPORAL_RESIDUAL = "undeclared_temporal_residual"
    MISSING_TEMPORAL_BLOCKS = "missing_temporal_blocks"
    MIXED_OPTIMIZED_DEPENDENCY = "mixed_optimized_dependency"
    EXPLICIT_DENSE_SOLVER = "explicit_dense_solver"


@dataclass(frozen=True)
class TemporalAnalysis:
    direct_eligible: bool
    reason: LinearizationReason
    detail: str
    variable_name: str | None = None
    time_length: int | None = None
    tangent_width: int | None = None
    reduced_width: int | None = None
    bandwidth: int = 0
    patterns: tuple[tuple[str, TemporalPattern], ...] = ()


def _check_flat(name: str, value: torch.Tensor, matrix: BlockBandedMatrix) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim < 1 or value.shape[-1] != matrix.size:
        raise ValueError(f"{name} must end in flattened size {matrix.size}, got {tuple(value.shape)}")
    if value.dtype != matrix.bands.dtype or value.device != matrix.bands.device:
        raise ValueError(
            f"{name} must share band dtype/device {matrix.bands.dtype}/{matrix.bands.device}, "
            f"got {value.dtype}/{value.device}"
        )


@dataclass(frozen=True)
class BlockBandedMatrix:
    """Symmetric matrix where ``bands[..., t, k]`` stores ``H[t, t-k]``."""

    bands: torch.Tensor
    bandwidth: int

    def __post_init__(self) -> None:
        if not isinstance(self.bands, torch.Tensor):
            raise TypeError("BlockBandedMatrix bands must be a torch.Tensor")
        if self.bands.ndim < 4:
            raise ValueError("BlockBandedMatrix bands must have shape (B..., T, w+1, d, d)")
        if isinstance(self.bandwidth, bool) or not isinstance(self.bandwidth, int):
            raise TypeError("BlockBandedMatrix bandwidth must be an int")
        if self.bandwidth < 0:
            raise ValueError("BlockBandedMatrix bandwidth must be non-negative")
        if self.bands.shape[-1] != self.bands.shape[-2] or self.bands.shape[-1] <= 0:
            raise ValueError("BlockBandedMatrix blocks must be non-empty and square")
        if self.bands.shape[-3] != self.bandwidth + 1:
            raise ValueError(
                "BlockBandedMatrix band axis must have bandwidth + 1 entries; "
                f"got {self.bands.shape[-3]} and bandwidth={self.bandwidth}"
            )
        if self.bands.shape[-4] <= 0:
            raise ValueError("BlockBandedMatrix time length must be positive")
        if self.bandwidth >= self.bands.shape[-4]:
            raise ValueError("BlockBandedMatrix bandwidth must be smaller than time length")
        if not self.bands.is_floating_point():
            raise TypeError("BlockBandedMatrix bands must use a floating dtype")

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return tuple(self.bands.shape[:-4])

    @property
    def time_length(self) -> int:
        return self.bands.shape[-4]

    @property
    def block_size(self) -> int:
        return self.bands.shape[-1]

    @property
    def size(self) -> int:
        return self.time_length * self.block_size

    @property
    def finite(self) -> torch.Tensor:
        return torch.isfinite(self.bands).all(dim=(-4, -3, -2, -1))

    def diagonal(self) -> torch.Tensor:
        diagonal = self.bands[..., :, 0, :, :].diagonal(dim1=-2, dim2=-1)
        return diagonal.reshape(*self.batch_shape, self.size)

    def densify(self) -> torch.Tensor:
        dense = self.bands.new_zeros(*self.batch_shape, self.size, self.size)
        d = self.block_size
        for time in range(self.time_length):
            row = slice(time * d, (time + 1) * d)
            for offset in range(min(self.bandwidth, time) + 1):
                column = slice((time - offset) * d, (time - offset + 1) * d)
                block = self.bands[..., time, offset, :, :]
                dense[..., row, column] = block
                if offset:
                    dense[..., column, row] = block.mT
        return dense

    def matvec(self, vector: torch.Tensor) -> torch.Tensor:
        _check_flat("vector", vector, self)
        batch = torch.broadcast_shapes(self.batch_shape, tuple(vector.shape[:-1]))
        bands = self.bands.expand(*batch, *self.bands.shape[-4:])
        shaped = vector.expand(*batch, self.size).reshape(*batch, self.time_length, self.block_size)
        result = (bands[..., :, 0, :, :] @ shaped.unsqueeze(-1)).squeeze(-1)
        for offset in range(1, self.bandwidth + 1):
            lower = bands[..., offset:, offset, :, :]
            result[..., offset:, :] += (lower @ shaped[..., :-offset, :].unsqueeze(-1)).squeeze(-1)
            result[..., :-offset, :] += (lower.mT @ shaped[..., offset:, :].unsqueeze(-1)).squeeze(-1)
        return result.reshape(*batch, self.size)

    def add_diagonal(self, diagonal: torch.Tensor) -> BlockBandedMatrix:
        _check_flat("diagonal", diagonal, self)
        batch = torch.broadcast_shapes(self.batch_shape, tuple(diagonal.shape[:-1]))
        bands = self.bands.expand(*batch, *self.bands.shape[-4:]).clone()
        shaped = diagonal.expand(*batch, self.size).reshape(*batch, self.time_length, self.block_size)
        bands[..., :, 0, :, :].diagonal(dim1=-2, dim2=-1).add_(shaped)
        return BlockBandedMatrix(bands, self.bandwidth)

    def scaled_restricted(
        self,
        scale: torch.Tensor,
        movable: torch.Tensor,
        diagonal: torch.Tensor,
    ) -> BlockBandedMatrix:
        for name, value in (("scale", scale), ("movable", movable), ("diagonal", diagonal)):
            _check_flat(name, value, self)
        batch = torch.broadcast_shapes(
            self.batch_shape, tuple(scale.shape[:-1]), tuple(movable.shape[:-1]), tuple(diagonal.shape[:-1])
        )
        bands = self.bands.expand(*batch, *self.bands.shape[-4:])
        factor = (scale * movable).expand(*batch, self.size).reshape(*batch, self.time_length, self.block_size)
        transformed = torch.zeros_like(bands)
        for offset in range(self.bandwidth + 1):
            transformed[..., offset:, offset, :, :] = (
                bands[..., offset:, offset, :, :]
                * factor[..., offset:, :].unsqueeze(-1)
                * factor[..., : self.time_length - offset, :].unsqueeze(-2)
            )
        return BlockBandedMatrix(transformed, self.bandwidth).add_diagonal(diagonal)


def _analysis(
    eligible: bool,
    reason: LinearizationReason,
    detail: str,
    spec: VarSpec | None = None,
    *,
    bandwidth: int = 0,
    patterns: tuple[tuple[str, TemporalPattern], ...] = (),
) -> TemporalAnalysis:
    return TemporalAnalysis(
        eligible,
        reason,
        detail,
        None if spec is None else spec.name,
        None if spec is None else spec.time_length,
        None if spec is None else spec.temporal_tangent_width,
        None if spec is None or not spec.temporal_mask_is_separable else spec.temporal_reduced_width,
        bandwidth,
        patterns,
    )


def analyze_temporal_problem(problem: Problem) -> TemporalAnalysis:  # noqa: PLR0911
    """Analyze static declarations without reading numeric values."""
    optimized = tuple(spec for spec in problem.vars if spec.free_dim > 0)
    temporal = tuple(spec for spec in optimized if spec.time_axis is not None)
    if not temporal:
        return _analysis(False, LinearizationReason.NO_TIME_VARIABLE, "problem has no free time-annotated variable")
    if len(optimized) != 1 or len(temporal) != 1:
        names = tuple(spec.name for spec in optimized)
        return _analysis(
            False,
            LinearizationReason.MULTIPLE_OPTIMIZED_VARIABLES,
            f"structured v1 requires exactly one free variable; found {names}",
        )
    spec = temporal[0]
    if not spec.temporal_mask_is_separable:
        return _analysis(
            False,
            LinearizationReason.NONSEPARABLE_MASK,
            f"temporal variable {spec.name!r} mask differs across knots",
            spec,
        )

    patterns: list[tuple[str, TemporalPattern]] = []
    missing: list[str] = []
    bandwidth = 0
    for item in problem.residuals:
        dependencies = problem._item_variable_reads[item.name]
        if spec.name not in dependencies:
            continue
        other_free = tuple(
            name for name in dependencies if name != spec.name and problem._vars_by_name[name].free_dim > 0
        )
        if other_free:
            return _analysis(
                False,
                LinearizationReason.MIXED_OPTIMIZED_DEPENDENCY,
                f"residual {item.name!r} couples {spec.name!r} to free variables {other_free}",
                spec,
            )
        declaration = getattr(item.residual, "temporal_structure", None)
        if not callable(declaration):
            return _analysis(
                False,
                LinearizationReason.UNDECLARED_TEMPORAL_RESIDUAL,
                f"residual {item.name!r} depends on {spec.name!r} but declares no temporal structure",
                spec,
            )
        pattern = declaration(spec.name)
        if not isinstance(pattern, TemporalPattern):
            raise TypeError(
                f"Residual {item.name!r} temporal_structure({spec.name!r}) must return TemporalPattern, "
                f"not {type(pattern).__name__}"
            )
        if pattern.rows * pattern.row_width != item.residual.dim:
            raise ValueError(
                f"Residual {item.name!r} temporal pattern rows*row_width must equal dim={item.residual.dim}, "
                f"got {pattern.rows}*{pattern.row_width}"
            )
        first, last = (
            pattern.row_origin + pattern.offsets[0],
            pattern.row_origin + pattern.rows - 1 + pattern.offsets[-1],
        )
        if first < 0 or last >= spec.time_length:
            raise ValueError(
                f"Residual {item.name!r} temporal pattern reaches knots [{first}, {last}] "
                f"outside [0, {spec.time_length})"
            )
        patterns.append((item.name, pattern))
        bandwidth = max(bandwidth, *(abs(value) for value in pattern.hessian_offsets))
        if not callable(getattr(item.residual, "temporal_jacobian_blocks", None)):
            missing.append(item.name)

    entries = tuple(patterns)
    if missing:
        return _analysis(
            False,
            LinearizationReason.MISSING_TEMPORAL_BLOCKS,
            f"residuals {tuple(missing)} declare support but no temporal numeric blocks",
            spec,
            bandwidth=bandwidth,
            patterns=entries,
        )
    return _analysis(
        True,
        LinearizationReason.ELIGIBLE_BANDED,
        f"all residuals depending on {spec.name!r} provide temporal blocks",
        spec,
        bandwidth=bandwidth,
        patterns=entries,
    )


@dataclass(frozen=True)
class _StructuredTerm:
    row_slice: slice
    pattern: TemporalPattern
    blocks: tuple[tuple[int, torch.Tensor], ...]


@dataclass(frozen=True)
class StructuredNormal:
    gradient: torch.Tensor
    normal_diagonal: torch.Tensor
    normal: BlockBandedMatrix
    finite: torch.Tensor
    _residual_dim: int = field(repr=False)
    _terms: tuple[_StructuredTerm, ...] = field(repr=False)

    def _validate(self, name: str, vector: torch.Tensor, size: int) -> None:
        if not isinstance(vector, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        expected = (*self.normal.batch_shape, size)
        if tuple(vector.shape) != expected:
            raise ValueError(f"{name} must have shape {expected}, got {tuple(vector.shape)}")
        if vector.dtype != self.gradient.dtype or vector.device != self.gradient.device:
            raise ValueError(f"{name} must preserve structured-normal dtype/device")

    def jvp(self, vector: torch.Tensor) -> torch.Tensor:
        self._validate("vector", vector, self.normal.size)
        batch = self.normal.batch_shape
        shaped = vector.reshape(*batch, self.normal.time_length, self.normal.block_size)
        result = vector.new_zeros(*batch, self._residual_dim)
        for term in self._terms:
            rows = vector.new_zeros(*batch, term.pattern.rows, term.pattern.row_width)
            for offset, block in term.blocks:
                start = term.pattern.row_origin + offset
                rows += torch.einsum("...rwd,...rd->...rw", block, shaped[..., start : start + term.pattern.rows, :])
            result[..., term.row_slice] = rows.reshape(*batch, -1)
        return result

    def vjp(self, cotangent: torch.Tensor) -> torch.Tensor:
        self._validate("cotangent", cotangent, self._residual_dim)
        batch = self.normal.batch_shape
        result = cotangent.new_zeros(*batch, self.normal.time_length, self.normal.block_size)
        for term in self._terms:
            rows = cotangent[..., term.row_slice].reshape(*batch, term.pattern.rows, term.pattern.row_width)
            for offset, block in term.blocks:
                start = term.pattern.row_origin + offset
                result[..., start : start + term.pattern.rows, :] += torch.einsum("...rwd,...rw->...rd", block, rows)
        return result.reshape(*batch, self.normal.size)

    def normal_matvec(self, vector: torch.Tensor) -> torch.Tensor:
        return self.vjp(self.jvp(vector))


def _broadcast_weight(weight: Weight, output: torch.Tensor) -> torch.Tensor:
    value = (
        weight.to(dtype=output.dtype, device=output.device)
        if isinstance(weight, torch.Tensor)
        else output.new_tensor(weight)
    )
    return value.reshape(*value.shape, *((1,) * (output.ndim - value.ndim)))


def _validate_weight(name: str, weight: Weight, batch: tuple[int, ...], exemplar: torch.Tensor) -> None:
    if not isinstance(weight, torch.Tensor):
        return
    if tuple(weight.shape) not in ((), batch):
        raise ValueError(
            f"Tensor weight for item {name!r} must be scalar or have exact batch shape {batch}, got {tuple(weight.shape)}"
        )
    if weight.dtype != exemplar.dtype or weight.device != exemplar.device:
        raise ValueError(f"Tensor weight for item {name!r} must preserve working dtype/device")


def assemble_structured_normal(  # noqa: PLR0912, PLR0915
    problem: Problem,
    values: Values,
    *,
    batch_shape: tuple[int, ...],
    weights: Mapping[str, Weight] | None = None,
    row_scale: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
    create_graph: bool = False,
    validate_runtime: bool = True,
) -> StructuredNormal:
    analysis = problem.temporal_analysis
    if not analysis.direct_eligible:
        raise ValueError(f"{analysis.reason.value}: {analysis.detail}")
    assert analysis.variable_name is not None
    assert analysis.time_length is not None
    assert analysis.reduced_width is not None
    spec = problem._vars_by_name[analysis.variable_name]
    exemplar = values[spec.name]
    ctx = problem._make_context(values)
    if residual is None:
        residual = problem._residual_with_context(values, batch_shape, ctx, weights, validate_runtime=validate_runtime)
    if row_scale is None:
        row_scale = torch.ones_like(residual)
    expected_rows = (*batch_shape, problem.dim_total)
    for name, value in (("residual", residual), ("row_scale", row_scale)):
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != expected_rows:
            actual = tuple(value.shape) if isinstance(value, torch.Tensor) else type(value).__name__
            raise ValueError(f"{name} must have shape {expected_rows}, got {actual}")
        if value.dtype != exemplar.dtype or value.device != exemplar.device:
            raise ValueError(f"{name} must preserve values dtype/device")

    T, d, w = analysis.time_length, analysis.reduced_width, analysis.bandwidth
    bands = exemplar.new_zeros(*batch_shape, T, w + 1, d, d)
    gradient = exemplar.new_zeros(*batch_shape, T, d)
    terms: list[_StructuredTerm] = []
    patterns = dict(analysis.patterns)
    weighted_residual = residual * row_scale
    graph_inputs = (*values.values(), *problem.parameters.values())
    graph_required = create_graph and any(value.requires_grad for value in graph_inputs)

    for item in problem.residuals:
        pattern = patterns.get(item.name)
        if pattern is None:
            continue
        weight = problem._weights_for(item, weights)
        if validate_runtime:
            _validate_weight(item.name, weight, batch_shape, exemplar)
        if isinstance(weight, Real) and float(weight) == 0.0:
            continue
        raw = item.residual.temporal_jacobian_blocks(ctx, spec.name)
        if not isinstance(raw, Mapping):
            raise TypeError(f"Residual {item.name!r} temporal_jacobian_blocks must return a mapping")
        if set(raw) != set(pattern.offsets):
            raise ValueError(
                f"Residual {item.name!r} temporal block offsets must be {pattern.offsets}, got {tuple(raw)}"
            )
        item_scale = row_scale[..., problem.row_offsets[item.name]].reshape(
            *batch_shape, pattern.rows, pattern.row_width, 1
        )
        blocks: list[tuple[int, torch.Tensor]] = []
        for offset in pattern.offsets:
            block = raw[offset]
            expected = (*batch_shape, pattern.rows, pattern.row_width, d)
            if not isinstance(block, torch.Tensor) or tuple(block.shape) != expected:
                actual = tuple(block.shape) if isinstance(block, torch.Tensor) else type(block).__name__
                raise ValueError(
                    f"Residual {item.name!r} temporal block {offset} must have shape {expected}, got {actual}"
                )
            if block.dtype != exemplar.dtype or block.device != exemplar.device:
                raise ValueError(f"Residual {item.name!r} temporal block {offset} must preserve dtype/device")
            if graph_required and not block.requires_grad:
                raise ValueError(f"Temporal block {(item.name, spec.name, offset)!r} cannot honor create_graph=True")
            weighted = block * _broadcast_weight(weight, block) * item_scale
            blocks.append((offset, weighted if create_graph else weighted.detach()))

        row = weighted_residual[..., problem.row_offsets[item.name]].reshape(
            *batch_shape, pattern.rows, pattern.row_width
        )
        by_offset = dict(blocks)
        for offset, block in blocks:
            start = pattern.row_origin + offset
            gradient[..., start : start + pattern.rows, :] += torch.einsum("...rwd,...rw->...rd", block, row)
        for later in pattern.offsets:
            for earlier in pattern.offsets:
                if later < earlier:
                    continue
                start, offset = pattern.row_origin + later, later - earlier
                contribution = torch.einsum("...rwi,...rwj->...rij", by_offset[later], by_offset[earlier])
                bands[..., start : start + pattern.rows, offset, :, :] += contribution
        terms.append(_StructuredTerm(problem.row_offsets[item.name], pattern, tuple(blocks)))

    normal = BlockBandedMatrix(bands, w)
    flat_gradient = gradient.reshape(*batch_shape, T * d)
    diagonal = normal.diagonal()
    finite = normal.finite & torch.isfinite(flat_gradient).all(dim=-1) & torch.isfinite(diagonal).all(dim=-1)
    return StructuredNormal(flat_gradient, diagonal, normal, finite, problem.dim_total, tuple(terms))


__all__ = [
    "BlockBandedMatrix",
    "LinearizationReason",
    "StructuredNormal",
    "TemporalAnalysis",
    "analyze_temporal_problem",
    "assemble_structured_normal",
]
