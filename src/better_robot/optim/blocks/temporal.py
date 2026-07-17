"""Cached temporal analysis and direct block-banded normal assembly."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Real
from typing import TYPE_CHECKING

import torch

from ...residuals.structure import TemporalPattern
from ..structure import BlockBandedMatrix, LinearizationReason, TemporalAnalysis

if TYPE_CHECKING:
    from .problem import Problem, Weight
    from .variables import Values


def _analysis_failure(
    reason: LinearizationReason,
    detail: str,
    *,
    variable_name: str | None = None,
    time_length: int | None = None,
    tangent_width: int | None = None,
    reduced_width: int | None = None,
) -> TemporalAnalysis:
    return TemporalAnalysis(
        operator_eligible=False,
        direct_eligible=False,
        reason=reason,
        detail=detail,
        variable_name=variable_name,
        time_length=time_length,
        tangent_width=tangent_width,
        reduced_width=reduced_width,
    )


def analyze_temporal_problem(problem: Problem) -> TemporalAnalysis:  # noqa: PLR0911
    """Analyze declarations once without reading numeric tensor values."""
    optimized = tuple(spec for spec in problem.vars if spec.free_dim > 0)
    temporal = tuple(spec for spec in optimized if spec.time_axis is not None)
    if not temporal:
        return _analysis_failure(
            LinearizationReason.NO_TIME_VARIABLE,
            "problem has no free time-annotated variable",
        )
    if len(optimized) != 1 or len(temporal) != 1:
        names = tuple(spec.name for spec in optimized)
        return _analysis_failure(
            LinearizationReason.MULTIPLE_OPTIMIZED_VARIABLES,
            f"structured v1 requires exactly one free variable; found {names}",
        )

    spec = temporal[0]
    if not spec.temporal_mask_is_separable:
        return _analysis_failure(
            LinearizationReason.NONSEPARABLE_MASK,
            f"temporal variable {spec.name!r} mask differs across knots",
            variable_name=spec.name,
            time_length=spec.time_length,
            tangent_width=spec.temporal_tangent_width,
        )

    reduced_width = spec.temporal_reduced_width
    patterns: list[tuple[str, TemporalPattern]] = []
    missing_blocks: list[str] = []
    bandwidth = 0
    for item in problem.residuals:
        dependencies = problem._item_variable_reads[item.name]
        if spec.name not in dependencies:
            continue
        other_free = tuple(
            name for name in dependencies if name != spec.name and problem._vars_by_name[name].free_dim > 0
        )
        if other_free:
            return _analysis_failure(
                LinearizationReason.MIXED_OPTIMIZED_DEPENDENCY,
                f"residual {item.name!r} couples {spec.name!r} to free variables {other_free}",
                variable_name=spec.name,
                time_length=spec.time_length,
                tangent_width=spec.temporal_tangent_width,
                reduced_width=reduced_width,
            )
        declaration = getattr(item.residual, "temporal_structure", None)
        if not callable(declaration):
            return _analysis_failure(
                LinearizationReason.UNDECLARED_TEMPORAL_RESIDUAL,
                f"residual {item.name!r} depends on {spec.name!r} but declares no temporal structure",
                variable_name=spec.name,
                time_length=spec.time_length,
                tangent_width=spec.temporal_tangent_width,
                reduced_width=reduced_width,
            )
        pattern = declaration(spec.name)
        if not isinstance(pattern, TemporalPattern):
            raise TypeError(
                f"Residual {item.name!r} temporal_structure({spec.name!r}) must return "
                "TemporalPattern, not "
                f"{type(pattern).__name__}"
            )
        if pattern.rows * pattern.row_width != item.residual.dim:
            raise ValueError(
                f"Residual {item.name!r} temporal pattern rows*row_width must equal "
                f"dim={item.residual.dim}, got {pattern.rows}*{pattern.row_width}"
            )
        first = pattern.row_origin + pattern.offsets[0]
        last = pattern.row_origin + pattern.rows - 1 + pattern.offsets[-1]
        if first < 0 or last >= spec.time_length:
            raise ValueError(
                f"Residual {item.name!r} temporal pattern reaches knots [{first}, {last}] "
                f"outside [0, {spec.time_length})"
            )
        patterns.append((item.name, pattern))
        bandwidth = max(bandwidth, *(abs(value) for value in pattern.hessian_offsets))
        if not callable(getattr(item.residual, "temporal_jacobian_blocks", None)):
            missing_blocks.append(item.name)

    if missing_blocks:
        return TemporalAnalysis(
            operator_eligible=True,
            direct_eligible=False,
            reason=LinearizationReason.MISSING_TEMPORAL_BLOCKS,
            detail=f"residuals {tuple(missing_blocks)} declare support but no temporal numeric blocks",
            variable_name=spec.name,
            time_length=spec.time_length,
            tangent_width=spec.temporal_tangent_width,
            reduced_width=reduced_width,
            bandwidth=bandwidth,
            patterns=tuple(patterns),
        )
    return TemporalAnalysis(
        operator_eligible=True,
        direct_eligible=True,
        reason=LinearizationReason.ELIGIBLE_BANDED,
        detail=f"all residuals depending on {spec.name!r} provide temporal blocks",
        variable_name=spec.name,
        time_length=spec.time_length,
        tangent_width=spec.temporal_tangent_width,
        reduced_width=reduced_width,
        bandwidth=bandwidth,
        patterns=tuple(patterns),
    )


@dataclass(frozen=True)
class _StructuredTerm:
    row_slice: slice
    pattern: TemporalPattern
    blocks: tuple[tuple[int, torch.Tensor], ...]


@dataclass(frozen=True)
class StructuredNormal:
    """One robust-row-scaled physical-tangent temporal linearization."""

    gradient: torch.Tensor
    normal_diagonal: torch.Tensor
    normal: BlockBandedMatrix
    finite: torch.Tensor
    _residual_dim: int = field(repr=False)
    _terms: tuple[_StructuredTerm, ...] = field(repr=False)

    def _validate_vector(self, name: str, vector: torch.Tensor, size: int) -> None:
        if not isinstance(vector, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        expected = (*self.normal.batch_shape, size)
        if tuple(vector.shape) != expected:
            raise ValueError(f"{name} must have shape {expected}, got {tuple(vector.shape)}")
        if vector.dtype != self.gradient.dtype or vector.device != self.gradient.device:
            raise ValueError(f"{name} must preserve structured-normal dtype/device")

    def jvp(self, vector: torch.Tensor) -> torch.Tensor:
        """Apply the globally row-scaled J to a flat physical tangent."""
        self._validate_vector("vector", vector, self.normal.size)
        batch_shape = self.normal.batch_shape
        shaped = vector.reshape(*batch_shape, self.normal.time_length, self.normal.block_size)
        result = vector.new_zeros(*batch_shape, self._residual_dim)
        for term in self._terms:
            pattern = term.pattern
            rows = vector.new_zeros(*batch_shape, pattern.rows, pattern.row_width)
            for offset, block in term.blocks:
                start = pattern.row_origin + offset
                selected = shaped[..., start : start + pattern.rows, :]
                rows = rows + torch.einsum("...rwd,...rd->...rw", block, selected)
            result[..., term.row_slice] = rows.reshape(*batch_shape, pattern.rows * pattern.row_width)
        return result

    def vjp(self, cotangent: torch.Tensor) -> torch.Tensor:
        """Apply the globally row-scaled J transpose to a residual cotangent."""
        self._validate_vector("cotangent", cotangent, self._residual_dim)
        batch_shape = self.normal.batch_shape
        result = cotangent.new_zeros(*batch_shape, self.normal.time_length, self.normal.block_size)
        for term in self._terms:
            pattern = term.pattern
            rows = cotangent[..., term.row_slice].reshape(
                *batch_shape,
                pattern.rows,
                pattern.row_width,
            )
            for offset, block in term.blocks:
                start = pattern.row_origin + offset
                contribution = torch.einsum("...rwd,...rw->...rd", block, rows)
                result[..., start : start + pattern.rows, :] = (
                    result[..., start : start + pattern.rows, :] + contribution
                )
        return result.reshape(*batch_shape, self.normal.size)

    def normal_matvec(self, vector: torch.Tensor) -> torch.Tensor:
        """Apply ``J.T @ J`` without materializing dense J."""
        return self.vjp(self.jvp(vector))


def _broadcast_weight(weight: Weight, output: torch.Tensor) -> torch.Tensor:
    result = torch.as_tensor(weight, dtype=output.dtype, device=output.device)
    while result.ndim < output.ndim:
        result = result.unsqueeze(-1)
    return result


def _is_inactive(weight: Weight) -> bool:
    return isinstance(weight, Real) and float(weight) == 0.0


def _validate_runtime_weight(
    name: str,
    weight: Weight,
    batch_shape: tuple[int, ...],
    exemplar: torch.Tensor,
) -> None:
    if not isinstance(weight, torch.Tensor):
        return
    if tuple(weight.shape) not in ((), batch_shape):
        raise ValueError(
            f"Tensor weight for item {name!r} must be scalar or have exact batch "
            f"shape {batch_shape}, got {tuple(weight.shape)}"
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
    """Assemble exact lower bands and flat operators from temporal block hooks."""
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
        residual = problem._residual_with_context(
            values,
            batch_shape,
            ctx,
            weights,
            validate_runtime=validate_runtime,
        )
    if row_scale is None:
        row_scale = torch.ones_like(residual)
    expected_rows = (*batch_shape, problem.dim_total)
    for name, tensor in (("residual", residual), ("row_scale", row_scale)):
        if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != expected_rows:
            actual = tuple(tensor.shape) if isinstance(tensor, torch.Tensor) else type(tensor).__name__
            raise ValueError(f"{name} must have shape {expected_rows}, got {actual}")
        if tensor.dtype != exemplar.dtype or tensor.device != exemplar.device:
            raise ValueError(f"{name} must preserve values dtype/device")

    T = analysis.time_length
    d = analysis.reduced_width
    w = analysis.bandwidth
    bands = exemplar.new_zeros(*batch_shape, T, w + 1, d, d)
    gradient = exemplar.new_zeros(*batch_shape, T, d)
    terms: list[_StructuredTerm] = []
    patterns = dict(analysis.patterns)
    weighted_residual = residual * row_scale

    for item in problem.residuals:
        pattern = patterns.get(item.name)
        if pattern is None:
            continue
        weight = problem._weights_for(item, weights)
        if validate_runtime:
            _validate_runtime_weight(item.name, weight, batch_shape, exemplar)
        if _is_inactive(weight):
            continue
        hook = item.residual.temporal_jacobian_blocks
        raw = hook(ctx.restrict(item.residual.reads), spec.name)
        if not isinstance(raw, Mapping):
            raise TypeError(f"Residual {item.name!r} temporal_jacobian_blocks must return a mapping")
        if set(raw) != set(pattern.offsets):
            raise ValueError(
                f"Residual {item.name!r} temporal block offsets must be {pattern.offsets}, got {tuple(raw)}"
            )
        item_scale = row_scale[..., problem.row_offsets[item.name]].reshape(
            *batch_shape,
            pattern.rows,
            pattern.row_width,
            1,
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
            if create_graph:
                graph_inputs = (*values.values(), *problem.parameters.values())
                if any(value.requires_grad for value in graph_inputs) and not block.requires_grad:
                    raise ValueError(
                        f"Temporal block {(item.name, spec.name, offset)!r} cannot honor create_graph=True"
                    )
            weighted = block * _broadcast_weight(weight, block) * item_scale
            blocks.append((offset, weighted if create_graph else weighted.detach()))

        row = weighted_residual[..., problem.row_offsets[item.name]].reshape(
            *batch_shape,
            pattern.rows,
            pattern.row_width,
        )
        by_offset = dict(blocks)
        for offset, block in blocks:
            start = pattern.row_origin + offset
            contribution = torch.einsum("...rwd,...rw->...rd", block, row)
            gradient[..., start : start + pattern.rows, :] = (
                gradient[..., start : start + pattern.rows, :] + contribution
            )
        for later in pattern.offsets:
            for earlier in pattern.offsets:
                if later < earlier:
                    continue
                start = pattern.row_origin + later
                contribution = torch.einsum(
                    "...rwi,...rwj->...rij",
                    by_offset[later],
                    by_offset[earlier],
                )
                offset = later - earlier
                bands[..., start : start + pattern.rows, offset, :, :] = (
                    bands[..., start : start + pattern.rows, offset, :, :] + contribution
                )
        terms.append(_StructuredTerm(problem.row_offsets[item.name], pattern, tuple(blocks)))

    normal = BlockBandedMatrix(bands, w)
    flat_gradient = gradient.reshape(*batch_shape, T * d)
    normal_diagonal = normal.diagonal()
    finite = normal.finite & torch.isfinite(flat_gradient).all(dim=-1) & torch.isfinite(normal_diagonal).all(dim=-1)
    return StructuredNormal(
        gradient=flat_gradient,
        normal_diagonal=normal_diagonal,
        normal=normal,
        finite=finite,
        _residual_dim=problem.dim_total,
        _terms=tuple(terms),
    )


__all__ = ["StructuredNormal", "analyze_temporal_problem", "assemble_structured_normal"]
