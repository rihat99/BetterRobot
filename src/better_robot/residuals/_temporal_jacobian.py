"""Dependency-light helpers for temporal residual Jacobian blocks."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from .structure import TemporalPattern


def temporal_free_indices(
    ctx: Mapping[str, object],
    variable_name: str,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Return one knot's reduced tangent indices from an evaluation context."""
    getter = getattr(ctx, "temporal_free_indices", None)
    if getter is None:
        raise TypeError("temporal analytic blocks require a context with temporal_free_indices(variable_name)")
    indices = getter(variable_name)
    if not isinstance(indices, torch.Tensor) or indices.ndim != 1:
        raise TypeError("temporal_free_indices must return a one-dimensional tensor")
    return indices.to(device=device, dtype=torch.int64)


def dense_temporal_jacobian(
    pattern: TemporalPattern,
    blocks: Mapping[int, torch.Tensor],
    *,
    horizon: int,
) -> torch.Tensor:
    """Densify ``offset -> (..., rows, row_width, d)`` blocks.

    This is the dense-oracle path used by residual ``jacobian_blocks`` methods
    and their parity tests. Structured assembly consumes the blocks directly.
    """
    dense: torch.Tensor | None = None
    for offset in pattern.offsets:
        block = blocks[offset]
        first_knot = pattern.row_origin + offset
        targets = torch.arange(pattern.rows, device=block.device) + first_knot
        selector = torch.nn.functional.one_hot(targets, num_classes=horizon).to(
            device=block.device,
            dtype=block.dtype,
        )
        contribution = torch.einsum("...rwd,rt->...rwtd", block, selector)
        dense = contribution if dense is None else dense + contribution

    assert dense is not None
    return dense.reshape(
        *dense.shape[:-4],
        pattern.rows * pattern.row_width,
        horizon * dense.shape[-1],
    )


__all__ = ["dense_temporal_jacobian", "temporal_free_indices"]
