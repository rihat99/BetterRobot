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
    if isinstance(horizon, bool) or not isinstance(horizon, int):
        raise TypeError("horizon must be an int")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if set(blocks) != set(pattern.offsets):
        raise ValueError(
            "temporal block keys must exactly match pattern offsets; "
            f"expected {pattern.offsets}, got {tuple(sorted(blocks))}"
        )

    dense: torch.Tensor | None = None
    batch_shape: tuple[int, ...] | None = None
    reduced_width: int | None = None
    for offset in pattern.offsets:
        block = blocks[offset]
        if not isinstance(block, torch.Tensor) or block.ndim < 3:
            raise TypeError("temporal Jacobian blocks must be tensors with at least three dimensions")
        if tuple(block.shape[-3:-1]) != (pattern.rows, pattern.row_width):
            raise ValueError(
                "temporal Jacobian block has wrong row shape; expected "
                f"(..., {pattern.rows}, {pattern.row_width}, d), got {tuple(block.shape)}"
            )
        current_batch = tuple(block.shape[:-3])
        current_width = int(block.shape[-1])
        if batch_shape is None:
            batch_shape = current_batch
            reduced_width = current_width
        elif current_batch != batch_shape or current_width != reduced_width:
            raise ValueError("all temporal Jacobian blocks must share batch shape and reduced width")

        first_knot = pattern.row_origin + offset
        last_knot = pattern.rows - 1 + pattern.row_origin + offset
        if first_knot < 0 or last_knot >= horizon:
            raise ValueError(
                f"temporal offset {offset} addresses knots [{first_knot}, {last_knot}] outside horizon {horizon}"
            )
        targets = torch.arange(pattern.rows, device=block.device) + first_knot
        selector = torch.nn.functional.one_hot(targets, num_classes=horizon).to(
            device=block.device,
            dtype=block.dtype,
        )
        contribution = torch.einsum("...rwd,rt->...rwtd", block, selector)
        dense = contribution if dense is None else dense + contribution

    assert dense is not None and batch_shape is not None and reduced_width is not None
    return dense.reshape(
        *batch_shape,
        pattern.rows * pattern.row_width,
        horizon * reduced_width,
    )


__all__ = ["dense_temporal_jacobian", "temporal_free_indices"]
