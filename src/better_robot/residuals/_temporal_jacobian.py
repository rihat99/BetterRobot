"""Dependency-light helpers for temporal residual Jacobian blocks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from .utils import RobotVariableLike as _TemporalVariable
from .structure import TemporalPattern


def dense_temporal_jacobian(
    pattern: TemporalPattern,
    blocks: Mapping[int, torch.Tensor],
    *,
    horizon: int,
) -> torch.Tensor:
    """Densify ``offset -> (..., rows, row_width, d)`` blocks."""
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


def dense_temporal_residual(residual: Any, variable: _TemporalVariable, horizon: int) -> tuple[torch.Tensor, ...]:
    pattern = residual.temporal_structure(variable)
    assert pattern is not None
    return (dense_temporal_jacobian(pattern, residual.temporal_jacobian_blocks(variable), horizon=horizon),)


__all__ = ["dense_temporal_jacobian"]
