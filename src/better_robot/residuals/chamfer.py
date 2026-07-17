"""Fixed-shape masked Chamfer residuals for padded point clouds."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from ._point_cloud import _detached_nearest


class MaskedChamferResidual:
    """Bidirectional nearest-distance rows for padded per-frame clouds.

    Inputs follow the accepted padded convention: point tensors have shape
    ``(B..., frames, count, coordinates)`` and validity masks have shape
    ``(B..., frames, count)``.  Output rows are source-to-target distances
    followed by target-to-source distances, with invalid padding represented
    by finite zeros.  Set ``bidirectional=False`` for source-to-target only.

    Nearest-neighbour indices are selected from detached pairwise distances.
    Distances are reconstructed from the matched graph-carrying points, so
    gradients reach only the selected correspondence.  ``vertex_weights``,
    when named, multiply source-to-target residual amplitudes.
    """

    def __init__(
        self,
        frames: int,
        source_count: int,
        target_count: int,
        *,
        source: str = "points",
        target: str = "target_points",
        source_validity: str = "point_validity",
        target_validity: str = "target_validity",
        vertex_weights: str | None = None,
        bidirectional: bool = True,
        chunk_size: int = 4096,
        name: str = "masked_chamfer",
    ) -> None:
        for label, value in (
            ("frames", frames),
            ("source_count", source_count),
            ("target_count", target_count),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{label} must be a positive integer, got {value!r}")
        names = (source, target, source_validity, target_validity)
        if any(not isinstance(value, str) or not value for value in names):
            raise ValueError("point and validity context names must be non-empty strings")
        if len(set(names)) != len(names):
            raise ValueError("point and validity context names must be unique")
        if vertex_weights is not None and (not isinstance(vertex_weights, str) or not vertex_weights):
            raise ValueError("vertex_weights must be a non-empty context name or None")
        if vertex_weights is not None and vertex_weights in names:
            raise ValueError("vertex_weights must name a distinct context entry")
        if not isinstance(bidirectional, bool):
            raise TypeError("bidirectional must be bool")
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
            raise ValueError(f"chunk_size must be a positive integer, got {chunk_size!r}")
        if not isinstance(name, str) or not name:
            raise ValueError("name must be a non-empty string")

        self.frames = frames
        self.source_count = source_count
        self.target_count = target_count
        self.source = source
        self.target = target
        self.source_validity = source_validity
        self.target_validity = target_validity
        self.vertex_weights = vertex_weights
        self.bidirectional = bidirectional
        self.chunk_size = chunk_size
        self.name = name
        self.reads = (*names, *((vertex_weights,) if vertex_weights is not None else ()))
        rows_per_frame = source_count + (target_count if bidirectional else 0)
        self.dim = frames * rows_per_frame

    @staticmethod
    def _require_shape(tensor: torch.Tensor, suffix: tuple[int, ...], label: str) -> None:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{label} must be a torch.Tensor")
        if tensor.ndim < len(suffix) or tuple(tensor.shape[-len(suffix) :]) != suffix:
            raise ValueError(f"{label} must end in {suffix}, got {tuple(tensor.shape)}")

    def __call__(self, ctx: Mapping[str, Any]) -> torch.Tensor:
        source = ctx[self.source]
        target = ctx[self.target]
        source_validity = ctx[self.source_validity]
        target_validity = ctx[self.target_validity]
        self._require_shape(source, (self.frames, self.source_count, 3), self.source)
        self._require_shape(target, (self.frames, self.target_count, 3), self.target)
        self._require_shape(source_validity, (self.frames, self.source_count), self.source_validity)
        self._require_shape(target_validity, (self.frames, self.target_count), self.target_validity)

        forward = _detached_nearest(
            source,
            target,
            query_validity=source_validity,
            reference_validity=target_validity,
            chunk_size=self.chunk_size,
        ).distance
        if self.vertex_weights is not None:
            weights = ctx[self.vertex_weights]
            self._require_shape(weights, (self.frames, self.source_count), self.vertex_weights)
            if not weights.is_floating_point():
                raise TypeError(f"{self.vertex_weights} must use a floating dtype")
            if weights.dtype != forward.dtype or weights.device != forward.device:
                raise ValueError(f"{self.vertex_weights} must share point dtype/device")
            forward = forward * weights

        rows = [forward]
        if self.bidirectional:
            reverse = _detached_nearest(
                target,
                source,
                query_validity=target_validity,
                reference_validity=source_validity,
                chunk_size=self.chunk_size,
            ).distance
            rows.append(reverse)
        combined = torch.cat(rows, dim=-1)
        return combined.reshape(*combined.shape[:-2], self.dim)


__all__ = ["MaskedChamferResidual"]
