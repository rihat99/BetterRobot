"""Fixed-shape masked Chamfer residuals for padded point clouds."""

from __future__ import annotations

from numbers import Real

import torch

from .._validation import check_tensor
from ._point_cloud import _detached_nearest, _validate_clouds
from ._variables import VariableLike as _VariableLike, value, variables
from .base import Residual, Weight


class MaskedChamferResidual(Residual):
    """Bidirectional nearest-distance rows for padded per-frame clouds.

    Point tensors end in ``(frames, count, 3)`` and validity masks end in
    ``(frames, count)``. Invalid padding produces finite zero rows. Nearest
    indices are detached while distances retain gradients to the selected
    points. ``vertex_weights`` are domain confidence multipliers, distinct
    from the optimizer-level ``weight`` inherited from :class:`Residual`.
    """

    def __init__(
        self,
        source: _VariableLike | torch.Tensor,
        target: _VariableLike | torch.Tensor,
        source_validity: _VariableLike | torch.Tensor,
        target_validity: _VariableLike | torch.Tensor,
        *,
        vertex_weights: _VariableLike | torch.Tensor | None = None,
        bidirectional: bool = True,
        chunk_size: int = 4096,
        weight: Weight | Real | torch.Tensor = 1.0,
        kernel: object | None = None,
        name: str = "masked_chamfer",
    ) -> None:
        source_tensor = check_tensor("source", value(source, "source"), floating=True)
        target_tensor = check_tensor(
            "target",
            value(target, "target"),
            floating=True,
            dtype=source_tensor.dtype,
            device=source_tensor.device,
        )
        if source_tensor.ndim < 3 or source_tensor.shape[-1] != 3:
            raise ValueError(f"source must end in (frames, count, 3), got {tuple(source_tensor.shape)}")
        if target_tensor.ndim < 3 or target_tensor.shape[-1] != 3:
            raise ValueError(f"target must end in (frames, count, 3), got {tuple(target_tensor.shape)}")
        frames = int(source_tensor.shape[-3])
        source_count = int(source_tensor.shape[-2])
        target_count = int(target_tensor.shape[-2])
        if frames <= 0 or source_count <= 0 or target_count <= 0:
            raise ValueError("point-cloud frame and padded point axes must be non-empty")
        if target_tensor.shape[-3] != frames:
            raise ValueError(
                f"source and target must have the same frames axis, got {frames} and {target_tensor.shape[-3]}"
            )
        check_tensor(
            "source_validity",
            value(source_validity, "source_validity"),
            shape=(frames, source_count),
            dtype=torch.bool,
            device=source_tensor.device,
        )
        check_tensor(
            "target_validity",
            value(target_validity, "target_validity"),
            shape=(frames, target_count),
            dtype=torch.bool,
            device=source_tensor.device,
        )
        if vertex_weights is not None:
            check_tensor(
                "vertex_weights",
                value(vertex_weights, "vertex_weights"),
                shape=(frames, source_count),
                floating=True,
                dtype=source_tensor.dtype,
                device=source_tensor.device,
            )
        if not isinstance(bidirectional, bool):
            raise TypeError("bidirectional must be bool")
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
            raise ValueError(f"chunk_size must be a positive integer, got {chunk_size!r}")

        self.source = source
        self.target = target
        self.source_validity = source_validity
        self.target_validity = target_validity
        self.vertex_weights = vertex_weights
        self.frames = frames
        self.source_count = source_count
        self.target_count = target_count
        self.bidirectional = bidirectional
        self.chunk_size = chunk_size
        rows_per_frame = source_count + (target_count if bidirectional else 0)
        super().__init__(
            *variables(source, target, source_validity, target_validity, vertex_weights),
            dim=frames * rows_per_frame,
            weight=weight,
            kernel=kernel,
            name=name,
        )

    def error(self) -> torch.Tensor:
        source, target, source_validity, target_validity = _validate_clouds(
            value(self.source, "source"),
            value(self.target, "target"),
            value(self.source_validity, "source_validity"),
            value(self.target_validity, "target_validity"),
        )
        forward = _detached_nearest(
            source,
            target,
            query_validity=source_validity,
            reference_validity=target_validity,
            chunk_size=self.chunk_size,
        ).distance
        if self.vertex_weights is not None:
            weights = check_tensor(
                "vertex_weights",
                value(self.vertex_weights, "vertex_weights"),
                shape=(self.frames, self.source_count),
                floating=True,
                dtype=forward.dtype,
                device=forward.device,
            )
            forward = forward * weights

        rows = [forward]
        if self.bidirectional:
            rows.append(
                _detached_nearest(
                    target,
                    source,
                    query_validity=target_validity,
                    reference_validity=source_validity,
                    chunk_size=self.chunk_size,
                ).distance
            )
        combined = torch.cat(rows, dim=-1)
        return combined.reshape(*combined.shape[:-2], self.dim)


__all__ = ["MaskedChamferResidual"]
