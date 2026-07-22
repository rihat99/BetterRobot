"""Fixed-shape masked Chamfer residuals for padded point clouds."""

from __future__ import annotations

from numbers import Real

import torch

from .._validation import check_tensor
from ._point_cloud import _detached_nearest, _validate_clouds
from .base import Residual, Weight
from .nodes import Node
from .utils import VariableLike as _VariableLike, value, variables

_TensorInput = _VariableLike | Node | torch.Tensor


def _input_value(input_value: _TensorInput, name: str) -> torch.Tensor:
    return check_tensor(name, input_value.value() if isinstance(input_value, Node) else value(input_value, name))


class MaskedChamferResidual(Residual):
    """Bidirectional nearest-distance rows for padded per-frame clouds.

    Point tensors end in ``(frames, count, 3)`` and validity masks end in
    ``(frames, count)``. Invalid padding produces finite zero rows. Nearest
    indices are detached while distances retain gradients to the selected
    points. Inputs may be Nodes, Variables, or bare construction-time tensors;
    static Variables are updatable through ``Problem.update()``.
    ``vertex_weights`` are non-negative detached confidence values whose safe
    square roots scale forward rows, making their L2 objective contribution
    linear in confidence. They are distinct from outer objective ``weight``.
    """

    def __init__(
        self,
        source: _VariableLike | Node | torch.Tensor,
        target: _VariableLike | Node | torch.Tensor,
        source_validity: _VariableLike | Node | torch.Tensor,
        target_validity: _VariableLike | Node | torch.Tensor,
        *,
        vertex_weights: _VariableLike | Node | torch.Tensor | None = None,
        bidirectional: bool = True,
        chunk_size: int = 4096,
        weight: Real | torch.Tensor = 1.0,
        row_weight: Weight | Real | torch.Tensor = 1.0,
        kernel: object | None = None,
        name: str = "masked_chamfer",
    ) -> None:
        source_tensor = check_tensor("source", _input_value(source, "source"), floating=True)
        target_tensor = check_tensor(
            "target",
            _input_value(target, "target"),
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
            _input_value(source_validity, "source_validity"),
            shape=(frames, source_count),
            dtype=torch.bool,
            device=source_tensor.device,
        )
        check_tensor(
            "target_validity",
            _input_value(target_validity, "target_validity"),
            shape=(frames, target_count),
            dtype=torch.bool,
            device=source_tensor.device,
        )
        if vertex_weights is not None:
            check_tensor(
                "vertex_weights",
                _input_value(vertex_weights, "vertex_weights"),
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
        inputs = (source, target, source_validity, target_validity, vertex_weights)
        self.nodes = tuple(
            {id(input_value): input_value for input_value in inputs if isinstance(input_value, Node)}.values()
        )
        self.frames = frames
        self.source_count = source_count
        self.target_count = target_count
        self.bidirectional = bidirectional
        self.chunk_size = chunk_size
        rows_per_frame = source_count + (target_count if bidirectional else 0)
        super().__init__(
            *variables(*inputs),
            dim=frames * rows_per_frame,
            weight=weight,
            row_weight=row_weight,
            kernel=kernel,
            name=name,
        )

    def error(self) -> torch.Tensor:
        source, target, source_validity, target_validity = _validate_clouds(
            _input_value(self.source, "source"),
            _input_value(self.target, "target"),
            _input_value(self.source_validity, "source_validity"),
            _input_value(self.target_validity, "target_validity"),
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
                _input_value(self.vertex_weights, "vertex_weights"),
                shape=(self.frames, self.source_count),
                floating=True,
                dtype=forward.dtype,
                device=forward.device,
            )
            confidence = weights.detach()
            positive = confidence > 0.0
            safe_confidence = torch.where(positive, confidence, torch.ones_like(confidence))
            confidence_scale = torch.where(positive, torch.sqrt(safe_confidence), torch.zeros_like(confidence))
            forward = forward * confidence_scale

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
