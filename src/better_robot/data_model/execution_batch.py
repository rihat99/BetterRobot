"""Frozen flat-execution-batch ABI for whole-pass kernels.

Inputs retain their unique physical batches.  ``batch_indices`` maps every
execution element back to the corresponding input row, avoiding stride-zero
broadcast views and physical repetition of shared model values.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Sequence

import torch

from ..exceptions import ShapeError


@dataclass(frozen=True)
class ExecutionInput:
    """One unique flattened input plus its execution-to-input row map."""

    tensor: torch.Tensor
    batch_shape: tuple[int, ...]
    event_shape: tuple[int, ...]
    batch_indices: torch.Tensor

    def gather(self) -> torch.Tensor:
        """Materialize execution rows; intended for torch oracles and tests."""

        return self.tensor.index_select(0, self.batch_indices)

    def reduce_gradient(self, execution_gradient: torch.Tensor) -> torch.Tensor:
        """Reduce per-execution gradients to this input's unique batch rows."""

        expected = (self.batch_indices.numel(), *self.event_shape)
        if tuple(execution_gradient.shape) != expected:
            raise ShapeError(
                f"execution gradient has shape {tuple(execution_gradient.shape)}, "
                f"expected {expected}"
            )
        reduced = execution_gradient.new_zeros((self.tensor.shape[0], *self.event_shape))
        reduced.index_add_(0, self.batch_indices, execution_gradient)
        return reduced.reshape(*self.batch_shape, *self.event_shape)


@dataclass(frozen=True)
class ExecutionBatch:
    """Broadcast batch flattened to ``E`` with one index map per input."""

    batch_shape: tuple[int, ...]
    size: int
    q: ExecutionInput
    values: tuple[ExecutionInput, ...]

    def unflatten(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.shape[0] != self.size:
            raise ShapeError(
                f"flat output has leading size {tensor.shape[0]}, expected E={self.size}"
            )
        return tensor.reshape(*self.batch_shape, *tensor.shape[1:])


def _flatten_input(
    tensor: torch.Tensor,
    event_ndims: int,
    execution_shape: tuple[int, ...],
) -> ExecutionInput:
    if event_ndims < 0 or event_ndims > tensor.ndim:
        raise ShapeError(
            f"event_ndims={event_ndims} is invalid for tensor shape {tuple(tensor.shape)}"
        )
    batch_shape = tuple(tensor.shape[: tensor.ndim - event_ndims])
    event_shape = tuple(tensor.shape[tensor.ndim - event_ndims :]) if event_ndims else ()
    unique_rows = prod(batch_shape) if batch_shape else 1
    flat = tensor.reshape(unique_rows, *event_shape)

    pad = len(execution_shape) - len(batch_shape)
    if pad < 0:
        raise ShapeError(
            f"input batch {batch_shape} has more axes than execution batch {execution_shape}"
        )
    padded = (1,) * pad + batch_shape
    for source, target in zip(padded, execution_shape):
        if source not in (1, target):
            raise ShapeError(
                f"input batch shape {batch_shape} cannot broadcast to {execution_shape}"
            )

    ids = torch.arange(unique_rows, dtype=torch.int64, device=tensor.device)
    ids = ids.reshape(padded or ())
    batch_indices = ids.expand(execution_shape or ()).reshape(-1)
    return ExecutionInput(flat, batch_shape, event_shape, batch_indices)


def flatten_execution_batch(
    q: torch.Tensor,
    value_tensors: Sequence[torch.Tensor] = (),
    *,
    value_event_ndims: Sequence[int] = (),
) -> ExecutionBatch:
    """Create the stable flat-``E`` ABI for ``q`` and model-value tensors.

    ``q`` always has one event dimension.  ``value_event_ndims`` describes
    the trailing, non-batch rank of each value tensor (for example, 2 for
    joint placements shaped ``(..., njoints, 7)``).
    """

    if q.ndim < 1:
        raise ShapeError("q must have a trailing configuration dimension")
    if len(value_tensors) != len(value_event_ndims):
        raise ShapeError(
            "value_tensors and value_event_ndims must have the same length"
        )
    devices = {q.device, *(tensor.device for tensor in value_tensors)}
    if len(devices) != 1:
        raise ShapeError("all execution-batch inputs must be on the same device")

    batch_shapes = [tuple(q.shape[:-1])]
    for tensor, event_ndims in zip(value_tensors, value_event_ndims):
        if event_ndims < 0 or event_ndims > tensor.ndim:
            raise ShapeError(
                f"event_ndims={event_ndims} is invalid for value shape {tuple(tensor.shape)}"
            )
        batch_shapes.append(tuple(tensor.shape[: tensor.ndim - event_ndims]))
    try:
        execution_shape = tuple(torch.broadcast_shapes(*batch_shapes))
    except RuntimeError as exc:
        joined = ", ".join(str(shape) for shape in batch_shapes)
        raise ShapeError(f"input batch shapes do not broadcast: {joined}") from exc

    q_view = _flatten_input(q, 1, execution_shape)
    value_views = tuple(
        _flatten_input(tensor, event_ndims, execution_shape)
        for tensor, event_ndims in zip(value_tensors, value_event_ndims)
    )
    size = prod(execution_shape) if execution_shape else 1
    return ExecutionBatch(execution_shape, size, q_view, value_views)


__all__ = [
    "ExecutionBatch",
    "ExecutionInput",
    "flatten_execution_batch",
]
