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
            raise ShapeError(f"execution gradient has shape {tuple(execution_gradient.shape)}, expected {expected}")
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
            raise ShapeError(f"flat output has leading size {tensor.shape[0]}, expected E={self.size}")
        return tensor.reshape(*self.batch_shape, *tensor.shape[1:])


def broadcast_execution_batch_shape(
    q: torch.Tensor,
    value_tensors: Sequence[torch.Tensor] = (),
    *,
    value_event_ndims: Sequence[int] = (),
    value_names: Sequence[str] = (),
) -> tuple[int, ...]:
    """Return the right-aligned broadcast of query and value batches.

    This is the allocation-free torch-lane counterpart of
    :func:`flatten_execution_batch`.  Event dimensions are excluded from
    broadcasting.  In particular, a per-person value table ``(B, N, D)``
    does *not* implicitly acquire a trajectory axis when paired with
    ``q.shape == (B, T, nq)``; callers must pass ``(B, 1, N, D)``.
    """

    if q.ndim < 1:
        raise ShapeError("q must have a trailing configuration dimension")
    if len(value_tensors) != len(value_event_ndims):
        raise ShapeError("value_tensors and value_event_ndims must have the same length")
    if value_names and len(value_names) != len(value_tensors):
        raise ShapeError("value_names and value_tensors must have the same length")

    devices = {q.device, *(tensor.device for tensor in value_tensors)}
    if len(devices) != 1:
        raise ShapeError("all execution-batch inputs must be on the same device")

    q_batch = tuple(q.shape[:-1])
    execution_shape = q_batch
    for index, (tensor, event_ndims) in enumerate(zip(value_tensors, value_event_ndims, strict=True)):
        if event_ndims < 0 or event_ndims > tensor.ndim:
            raise ShapeError(f"event_ndims={event_ndims} is invalid for value shape {tuple(tensor.shape)}")
        value_batch = tuple(tensor.shape[: tensor.ndim - event_ndims])
        name = value_names[index] if value_names else f"value_tensors[{index}]"
        try:
            execution_shape = tuple(torch.broadcast_shapes(execution_shape, value_batch))
        except RuntimeError as exc:
            left_name = "q" if execution_shape == q_batch else "execution"
            raise ShapeError(
                f"cannot broadcast {left_name} batch {execution_shape} with "
                f"{name} batch {value_batch} (input batch shapes do not "
                "broadcast) — model-value batch dims must be "
                "right-aligned-broadcastable against the q batch. For (B, T) "
                "trajectories with per-person values, add a singleton time axis "
                "before the event dimensions, e.g. placements shaped "
                "(B, 1, njoints, 7)."
            ) from exc
    return execution_shape


def broadcast_to_execution_batch(
    tensor: torch.Tensor,
    batch_shape: tuple[int, ...],
    event_shape: tuple[int, ...],
    *,
    name: str,
) -> torch.Tensor:
    """Broadcast ``tensor`` to ``(*batch_shape, *event_shape)`` as a view."""

    trailing = tuple(tensor.shape[-len(event_shape) :]) if event_shape else ()
    if len(event_shape) > tensor.ndim or trailing != event_shape:
        actual = tuple(tensor.shape)
        raise ShapeError(f"{name} has shape {actual}; expected trailing event shape {event_shape}")
    source_batch = tuple(tensor.shape[: tensor.ndim - len(event_shape)])
    try:
        return torch.broadcast_to(tensor, (*batch_shape, *event_shape))
    except RuntimeError as exc:
        raise ShapeError(f"{name} batch {source_batch} cannot broadcast to execution batch {batch_shape}") from exc


def _flatten_input(
    tensor: torch.Tensor,
    event_ndims: int,
    execution_shape: tuple[int, ...],
) -> ExecutionInput:
    batch_shape = tuple(tensor.shape[: tensor.ndim - event_ndims])
    event_shape = tuple(tensor.shape[tensor.ndim - event_ndims :]) if event_ndims else ()
    unique_rows = prod(batch_shape) if batch_shape else 1
    flat = tensor.reshape(unique_rows, *event_shape)
    padded = (1,) * (len(execution_shape) - len(batch_shape)) + batch_shape
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

    execution_shape = broadcast_execution_batch_shape(
        q,
        value_tensors,
        value_event_ndims=value_event_ndims,
    )

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
    "broadcast_execution_batch_shape",
    "broadcast_to_execution_batch",
    "flatten_execution_batch",
]
