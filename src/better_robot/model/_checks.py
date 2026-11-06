"""Structural checks of public tensor inputs: trailing shape, dtype, device and batch broadcast.

The one validator (decision record 0021): every public boundary calls `check_tensor` once per tensor argument
and `batch_shape` / `expand_batch` to align it with the model's value batch. Raw passes below trust the result.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ..exceptions import DeviceMismatchError, DtypeMismatchError, ShapeError

if TYPE_CHECKING:
    from .model import Model


def check_tensor(
    name: str, value: object, *, shape: tuple[int, ...], dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """``value`` as a tensor whose trailing axes are ``shape`` in the model's ``dtype`` on its ``device``."""
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value).__name__}")
    if value.ndim < len(shape) or tuple(value.shape[value.ndim - len(shape) :]) != shape:
        raise ShapeError(f"{name}.shape must end in {shape}, got {tuple(value.shape)}")
    if value.dtype not in (torch.float32, torch.float64):
        raise DtypeMismatchError(f"{name}.dtype={value.dtype} is unsupported; use torch.float32 or torch.float64")
    if value.dtype != dtype:
        raise DtypeMismatchError(
            f"{name}.dtype={value.dtype} != model.dtype={dtype}; cast {name} or call model.to(dtype={name}.dtype)"
        )
    if value.device != device:
        raise DeviceMismatchError(
            f"{name}.device={value.device} != model.device={device}; call model.to({name}.device) or "
            f"{name}.to(model.device)"
        )
    return value


def batch_shape(model: Model, name: str, value: torch.Tensor, event: tuple[int, ...]) -> tuple[int, ...]:
    """Broadcast of ``value``'s leading axes (before ``event``) with the model's value batch ``I...``."""
    batch = tuple(value.shape[: value.ndim - len(event)])
    for field in ("joint_placements", "body_inertias", "frame_placements"):
        values = getattr(model, field)
        value_batch = tuple(values.shape[:-2])
        try:
            batch = tuple(torch.broadcast_shapes(batch, value_batch))
        except RuntimeError as error:
            raise ShapeError(f"cannot broadcast {name} batch {batch} with {field} batch {value_batch}") from error
    return batch


def expand_batch(name: str, value: torch.Tensor, batch: tuple[int, ...], event: tuple[int, ...]) -> torch.Tensor:
    """``value`` broadcast to ``(*batch, *event)`` as a view."""
    try:
        return torch.broadcast_to(value, (*batch, *event))
    except RuntimeError as error:
        raise ShapeError(f"cannot broadcast {name} {tuple(value.shape)} to batch {batch}") from error


__all__ = ["batch_shape", "check_tensor", "expand_batch"]
