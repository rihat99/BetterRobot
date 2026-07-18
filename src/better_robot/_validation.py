"""Small shared checks for public tensor boundaries."""

from __future__ import annotations

import torch

from .exceptions import DeviceMismatchError, DtypeMismatchError, ShapeError


def check_tensor(
    name: str,
    value: object,
    *,
    shape: tuple[int, ...] | None = None,
    floating: bool = False,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Return a tensor after checking its optional trailing-shape contract."""

    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value).__name__}")
    if shape is not None and (value.ndim < len(shape) or tuple(value.shape[-len(shape) :]) != shape):
        raise ShapeError(f"{name}.shape must end in {shape}, got {tuple(value.shape)}")
    if floating and not value.is_floating_point():
        raise DtypeMismatchError(f"{name}.dtype must be floating, got {value.dtype}")
    if dtype is not None and value.dtype != dtype:
        raise DtypeMismatchError(f"{name}.dtype must be {dtype}, got {value.dtype}")
    if device is not None and value.device != device:
        raise DeviceMismatchError(f"{name}.device must be {device}, got {value.device}")
    return value


__all__ = ["check_tensor"]
