"""Input checks of the public ``spatial`` functions: types, trailing shapes, dtypes and devices, never tensor values."""

from __future__ import annotations

import torch

from ..exceptions import DeviceMismatchError, DtypeMismatchError, ShapeError


def check_tensors(**tensors: tuple[object, tuple[int, ...]]) -> None:
    """Raise unless each ``name=(tensor, shape)`` is a floating tensor ending in ``shape``, all on one dtype and device.

    ``TypeError`` for a value that is not a ``torch.Tensor``; ``DtypeMismatchError`` for a dtype that is not floating
    or differs from the first tensor's; ``ShapeError`` for a trailing shape other than ``shape`` (``()`` accepts
    any); ``DeviceMismatchError`` for a device other than the first tensor's. The messages name the fix.
    """
    for name, (tensor, shape) in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a floating torch.Tensor, got {type(tensor).__name__}")
        if not tensor.is_floating_point():
            raise DtypeMismatchError(
                f"{name}.dtype={tensor.dtype} is not floating; pass {name}.float() or {name}.double()"
            )
        if tensor.ndim < len(shape) or tuple(tensor.shape[tensor.ndim - len(shape) :]) != shape:
            trailing = ", ".join(str(size) for size in shape)
            raise ShapeError(f"{name} must have shape (..., {trailing}), got {tuple(tensor.shape)}")
    (first_name, (first, _)), *others = tensors.items()
    for name, (tensor, _) in others:
        if tensor.dtype != first.dtype:
            raise DtypeMismatchError(
                f"{name}.dtype={tensor.dtype} differs from {first_name}.dtype={first.dtype}; "
                f"pass {name}.to({first.dtype})"
            )
        if tensor.device != first.device:
            raise DeviceMismatchError(
                f"{name} is on {tensor.device} but {first_name} is on {first.device}; pass {name}.to({first.device})"
            )
