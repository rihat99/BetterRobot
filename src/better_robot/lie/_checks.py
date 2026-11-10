"""The one input validator of the library (contract 06, C-ERR-1): `lie`, `spatial` and the `model` boundaries call it.

It reads types, trailing shapes, dtypes and devices, never tensor values.
"""

from __future__ import annotations

import torch

from ..exceptions import DeviceMismatchError, DtypeMismatchError, ShapeError

_DTYPES = (torch.float32, torch.float64)


def check_dtype(dtype: object) -> torch.dtype:
    """``dtype`` once it is float32 or float64, the dtypes the library computes in; else ``DtypeMismatchError``."""
    if dtype not in _DTYPES:
        raise DtypeMismatchError(f"dtype={dtype!r} is not float32 or float64; pass torch.float32 or torch.float64")
    return dtype


def check_tensor(
    name: str,
    value: object,
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
    owner: str = "model",
) -> torch.Tensor:
    """``value`` once it is a float32 or float64 tensor whose last axes are ``shape`` (``()`` accepts any), in
    ``dtype`` on ``device`` when they are given (``owner`` names whose they are in the message).

    ``TypeError`` for a value that is not a ``torch.Tensor`` (the Python type error; the error tree has no node for
    it, C-ERR-2), ``DtypeMismatchError`` for another dtype, ``ShapeError`` for other trailing axes,
    ``DeviceMismatchError`` for another device. Every message names the fix.
    """
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value).__name__}")
    if value.dtype not in _DTYPES:
        raise DtypeMismatchError(
            f"{name}.dtype={value.dtype} is not float32 or float64; pass {name}.float() or {name}.double()"
        )
    if value.ndim < len(shape) or tuple(value.shape[value.ndim - len(shape) :]) != shape:
        raise ShapeError(f"{name} must have shape (..., {', '.join(map(str, shape))}), got {tuple(value.shape)}")
    if dtype is not None and value.dtype != dtype:
        raise DtypeMismatchError(f"{name}.dtype={value.dtype} != {owner}.dtype={dtype}; pass {name}.to({dtype})")
    if device is not None and value.device != device:
        raise DeviceMismatchError(f"{name}.device={value.device} != {owner}.device={device}; pass {name}.to({device})")
    return value


def check_tensors(**tensors: tuple[object, tuple[int, ...]]) -> None:
    """`check_tensor` on each ``name=(value, shape)``, every value in the first value's dtype on its device, and the
    leading (batch) axes before each ``shape`` broadcast together; else ``ShapeError`` naming both arguments."""
    (first_name, (first, first_shape)), *others = tensors.items()
    reference = check_tensor(first_name, first, shape=first_shape)
    batches = {first_name: tuple(reference.shape[: reference.ndim - len(first_shape)])}
    for name, (value, shape) in others:
        checked = check_tensor(
            name, value, shape=shape, dtype=reference.dtype, device=reference.device, owner=first_name
        )
        batches[name] = tuple(checked.shape[: checked.ndim - len(shape)])
    check_broadcast(**batches)


def check_broadcast(**batches: tuple[int, ...]) -> None:
    """``ShapeError`` naming two arguments whose batch axes ``name=shape`` do not broadcast together."""
    named = list(batches.items())
    for index, (name, batch) in enumerate(named):
        for earlier, earlier_batch in named[:index]:
            if any(a != b and 1 not in (a, b) for a, b in zip(reversed(batch), reversed(earlier_batch))):
                raise ShapeError(
                    f"{earlier} has batch axes {tuple(earlier_batch)} and {name} {tuple(batch)}, which do not "
                    "broadcast; give both the same leading axes (size 1 repeats)"
                )
