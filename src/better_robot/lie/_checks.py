"""The one input check of the `lie` boundary: a floating tensor with the trailing shape a function expects."""

from __future__ import annotations

import torch

from ..exceptions import DtypeMismatchError, ShapeError


def check_tensor(name: str, value: object, *, trailing: tuple[int, ...]) -> torch.Tensor:
    """``value`` once it is a floating tensor whose last axes are ``trailing``.

    ``TypeError`` for a value that is not a ``torch.Tensor`` (the Python type error; the error tree has no node for
    it), ``DtypeMismatchError`` for an integer or boolean dtype, ``ShapeError`` for other trailing axes.
    """
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value).__name__}")
    if not value.is_floating_point():
        raise DtypeMismatchError(f"{name}.dtype={value.dtype} is not floating; pass {name}.float() or {name}.double()")
    if tuple(value.shape[value.ndim - len(trailing) :]) != trailing or value.ndim < len(trailing):
        raise ShapeError(f"{name} must have shape (..., {', '.join(map(str, trailing))}); got {tuple(value.shape)}")
    return value
