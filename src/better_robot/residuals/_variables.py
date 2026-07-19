"""Structural variable protocols and validation shared by residual modules."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch

from .._validation import check_tensor
from ..data_model.model import Model


@runtime_checkable
class ValueLike(Protocol):
    tensor: torch.Tensor
    name: str
    trainable: bool


@runtime_checkable
class VariableLike(ValueLike, Protocol):
    def tangent_dim(self) -> int: ...
    def difference(self, other: torch.Tensor) -> torch.Tensor: ...


@runtime_checkable
class RobotValueLike(ValueLike, Protocol):
    model: Model


@runtime_checkable
class ShapedVariableLike(VariableLike, Protocol):
    shape: tuple[int, ...]
    time_axis: int | None
    free_indices: torch.Tensor


@runtime_checkable
class RobotLike(ShapedVariableLike, Protocol):
    model: Model


class RobotVariableLike(RobotLike, Protocol):
    time_length: int
    temporal_free_indices: torch.Tensor

    def gather_tangent(self, full: torch.Tensor) -> torch.Tensor: ...


TemporalLike = ShapedVariableLike


class TemporalVariableLike(ShapedVariableLike, Protocol):
    time_length: int


def value(value: VariableLike | torch.Tensor, label: str) -> torch.Tensor:
    return check_tensor(label, value.tensor if isinstance(value, VariableLike) else value)


def variables(*values: object) -> tuple[VariableLike, ...]:
    return tuple(value for value in values if isinstance(value, VariableLike))


def static_value(
    value: torch.Tensor | VariableLike,
    *,
    name: str,
) -> tuple[torch.Tensor, tuple[VariableLike, ...]]:
    if isinstance(value, VariableLike):
        if value.trainable:
            raise ValueError(f"{name} Variable must have trainable=False")
        return value.tensor, (value,)
    return check_tensor(name, value, floating=True), ()


def current_value(
    value: torch.Tensor | VariableLike,
    exemplar: torch.Tensor,
    *,
    name: str,
    preserve: bool = True,
) -> torch.Tensor:
    tensor = value.tensor if isinstance(value, VariableLike) else value
    if tensor.dtype != exemplar.dtype or tensor.device != exemplar.device:
        if not preserve:
            raise ValueError(f"{name} must share working dtype/device with the optimized variable")
        raise ValueError(
            f"{name} must preserve working dtype/device {exemplar.dtype}/{exemplar.device}, "
            f"got {tensor.dtype}/{tensor.device}"
        )
    return tensor


def matches(variable: ValueLike | str, expected: ValueLike) -> bool:
    return variable is expected or variable == expected.name


def require_robot(variable: RobotVariableLike, name: str, *, temporal: bool) -> None:
    if not isinstance(variable, RobotLike):
        raise TypeError(f"{name} q must be a RobotVariable, got {type(variable).__name__}")
    expected_axis = 0 if temporal else None
    if variable.time_axis != expected_axis:
        raise ValueError(f"{name} q time_axis must be {expected_axis!r}, got {variable.time_axis!r}")
