"""Object-referenced residuals and square-root-information weights."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from itertools import count
from numbers import Real
from typing import Any

import torch

from .utils import VariableLike as _VariableLike


class Weight(ABC):
    """Square-root-information multiplier shared by errors and Jacobian rows."""

    @abstractmethod
    def apply(self, error: torch.Tensor) -> torch.Tensor:
        """Apply the weight to an error vector."""

    @abstractmethod
    def apply_jacobian(self, blocks: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        """Apply the identical row multiplier to Jacobian blocks."""

    def is_inactive(self) -> bool:
        """Return whether this is an explicit Python-zero inactive declaration."""
        return False


def _validate_weight_value(value: object, *, label: str) -> None:
    if not isinstance(value, (Real, torch.Tensor)):
        raise TypeError(f"{label} must be a real number or torch.Tensor, got {type(value).__name__}")
    if isinstance(value, torch.Tensor) and not value.is_floating_point():
        raise TypeError(f"{label} tensor must use a floating dtype, got {value.dtype}")


def _same_working_type(value: torch.Tensor, exemplar: torch.Tensor, *, label: str) -> None:
    if value.dtype != exemplar.dtype or value.device != exemplar.device:
        raise ValueError(
            f"{label} must preserve working dtype/device {exemplar.dtype}/{exemplar.device}, "
            f"got {value.dtype}/{value.device}"
        )


class ScaleWeight(Weight):
    """Scalar or per-batch square-root-information weight."""

    def __init__(self, value: Real | torch.Tensor) -> None:
        _validate_weight_value(value, label="ScaleWeight value")
        self.value = value

    def _multiplier(self, output: torch.Tensor, *, row_axes: int) -> Real | torch.Tensor:
        if not isinstance(self.value, torch.Tensor):
            return float(self.value)  # bench-ok: branch proves this is a static Python weight
        _same_working_type(self.value, output, label="ScaleWeight value")
        value = self.value
        batch_shape = tuple(output.shape[:-row_axes])
        if tuple(value.shape) not in ((), batch_shape):
            raise ValueError(
                f"ScaleWeight tensor must be scalar or have exact batch shape {batch_shape}, got {tuple(value.shape)}"
            )
        return value.reshape(*value.shape, *((1,) * row_axes))

    def apply(self, error: torch.Tensor) -> torch.Tensor:
        return error * self._multiplier(error, row_axes=1)

    def apply_jacobian(self, blocks: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        return tuple(block * self._multiplier(block, row_axes=2) for block in blocks)

    def is_inactive(self) -> bool:
        return isinstance(self.value, Real) and self.value == 0.0


class DiagonalWeight(Weight):
    """Per-row square-root-information weight with optional batch axes."""

    def __init__(self, diagonal: torch.Tensor) -> None:
        _validate_weight_value(diagonal, label="DiagonalWeight diagonal")
        if not isinstance(diagonal, torch.Tensor):
            raise TypeError("DiagonalWeight diagonal must be a torch.Tensor")
        if diagonal.ndim < 1:
            raise ValueError("DiagonalWeight diagonal must have a row axis")
        self.diagonal = diagonal

    def _apply(self, output: torch.Tensor, *, row_axes: int) -> torch.Tensor:
        _same_working_type(self.diagonal, output, label="DiagonalWeight diagonal")
        row_width = output.shape[-row_axes]
        if self.diagonal.shape[-1] != row_width:
            raise ValueError(f"DiagonalWeight row width must be {row_width}, got {self.diagonal.shape[-1]}")
        multiplier = self.diagonal if row_axes == 1 else self.diagonal.unsqueeze(-1)
        try:
            return output * multiplier
        except RuntimeError as exc:
            kind = "error" if row_axes == 1 else "Jacobian"
            raise ValueError(
                f"DiagonalWeight shape {tuple(self.diagonal.shape)} must broadcast to {kind} shape {tuple(output.shape)}"
            ) from exc

    def apply(self, error: torch.Tensor) -> torch.Tensor:
        return self._apply(error, row_axes=1)

    def apply_jacobian(self, blocks: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        return tuple(self._apply(block, row_axes=2) for block in blocks)


def _coerce_weight(value: Weight | Real | torch.Tensor, dim: int) -> Weight:
    if isinstance(value, Weight):
        return value
    _validate_weight_value(value, label="weight")
    if isinstance(value, torch.Tensor) and value.ndim and value.shape[-1] == dim:
        return DiagonalWeight(value)
    return ScaleWeight(value)


_RESIDUAL_COUNTERS: dict[type, count] = {}


def _residual_name(cls: type) -> str:
    counter = _RESIDUAL_COUNTERS.setdefault(cls, count())
    return f"{cls.__name__.lower()}_{next(counter)}"


class Residual(ABC):
    """A residual that holds ordered references to every value it reads."""

    def __init__(
        self,
        *variables: _VariableLike,
        dim: int,
        weight: Weight | Real | torch.Tensor = 1.0,
        kernel: object | None = None,
        group_size: int = 1,
        name: str | None = None,
    ) -> None:
        if isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0:
            raise ValueError(f"dim must be a positive int, got {dim!r}")
        if isinstance(group_size, bool) or not isinstance(group_size, int) or group_size <= 0 or dim % group_size:
            raise ValueError(f"group_size must be a positive divisor of dim={dim}, got {group_size!r}")
        if name is None:
            name = _residual_name(type(self))
        if not isinstance(name, str) or not name:
            raise ValueError(f"name must be a non-empty string, got {name!r}")
        ordered: list[_VariableLike] = []
        seen: set[int] = set()
        for variable in variables:
            if not isinstance(variable, _VariableLike):
                raise TypeError(
                    "residual variables must expose tensor/name/trainable and geometry hooks, "
                    f"got {type(variable).__name__}"
                )
            if id(variable) not in seen:
                ordered.append(variable)
                seen.add(id(variable))
        self.variables = tuple(ordered)
        self.dim = dim
        self.group_size = group_size
        self.name = name
        self.kernel = kernel
        self.weight = weight

    @property
    def weight(self) -> Weight:
        return self._weight

    @weight.setter
    def weight(self, value: Weight | Real | torch.Tensor) -> None:
        self._weight = _coerce_weight(value, self.dim)

    @abstractmethod
    def error(self) -> torch.Tensor:
        """Return the unweighted residual with shape ``(..., dim)``."""

    def jacobian(self) -> tuple[torch.Tensor, ...] | None:
        """Return complete analytic tangent blocks or ``None`` to request AD."""
        return None

    def weighted_error(self) -> torch.Tensor:
        """Return the square-root-information-weighted residual."""
        return self.weight.apply(self.error())


class _FunctionResidual(Residual):
    def __init__(self, fn: Callable[..., torch.Tensor], variables: tuple[_VariableLike, ...], **kwargs: Any) -> None:
        self.fn = fn
        super().__init__(*variables, **kwargs)

    def error(self) -> torch.Tensor:
        return self.fn(*(variable.tensor for variable in self.variables))

    def __getattr__(self, attribute: str) -> Any:
        if attribute in {"temporal_structure", "temporal_jacobian_blocks"}:
            return getattr(self.fn, attribute)
        raise AttributeError(attribute)


def residual(
    *variables_or_fn: _VariableLike | Callable[..., torch.Tensor],
    dim: int,
    weight: Weight | Real | torch.Tensor = 1.0,
    kernel: object | None = None,
    group_size: int = 1,
    name: str | None = None,
):
    """Adapt a tensor function into a :class:`Residual` instance."""

    direct_fn: Callable[..., torch.Tensor] | None = None
    values = variables_or_fn
    if values and callable(values[0]) and not isinstance(values[0], _VariableLike):
        direct_fn = values[0]
        values = values[1:]
    variables = tuple(values)

    def wrap(fn: Callable[..., torch.Tensor]) -> Residual:
        return _FunctionResidual(
            fn,
            variables,
            dim=dim,
            weight=weight,
            kernel=kernel,
            group_size=group_size,
            name=name or getattr(fn, "__name__", None),
        )

    return wrap(direct_fn) if direct_fn is not None else wrap


class Difference(Residual):
    """Manifold-aware displacement from a fixed target to a variable."""

    def __init__(
        self,
        variable: _VariableLike,
        target: torch.Tensor,
        *,
        weight: Weight | Real | torch.Tensor = 1.0,
        kernel: object | None = None,
        name: str | None = None,
    ) -> None:
        if not isinstance(target, torch.Tensor):
            raise TypeError(f"target must be a torch.Tensor, got {type(target).__name__}")
        self.variable = variable
        self.target = target
        super().__init__(
            variable,
            dim=variable.tangent_dim(),
            weight=weight,
            kernel=kernel,
            name=name,
        )

    def error(self) -> torch.Tensor:
        return self.variable.difference(self.target.to(self.variable.tensor))


__all__ = ["DiagonalWeight", "Difference", "Residual", "ScaleWeight", "Weight", "residual"]
