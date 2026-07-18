"""Named variable blocks and tangent-coordinate elimination."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeAlias

import torch

from ..exceptions import DeviceMismatchError, DtypeMismatchError
from .manifolds import Bounds, Euclidean, Manifold, RobotConfig, SE3Manifold, SO3Manifold

Values: TypeAlias = dict[str, torch.Tensor]


def _validate_optional_tensor(
    name: str,
    label: str,
    value: torch.Tensor | None,
    shape: tuple[int, ...],
) -> None:
    if value is None:
        return
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"VarSpec {name!r} {label} must be a torch.Tensor or None")
    if tuple(value.shape) != shape:
        raise ValueError(f"VarSpec {name!r} {label} must have tangent shape {shape}, got {tuple(value.shape)}")


@dataclass(frozen=True)
class VarSpec:
    """Static state event, manifold, bounds, and reduced tangent layout."""

    name: str
    shape: tuple[int, ...]
    manifold: Manifold = field(default_factory=Euclidean)
    bounds: Bounds | None = None
    scale: torch.Tensor | None = None
    mask: torch.Tensor | None = None
    time_axis: int | None = None
    _tangent_dim: int = field(init=False, repr=False, compare=False)
    _free_indices: torch.Tensor = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("VarSpec name must be non-empty")
        if not isinstance(self.shape, tuple) or any(not isinstance(size, int) or size <= 0 for size in self.shape):
            raise ValueError(f"VarSpec {self.name!r} shape must be a tuple of positive integers")
        if not isinstance(self.manifold, Manifold):
            raise TypeError(f"VarSpec {self.name!r} manifold does not implement Manifold")
        tangent_dim = self.manifold.tangent_dim(self.shape)
        if not isinstance(tangent_dim, int) or tangent_dim <= 0:
            raise ValueError(f"VarSpec {self.name!r} manifold tangent_dim must be a positive int, got {tangent_dim!r}")
        object.__setattr__(self, "_tangent_dim", tangent_dim)

        if self.time_axis is not None:
            if isinstance(self.time_axis, bool) or not isinstance(self.time_axis, int):
                raise TypeError(f"VarSpec {self.name!r} time_axis must be an int or None")
            if self.time_axis != 0 or not self.shape or tangent_dim % self.shape[0]:
                raise ValueError(f"VarSpec {self.name!r} time_axis must identify a leading, separable time axis")
        if self.bounds is not None and not isinstance(self.bounds, Bounds):
            raise TypeError(f"VarSpec {self.name!r} bounds must be Bounds or None")
        self.manifold.validate_bounds(self.bounds, name=self.name)
        expected_bounds = self.shape[-1:] if isinstance(self.manifold, RobotConfig) else self.shape
        if self.bounds is not None and tuple(self.bounds.lower.shape) != expected_bounds:
            raise ValueError(
                f"Bounds on VarSpec {self.name!r} must have state shape {expected_bounds}, "
                f"got {tuple(self.bounds.lower.shape)}"
            )
        tangent_shape = (tangent_dim,)
        _validate_optional_tensor(self.name, "scale", self.scale, tangent_shape)
        _validate_optional_tensor(self.name, "mask", self.mask, tangent_shape)
        if self.scale is not None and not self.scale.is_floating_point():
            raise TypeError(f"VarSpec {self.name!r} scale must use a floating dtype")
        indices = (
            torch.arange(tangent_dim, dtype=torch.long)
            if self.mask is None
            else torch.nonzero(self.mask.to(dtype=torch.bool), as_tuple=False).flatten()
        )
        object.__setattr__(self, "_free_indices", indices)

    @property
    def tangent_dim(self) -> int:
        return self._tangent_dim

    @property
    def free_dim(self) -> int:
        return int(self._free_indices.numel())

    @property
    def free_indices(self) -> torch.Tensor:
        return self._free_indices

    @property
    def free_scale(self) -> torch.Tensor | None:
        return None if self.scale is None else self.gather_tangent(self.scale)

    @property
    def time_length(self) -> int:
        if self.time_axis is None:
            raise ValueError(f"VarSpec {self.name!r} has no time_axis")
        return self.shape[0]

    @property
    def temporal_tangent_width(self) -> int:
        return self.tangent_dim // self.time_length

    @property
    def temporal_mask_is_separable(self) -> bool:
        if self.time_axis is None:
            return False
        if self.mask is None:
            return True
        shaped = self.mask.to(dtype=torch.bool).reshape(self.time_length, self.temporal_tangent_width)
        return bool(torch.equal(shaped, shaped[:1].expand_as(shaped)))

    @property
    def temporal_free_indices(self) -> torch.Tensor:
        if not self.temporal_mask_is_separable:
            raise ValueError(f"VarSpec {self.name!r} temporal mask is not separable across time")
        if self.mask is None:
            return torch.arange(self.temporal_tangent_width, dtype=torch.long)
        first = self.mask.to(dtype=torch.bool).reshape(self.time_length, self.temporal_tangent_width)[0]
        return torch.nonzero(first, as_tuple=False).flatten()

    @property
    def temporal_reduced_width(self) -> int:
        return int(self.temporal_free_indices.numel())

    def batch_shape(self, value: torch.Tensor) -> tuple[int, ...]:
        self.validate_value(value)
        return tuple(value.shape[: value.ndim - len(self.shape)])

    def validate_value(self, value: torch.Tensor) -> None:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Value for VarSpec {self.name!r} must be a torch.Tensor")
        if value.dtype not in (torch.float32, torch.float64):
            raise DtypeMismatchError(
                f"Value for VarSpec {self.name!r} has unsupported dtype={value.dtype}; use torch.float32 or torch.float64"
            )
        trailing = tuple(value.shape[-len(self.shape) :]) if self.shape else ()
        if value.ndim < len(self.shape) or trailing != self.shape:
            raise ValueError(
                f"Value for VarSpec {self.name!r} must end in event shape {self.shape}, got {tuple(value.shape)}"
            )
        if self.bounds is not None and (
            value.dtype != self.bounds.lower.dtype or value.device != self.bounds.lower.device
        ):
            raise ValueError(
                f"Value and Bounds for VarSpec {self.name!r} must have the same dtype/device, "
                f"got {value.dtype}/{value.device} and {self.bounds.lower.dtype}/{self.bounds.lower.device}"
            )

    def _tangent_event_shape(self) -> tuple[int, ...]:
        if isinstance(self.manifold, RobotConfig):
            return (*self.shape[:-1], self.manifold.model.nv)
        if isinstance(self.manifold, SO3Manifold):
            return (*self.shape[:-1], 3)
        if isinstance(self.manifold, SE3Manifold):
            return (*self.shape[:-1], 6)
        return self.shape

    def gather_tangent(self, full: torch.Tensor) -> torch.Tensor:
        if full.shape[-1] != self.tangent_dim:
            raise ValueError(
                f"Full tangent for VarSpec {self.name!r} must end in {self.tangent_dim}, got {tuple(full.shape)}"
            )
        return full if self.free_dim == self.tangent_dim else full.index_select(-1, self.free_indices.to(full.device))

    def expand_tangent(self, reduced: torch.Tensor) -> torch.Tensor:
        if reduced.shape[-1] != self.free_dim:
            raise ValueError(
                f"Reduced tangent for VarSpec {self.name!r} must end in {self.free_dim}, got {tuple(reduced.shape)}"
            )
        if self.free_dim == self.tangent_dim:
            return reduced
        full = reduced.new_zeros(*reduced.shape[:-1], self.tangent_dim)
        if self.free_dim:
            full.index_copy_(-1, self.free_indices.to(reduced.device), reduced)
        return full

    def retract(self, value: torch.Tensor, reduced_delta: torch.Tensor) -> torch.Tensor:
        batch = self.batch_shape(value)
        if not isinstance(reduced_delta, torch.Tensor):
            raise TypeError(f"Step for VarSpec {self.name!r} must be a torch.Tensor")
        if reduced_delta.dtype != value.dtype:
            raise DtypeMismatchError(
                f"Step and value for VarSpec {self.name!r} must share dtype; got {reduced_delta.dtype} and {value.dtype}"
            )
        if reduced_delta.device != value.device:
            raise DeviceMismatchError(
                f"Step and value for VarSpec {self.name!r} must share device; got {reduced_delta.device} and {value.device}"
            )
        expected = (*batch, self.free_dim)
        if tuple(reduced_delta.shape) != expected:
            raise ValueError(
                f"Step for VarSpec {self.name!r} must have shape {expected}, got {tuple(reduced_delta.shape)}"
            )
        full = self.expand_tangent(reduced_delta).reshape((*batch, *self._tangent_event_shape()))
        return self.manifold.project(self.manifold.retract(value, full), self.bounds)

    def difference(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        batch = self.batch_shape(x0)
        if tuple(x1.shape) != tuple(x0.shape):
            raise ValueError(f"VarSpec {self.name!r} difference inputs must have equal shape")
        self.validate_value(x1)
        if x1.dtype != x0.dtype:
            raise DtypeMismatchError(
                f"Difference inputs for VarSpec {self.name!r} must share dtype; got {x0.dtype} and {x1.dtype}"
            )
        if x1.device != x0.device:
            raise DeviceMismatchError(
                f"Difference inputs for VarSpec {self.name!r} must share device; got {x0.device} and {x1.device}"
            )
        return self.manifold.difference(x0, x1).reshape(*batch, self.tangent_dim)


def detach_values(values: Values) -> Values:
    return {name: value.detach() for name, value in values.items()}
