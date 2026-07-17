"""Named variable-block specifications and tangent elimination utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeAlias

import torch

from ...exceptions import DeviceMismatchError, DtypeMismatchError, QuaternionNormError
from .manifolds import (
    Bounds,
    Euclidean,
    Manifold,
    RobotConfig,
    SE3Manifold,
    SO3Manifold,
)

Values: TypeAlias = dict[str, torch.Tensor]


@dataclass(frozen=True)
class VarSpec:
    """Static description of one optimized variable block.

    ``shape`` is the full-state *event* shape. Leading axes in a value are
    explicit independent batch axes; they are never folded into ``shape``. A
    trajectory can therefore use event shape ``(T, nq)`` while an independent
    batch of trajectories has values shaped ``(B, T, nq)``.

    ``bounds`` constrain the state shape (``nq`` coordinates for each
    :class:`RobotConfig` event) and feasible retraction enforces them. ``scale``
    and ``mask`` live in the flattened tangent space instead: for RobotConfig
    those dimensions use ``nv``. A nonzero mask entry is free/trainable; zero
    entries are eliminated from Jacobians and normal systems rather than kept
    as zero columns. Solver trust-region or step bounds are intentionally not
    represented here.
    """

    name: str
    shape: tuple[int, ...]
    manifold: Manifold = field(default_factory=Euclidean)
    bounds: Bounds | None = None
    scale: torch.Tensor | None = None
    mask: torch.Tensor | None = None
    time_axis: int | None = None
    _tangent_dim: int = field(init=False, repr=False, compare=False)
    _free_indices: torch.Tensor = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:  # noqa: PLR0912 - validates one complete static block
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("VarSpec name must be non-empty")
        if not isinstance(self.shape, tuple):
            raise TypeError(f"VarSpec {self.name!r} shape must be a tuple[int, ...]")
        if any(not isinstance(size, int) or size <= 0 for size in self.shape):
            raise ValueError(f"VarSpec {self.name!r} shape must contain positive integers")
        if not isinstance(self.manifold, Manifold):
            raise TypeError(f"VarSpec {self.name!r} manifold does not implement Manifold")
        tangent_dim = self.manifold.tangent_dim(self.shape)
        if not isinstance(tangent_dim, int) or tangent_dim <= 0:
            raise ValueError(f"VarSpec {self.name!r} manifold tangent_dim must be a positive int, got {tangent_dim!r}")
        object.__setattr__(self, "_tangent_dim", tangent_dim)

        if self.time_axis is not None:
            if isinstance(self.time_axis, bool) or not isinstance(self.time_axis, int):
                raise TypeError(f"VarSpec {self.name!r} time_axis must be an int or None")
            if self.time_axis != 0:
                raise ValueError(
                    f"VarSpec {self.name!r} M5 time_axis must be 0 or None; "
                    "transpose the event layout so time is leading"
                )
            time_length = self.shape[0]
            if tangent_dim % time_length:
                raise ValueError(
                    f"VarSpec {self.name!r} tangent_dim={tangent_dim} is not divisible by time length {time_length}"
                )

        if self.bounds is not None and not isinstance(self.bounds, Bounds):
            raise TypeError(f"VarSpec {self.name!r} bounds must be Bounds or None")
        self.manifold.validate_bounds(self.bounds, name=self.name)
        if self.bounds is not None:
            # RobotConfig bounds are one nq slice and broadcast over trajectory
            # event axes. Euclidean bounds cover the full event shape.
            expected = self.shape[-1:] if isinstance(self.manifold, RobotConfig) else self.shape
            if tuple(self.bounds.lower.shape) != expected:
                raise ValueError(
                    f"Bounds on VarSpec {self.name!r} must have state shape {expected}, "
                    f"got {tuple(self.bounds.lower.shape)}"
                )
        expected_tangent = (tangent_dim,)
        for label, tensor in (("scale", self.scale), ("mask", self.mask)):
            if tensor is not None and not isinstance(tensor, torch.Tensor):
                raise TypeError(f"VarSpec {self.name!r} {label} must be a torch.Tensor or None")
            if tensor is not None and tuple(tensor.shape) != expected_tangent:
                state_label = (
                    f"state shape {self.shape} (nq={self.manifold.model.nq})"
                    if isinstance(self.manifold, RobotConfig)
                    else f"state shape {self.shape}"
                )
                raise ValueError(
                    f"VarSpec {self.name!r} {label} must have tangent shape "
                    f"{expected_tangent}, not {state_label}; got {tuple(tensor.shape)}"
                )
        if self.scale is not None:
            if not self.scale.is_floating_point():
                raise TypeError(f"VarSpec {self.name!r} scale must use a floating dtype")
            if bool(torch.any(~torch.isfinite(self.scale))) or bool(torch.any(self.scale <= 0)):
                raise ValueError(f"VarSpec {self.name!r} scale entries must be finite and > 0")

        if self.mask is None:
            indices = torch.arange(tangent_dim, dtype=torch.long)
        else:
            indices = torch.nonzero(self.mask.to(dtype=torch.bool), as_tuple=False).flatten()
        object.__setattr__(self, "_free_indices", indices)

    @property
    def tangent_dim(self) -> int:
        """Full flattened tangent dimension before mask elimination."""
        return self._tangent_dim

    @property
    def free_dim(self) -> int:
        """Tangent dimension after eliminating fixed coordinates."""
        return int(self._free_indices.numel())

    @property
    def free_indices(self) -> torch.Tensor:
        """Precomputed full-tangent indices retained by the mask."""
        return self._free_indices

    @property
    def free_scale(self) -> torch.Tensor | None:
        """Per-coordinate scale after applying the same mask elimination."""
        if self.scale is None:
            return None
        return self.gather_tangent(self.scale)

    @property
    def time_length(self) -> int:
        """Static horizon for a time-annotated variable."""
        if self.time_axis is None:
            raise ValueError(f"VarSpec {self.name!r} has no time_axis")
        return self.shape[0]

    @property
    def temporal_tangent_width(self) -> int:
        """Full tangent width of one time slice before mask elimination."""
        return self.tangent_dim // self.time_length

    @property
    def temporal_mask_is_separable(self) -> bool:
        """Whether every knot retains the same local tangent coordinates."""
        if self.time_axis is None:
            return False
        if self.mask is None:
            return True
        shaped = self.mask.to(dtype=torch.bool).reshape(self.time_length, self.temporal_tangent_width)
        return bool(torch.equal(shaped, shaped[:1].expand_as(shaped)))

    @property
    def temporal_free_indices(self) -> torch.Tensor:
        """Per-knot local free coordinates for a separable temporal mask."""
        if not self.temporal_mask_is_separable:
            raise ValueError(f"VarSpec {self.name!r} temporal mask is not separable across time")
        if self.mask is None:
            return torch.arange(self.temporal_tangent_width, dtype=torch.long)
        first = self.mask.to(dtype=torch.bool).reshape(self.time_length, self.temporal_tangent_width)[0]
        return torch.nonzero(first, as_tuple=False).flatten()

    @property
    def temporal_reduced_width(self) -> int:
        """Mask-reduced tangent width of one time slice."""
        return int(self.temporal_free_indices.numel())

    def batch_shape(self, value: torch.Tensor) -> tuple[int, ...]:
        self.validate_value(value)
        return tuple(value.shape[: value.ndim - len(self.shape)])

    def validate_value(self, value: torch.Tensor, *, check_feasible: bool = True) -> None:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Value for VarSpec {self.name!r} must be a torch.Tensor")
        if value.dtype not in (torch.float32, torch.float64):
            raise DtypeMismatchError(
                f"Value for VarSpec {self.name!r} has unsupported dtype={value.dtype}; "
                "use torch.float32 or torch.float64"
            )
        trailing_shape = tuple(value.shape[-len(self.shape) :]) if self.shape else ()
        if value.ndim < len(self.shape) or trailing_shape != self.shape:
            raise ValueError(
                f"Value for VarSpec {self.name!r} must end in event shape {self.shape}, got {tuple(value.shape)}"
            )
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"Value for VarSpec {self.name!r} must contain only finite entries")
        unit_slices: tuple[slice, ...] = ()
        if isinstance(self.manifold, SO3Manifold):
            unit_slices = (slice(-4, None),)
        elif isinstance(self.manifold, SE3Manifold):
            unit_slices = (slice(-4, None),)
        elif isinstance(self.manifold, RobotConfig):
            unit_slices = self.manifold.unit_coordinate_slices
        for unit_slice in unit_slices:
            coordinates = value[..., unit_slice]
            norms = coordinates.norm(dim=-1)
            if not torch.allclose(
                norms,
                torch.ones_like(norms),
                rtol=2e-5,
                atol=2e-6,
            ):
                raise QuaternionNormError(
                    f"Initial value for VarSpec {self.name!r} is not on its "
                    "configuration manifold; initial values are validated, not "
                    "silently normalized. Supply unit quaternion/unit-circle coordinates."
                )
        if check_feasible and self.bounds is not None:
            if value.dtype != self.bounds.lower.dtype or value.device != self.bounds.lower.device:
                raise ValueError(
                    f"Value and Bounds for VarSpec {self.name!r} must have the same "
                    f"dtype/device, got {value.dtype}/{value.device} and "
                    f"{self.bounds.lower.dtype}/{self.bounds.lower.device}"
                )
            lower = self.bounds.lower
            upper = self.bounds.upper
            outside = (value < lower) | (value > upper)
            if bool(torch.any(outside)):
                raise ValueError(
                    f"Initial value for VarSpec {self.name!r} is outside its state bounds; "
                    "initial values are validated, not silently clamped. Supply a feasible "
                    "start (notably, Panda q_neutral violates joint-4 limits and must be "
                    "projected explicitly by the caller)."
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
        """Gather free coordinates from a full flattened tangent."""
        if full.shape[-1] != self.tangent_dim:
            raise ValueError(
                f"Full tangent for VarSpec {self.name!r} must end in {self.tangent_dim}, got {tuple(full.shape)}"
            )
        if self.free_dim == self.tangent_dim:
            return full.clone()
        return self._gather_tangent_prevalidated(full)

    def _gather_tangent_prevalidated(self, full: torch.Tensor) -> torch.Tensor:
        """Gather free coordinates from a prevalidated full tangent."""
        if self.free_dim == self.tangent_dim:
            return full
        return full.index_select(-1, self.free_indices.to(device=full.device))

    def expand_tangent(self, reduced: torch.Tensor) -> torch.Tensor:
        """Scatter a reduced tangent into the full space, filling fixed DOFs with zero."""
        if reduced.shape[-1] != self.free_dim:
            raise ValueError(
                f"Reduced tangent for VarSpec {self.name!r} must end in {self.free_dim}, got {tuple(reduced.shape)}"
            )
        if self.free_dim == self.tangent_dim:
            return reduced.clone()
        return self._expand_tangent_prevalidated(reduced)

    def _expand_tangent_prevalidated(self, reduced: torch.Tensor) -> torch.Tensor:
        """Expand a reduced tangent whose shape/dtype/device were prevalidated."""
        if self.free_dim == self.tangent_dim:
            return reduced
        full = reduced.new_zeros(*reduced.shape[:-1], self.tangent_dim)
        if self.free_dim:
            full.index_copy_(-1, self.free_indices.to(device=reduced.device), reduced)
        return full

    def _retract_full_prevalidated(
        self,
        value: torch.Tensor,
        full: torch.Tensor,
        *,
        batch_shape: tuple[int, ...],
    ) -> torch.Tensor:
        shaped = full.reshape((*batch_shape, *self._tangent_event_shape()))
        retracted = self.manifold.retract(value, shaped)
        project = getattr(self.manifold, "_project_prevalidated", self.manifold.project)
        return project(retracted, self.bounds)

    def _retract_prevalidated(
        self,
        value: torch.Tensor,
        reduced_delta: torch.Tensor,
        *,
        batch_shape: tuple[int, ...],
    ) -> torch.Tensor:
        """Retract solver-owned tensors without repeating public validation."""
        full = self._expand_tangent_prevalidated(reduced_delta)
        return self._retract_full_prevalidated(value, full, batch_shape=batch_shape)

    def retract(self, value: torch.Tensor, reduced_delta: torch.Tensor) -> torch.Tensor:
        """Expand a reduced step, retract, and enforce state-space feasibility."""
        # Public retraction never turns an invalid starting state into an
        # apparently valid one by silently clamping it.
        batch_shape = self.batch_shape(value)
        if not isinstance(reduced_delta, torch.Tensor):
            raise TypeError(f"Step for VarSpec {self.name!r} must be a torch.Tensor")
        if reduced_delta.dtype != value.dtype:
            raise DtypeMismatchError(
                f"Step and value for VarSpec {self.name!r} must share dtype; "
                f"got {reduced_delta.dtype} and {value.dtype}"
            )
        if reduced_delta.device != value.device:
            raise DeviceMismatchError(
                f"Step and value for VarSpec {self.name!r} must share device; "
                f"got {reduced_delta.device} and {value.device}"
            )
        full = self.expand_tangent(reduced_delta)
        expected = (*batch_shape, self.tangent_dim)
        if tuple(full.shape) != expected:
            raise ValueError(
                f"Step for VarSpec {self.name!r} must have shape "
                f"{(*batch_shape, self.free_dim)}, got {tuple(reduced_delta.shape)}"
            )
        return self._retract_full_prevalidated(value, full, batch_shape=batch_shape)

    def _difference_prevalidated(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        *,
        batch_shape: tuple[int, ...],
    ) -> torch.Tensor:
        """Return a full tangent difference for prevalidated solver values."""
        delta = self.manifold.difference(x0, x1)
        return delta.reshape(*batch_shape, self.tangent_dim)

    def difference(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """Return a full flattened tangent difference between two states."""
        batch_shape = self.batch_shape(x0)
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
        return self._difference_prevalidated(x0, x1, batch_shape=batch_shape)


def detach_values(values: Values) -> Values:
    """Return graph-free accepted-state artifacts for diagnostics/warm starts."""
    return {name: value.detach() for name, value in values.items()}
