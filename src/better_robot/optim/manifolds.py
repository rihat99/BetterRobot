"""State manifolds and feasible state-space retraction."""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from operator import mul
from typing import Protocol, runtime_checkable

import torch

from ..data_model.model import Model
from ..lie import se3, so3


def _numel(shape: tuple[int, ...]) -> int:
    return reduce(mul, shape, 1)


@dataclass(frozen=True)
class Bounds:
    """Box constraints in full state coordinates."""

    lower: torch.Tensor
    upper: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.lower, torch.Tensor) or not isinstance(self.upper, torch.Tensor):
            raise TypeError("Bounds lower and upper must be torch.Tensor values")
        if self.lower.shape != self.upper.shape:
            raise ValueError(
                f"Bounds lower/upper shapes must match, got {tuple(self.lower.shape)} and {tuple(self.upper.shape)}"
            )
        if self.lower.dtype != self.upper.dtype or self.lower.device != self.upper.device:
            raise ValueError("Bounds lower and upper must have the same dtype and device")
        if not self.lower.is_floating_point():
            raise TypeError("Bounds lower and upper must use a floating dtype")
        if bool(torch.any(torch.isnan(self.lower))) or bool(torch.any(torch.isnan(self.upper))):
            raise ValueError("Bounds lower and upper must not contain NaN")
        if bool(torch.any(self.lower > self.upper)):
            raise ValueError("Bounds lower must be <= upper at every state coordinate")


@runtime_checkable
class Manifold(Protocol):
    def retract(self, x: torch.Tensor, dv: torch.Tensor) -> torch.Tensor: ...
    def difference(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor: ...
    def tangent_dim(self, shape: tuple[int, ...]) -> int: ...
    def project(self, x: torch.Tensor, bounds: Bounds | None) -> torch.Tensor: ...
    def validate_bounds(self, bounds: Bounds | None, *, name: str) -> None: ...


@dataclass(frozen=True)
class Euclidean:
    def retract(self, x: torch.Tensor, dv: torch.Tensor) -> torch.Tensor:
        return x + dv

    def difference(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        return x1 - x0

    def tangent_dim(self, shape: tuple[int, ...]) -> int:
        return _numel(shape)

    def project(self, x: torch.Tensor, bounds: Bounds | None) -> torch.Tensor:
        return x if bounds is None else torch.maximum(torch.minimum(x, bounds.upper), bounds.lower)

    def validate_bounds(self, bounds: Bounds | None, *, name: str) -> None:
        del bounds, name


_GROUP_BOUNDS_ERROR = (
    "{kind} variable blocks have no meaningful global box bound — neither in "
    "state space nor in tangent space. Express rotation limits as residuals "
    "(for example JointRotationPrior or SwingTwistLimitResidual), or use "
    "RobotConfig with joint "
    "limits. Got bounds={bounds!r} on VarSpec {name!r}."
)


@dataclass(frozen=True)
class SO3Manifold:
    def retract(self, x: torch.Tensor, dv: torch.Tensor) -> torch.Tensor:
        return so3.normalize(so3.compose(x, so3.exp(dv)))

    def difference(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        return so3.log(so3.normalize(so3.compose(so3.inverse(x0), x1)))

    def tangent_dim(self, shape: tuple[int, ...]) -> int:
        if not shape or shape[-1] != 4:
            raise ValueError(f"SO3 event shape must end in 4, got {shape}")
        return _numel(shape[:-1]) * 3

    def project(self, x: torch.Tensor, bounds: Bounds | None) -> torch.Tensor:
        del bounds
        return so3.normalize(x)

    def validate_bounds(self, bounds: Bounds | None, *, name: str) -> None:
        if bounds is not None:
            raise ValueError(_GROUP_BOUNDS_ERROR.format(kind="SO3", bounds=bounds, name=name))


@dataclass(frozen=True)
class SE3Manifold:
    def retract(self, x: torch.Tensor, dv: torch.Tensor) -> torch.Tensor:
        return se3.normalize(se3.compose(x, se3.exp(dv)))

    def difference(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        return se3.log(se3.compose(se3.inverse(x0), x1))

    def tangent_dim(self, shape: tuple[int, ...]) -> int:
        if not shape or shape[-1] != 7:
            raise ValueError(f"SE3 event shape must end in 7, got {shape}")
        return _numel(shape[:-1]) * 6

    def project(self, x: torch.Tensor, bounds: Bounds | None) -> torch.Tensor:
        del bounds
        return se3.normalize(x)

    def validate_bounds(self, bounds: Bounds | None, *, name: str) -> None:
        if bounds is not None:
            raise ValueError(_GROUP_BOUNDS_ERROR.format(kind="SE3", bounds=bounds, name=name))


@dataclass(frozen=True)
class _JointCoordinateLayout:
    box_mask: tuple[bool, ...]
    unit_ranges: tuple[tuple[int, int], ...]
    q_for_v: tuple[int, ...]
    unsafe_q: tuple[int, ...]


def _joint_coordinate_layout(joint) -> _JointCoordinateLayout:
    kind = joint.kind
    if kind == "composite":
        box_mask: list[bool] = []
        unit_ranges: list[tuple[int, int]] = []
        q_for_v: list[int] = []
        unsafe_q: list[int] = []
        q_offset = 0
        for child in joint.sub_joints:
            layout = _joint_coordinate_layout(child)
            box_mask.extend(layout.box_mask)
            unit_ranges.extend((q_offset + start, q_offset + stop) for start, stop in layout.unit_ranges)
            q_for_v.extend(q_offset + index if index >= 0 else -1 for index in layout.q_for_v)
            unsafe_q.extend(q_offset + index for index in layout.unsafe_q)
            q_offset += child.nq
        return _JointCoordinateLayout(tuple(box_mask), tuple(unit_ranges), tuple(q_for_v), tuple(unsafe_q))

    box_mask = (False,) * joint.nq
    unit_ranges: tuple[tuple[int, int], ...] = ()
    q_for_v = (-1,) * joint.nv
    unsafe_q: tuple[int, ...] = ()
    if kind == "free_flyer":
        box_mask, unit_ranges, unsafe_q = (True, True, True, False, False, False, False), ((3, 7),), (0, 1, 2)
    elif kind == "spherical":
        unit_ranges = ((0, 4),)
    elif kind == "planar":
        box_mask, unit_ranges, q_for_v = (True, True, False, False), ((2, 4),), (0, 1, -1)
    elif kind == "revolute_unbounded":
        unit_ranges = ((0, 2),)
    elif kind.startswith(("revolute", "prismatic")) or kind in {"helical", "translation"}:
        box_mask, q_for_v = (True,) * joint.nq, tuple(range(joint.nv))
    return _JointCoordinateLayout(box_mask, unit_ranges, q_for_v, unsafe_q)


@dataclass(frozen=True)
class RobotConfig:
    model: Model

    def retract(self, x: torch.Tensor, dv: torch.Tensor) -> torch.Tensor:
        return self.model.integrate(x, dv)

    def difference(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        return self.model.difference(x0, x1)

    def tangent_dim(self, shape: tuple[int, ...]) -> int:
        if not shape or shape[-1] != self.model.nq:
            raise ValueError(f"RobotConfig event shape must end in model.nq={self.model.nq}, got {shape}")
        return _numel(shape[:-1]) * self.model.nv

    @property
    def box_mask(self) -> torch.Tensor:
        mask = torch.zeros(self.model.nq, dtype=torch.bool)
        for joint, nq_joint, iq in zip(self.model.joint_models, self.model.nqs, self.model.idx_qs, strict=True):
            if nq_joint:
                mask[iq : iq + nq_joint] = torch.tensor(_joint_coordinate_layout(joint).box_mask)
        return mask

    def joint_bounds(self) -> Bounds:
        lower, upper = self.model.lower_pos_limit, self.model.upper_pos_limit
        box = self.box_mask.to(lower.device)
        return Bounds(
            torch.where(box, lower, torch.full_like(lower, -torch.inf)),
            torch.where(box, upper, torch.full_like(upper, torch.inf)),
        )

    @property
    def unit_coordinate_slices(self) -> tuple[slice, ...]:
        ranges: list[tuple[int, int]] = []
        for joint, nq_joint, iq in zip(self.model.joint_models, self.model.nqs, self.model.idx_qs, strict=True):
            if nq_joint:
                ranges.extend((iq + start, iq + stop) for start, stop in _joint_coordinate_layout(joint).unit_ranges)
        return tuple(slice(start, stop) for start, stop in ranges)

    def validate_bounds(self, bounds: Bounds | None, *, name: str) -> None:
        if bounds is None:
            return
        if tuple(bounds.lower.shape) != (self.model.nq,):
            raise ValueError(
                f"Bounds on RobotConfig VarSpec {name!r} must be nq-shaped ({self.model.nq},), "
                f"got {tuple(bounds.lower.shape)}"
            )
        protected = ~self.box_mask.to(bounds.lower.device)
        valid = torch.isneginf(bounds.lower[protected]).all() & torch.isposinf(bounds.upper[protected]).all()
        if not bool(valid):
            indices = torch.nonzero(protected, as_tuple=False).flatten().tolist()
            raise ValueError(
                f"RobotConfig VarSpec {name!r} bounds must be (-inf, +inf) on non-box manifold coordinates "
                f"{indices}; quaternion/unit-circle coordinates are never clamped"
            )

    def project(self, x: torch.Tensor, bounds: Bounds | None) -> torch.Tensor:
        return x if bounds is None else torch.maximum(torch.minimum(x, bounds.upper), bounds.lower)
