"""State manifolds and feasible state-space retraction for variable blocks.

There are two different kinds of bounds and they must never be conflated.
``Bounds`` in this module constrain *state/configuration* coordinates and are
enforced after retraction.  Trust regions and step clamps constrain tangent
coordinates and belong to solvers; no tangent-space box is accepted here.

SO(3), SE(3), and robot configurations use the repository's local/right
perturbation convention: ``x ⊕ dv = x * exp(dv)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from operator import mul
from typing import Protocol, runtime_checkable

import torch

from ...data_model.model import Model
from ...lie import se3, so3


def _numel(shape: tuple[int, ...]) -> int:
    return reduce(mul, shape, 1)


@dataclass(frozen=True)
class Bounds:
    """A box in a variable block's full state space, never tangent space."""

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
    """Structural protocol for a state space with a local tangent space."""

    def retract(self, x: torch.Tensor, dv: torch.Tensor) -> torch.Tensor:
        """Apply the local/right perturbation ``x ⊕ dv``."""

    def difference(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """Return the local tangent ``x1 ⊖ x0``."""

    def tangent_dim(self, shape: tuple[int, ...]) -> int:
        """Return the flattened tangent dimension for one event of ``shape``."""

    def project(self, x: torch.Tensor, bounds: Bounds | None) -> torch.Tensor:
        """Project a retracted state into its feasible state-space set."""

    def validate_bounds(self, bounds: Bounds | None, *, name: str) -> None:
        """Validate that state bounds are meaningful for this manifold."""


@dataclass(frozen=True)
class Euclidean:
    """Euclidean state with additive retraction."""

    def retract(self, x: torch.Tensor, dv: torch.Tensor) -> torch.Tensor:
        return x + dv

    def difference(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        return x1 - x0

    def tangent_dim(self, shape: tuple[int, ...]) -> int:
        return _numel(shape)

    def project(self, x: torch.Tensor, bounds: Bounds | None) -> torch.Tensor:
        return self._project_prevalidated(x, bounds)

    def _project_prevalidated(self, x: torch.Tensor, bounds: Bounds | None) -> torch.Tensor:
        """Project after static/value validation has run at the public boundary."""
        if bounds is None:
            return x
        return torch.maximum(torch.minimum(x, bounds.upper), bounds.lower)

    def validate_bounds(self, bounds: Bounds | None, *, name: str) -> None:
        del name
        if bounds is None:
            return


_GROUP_BOUNDS_ERROR = (
    "{kind} variable blocks have no meaningful global box bound — neither in "
    "state space nor in tangent space. Express rotation limits as residuals "
    "(for example JointRotationPrior or SwingTwistLimitResidual), or use "
    "RobotConfig with joint "
    "limits. Got bounds={bounds!r} on VarSpec {name!r}."
)


@dataclass(frozen=True)
class SO3Manifold:
    """Unit-quaternion SO(3) with local/right perturbations."""

    def retract(self, x: torch.Tensor, dv: torch.Tensor) -> torch.Tensor:
        return so3.normalize(so3.compose(x, so3.exp(dv)))

    def difference(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        return so3.log(so3.normalize(so3.compose(so3.inverse(x0), x1)))

    def tangent_dim(self, shape: tuple[int, ...]) -> int:
        if not shape or shape[-1] != 4:
            raise ValueError(f"SO3 event shape must end in 4, got {shape}")
        return _numel(shape[:-1]) * 3

    def project(self, x: torch.Tensor, bounds: Bounds | None) -> torch.Tensor:
        self.validate_bounds(bounds, name="<unknown>")
        return self._project_prevalidated(x, bounds)

    def _project_prevalidated(self, x: torch.Tensor, bounds: Bounds | None) -> torch.Tensor:
        """Normalize after the public boundary has established ``bounds is None``."""
        del bounds
        return so3.normalize(x)

    def validate_bounds(self, bounds: Bounds | None, *, name: str) -> None:
        if bounds is not None:
            raise ValueError(_GROUP_BOUNDS_ERROR.format(kind="SO3", bounds=bounds, name=name))


@dataclass(frozen=True)
class SE3Manifold:
    """Translation + unit-quaternion SE(3) with local/right perturbations."""

    def retract(self, x: torch.Tensor, dv: torch.Tensor) -> torch.Tensor:
        return se3.normalize(se3.compose(x, se3.exp(dv)))

    def difference(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        return se3.log(se3.compose(se3.inverse(x0), x1))

    def tangent_dim(self, shape: tuple[int, ...]) -> int:
        if not shape or shape[-1] != 7:
            raise ValueError(f"SE3 event shape must end in 7, got {shape}")
        return _numel(shape[:-1]) * 6

    def project(self, x: torch.Tensor, bounds: Bounds | None) -> torch.Tensor:
        self.validate_bounds(bounds, name="<unknown>")
        return self._project_prevalidated(x, bounds)

    def _project_prevalidated(self, x: torch.Tensor, bounds: Bounds | None) -> torch.Tensor:
        """Normalize after the public boundary has established ``bounds is None``."""
        del bounds
        return se3.normalize(x)

    def validate_bounds(self, bounds: Bounds | None, *, name: str) -> None:
        if bounds is not None:
            raise ValueError(_GROUP_BOUNDS_ERROR.format(kind="SE3", bounds=bounds, name=name))


@dataclass(frozen=True)
class _JointCoordinateLayout:
    """Static relationships between one joint's q and tangent coordinates."""

    box_mask: tuple[bool, ...]
    unit_ranges: tuple[tuple[int, int], ...]
    q_for_v: tuple[int, ...]
    unsafe_q: tuple[int, ...]


def _joint_coordinate_layout(joint) -> _JointCoordinateLayout:
    """Describe box, normalization, and tangent mapping semantics for one joint."""
    kind = joint.kind
    if kind == "composite":
        box_mask: list[bool] = []
        unit_ranges: list[tuple[int, int]] = []
        q_for_v: list[int] = []
        unsafe_q: list[int] = []
        q_offset = 0
        for child in joint.sub_joints:
            child_layout = _joint_coordinate_layout(child)
            box_mask.extend(child_layout.box_mask)
            unit_ranges.extend((q_offset + start, q_offset + stop) for start, stop in child_layout.unit_ranges)
            q_for_v.extend(q_offset + index if index >= 0 else -1 for index in child_layout.q_for_v)
            unsafe_q.extend(q_offset + index for index in child_layout.unsafe_q)
            q_offset += child.nq
        return _JointCoordinateLayout(tuple(box_mask), tuple(unit_ranges), tuple(q_for_v), tuple(unsafe_q))

    box_mask = (False,) * joint.nq
    unit_ranges = ()
    q_for_v = (-1,) * joint.nv
    unsafe_q = ()
    if kind == "free_flyer":
        box_mask = (True, True, True, False, False, False, False)
        unit_ranges = ((3, 7),)
        unsafe_q = (0, 1, 2)
    elif kind == "spherical":
        unit_ranges = ((0, 4),)
    elif kind == "planar":
        box_mask = (True, True, False, False)
        unit_ranges = ((2, 4),)
        q_for_v = (0, 1, -1)
    elif kind == "revolute_unbounded":
        unit_ranges = ((0, 2),)
    elif kind.startswith(("revolute", "prismatic")) or kind in {"helical", "translation"}:
        box_mask = (True,) * joint.nq
        q_for_v = tuple(range(joint.nv))
    return _JointCoordinateLayout(box_mask, unit_ranges, q_for_v, unsafe_q)


@dataclass(frozen=True)
class RobotConfig:
    """A robot configuration manifold wrapping ``Model.integrate/difference``."""

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
        for joint, nq_joint, iq in zip(
            self.model.joint_models,
            self.model.nqs,
            self.model.idx_qs,
            strict=True,
        ):
            if nq_joint:
                mask[iq : iq + nq_joint] = torch.tensor(_joint_coordinate_layout(joint).box_mask)
        return mask

    @property
    def unit_coordinate_slices(self) -> tuple[slice, ...]:
        """Configuration slices representing quaternions or unit circles."""
        ranges: list[tuple[int, int]] = []
        for joint, nq_joint, iq in zip(
            self.model.joint_models,
            self.model.nqs,
            self.model.idx_qs,
            strict=True,
        ):
            if nq_joint:
                ranges.extend((iq + start, iq + stop) for start, stop in _joint_coordinate_layout(joint).unit_ranges)
        return tuple(slice(start, stop) for start, stop in ranges)

    def validate_bounds(self, bounds: Bounds | None, *, name: str) -> None:
        if bounds is None:
            return
        if tuple(bounds.lower.shape) != (self.model.nq,):
            raise ValueError(
                f"Bounds on RobotConfig VarSpec {name!r} must be nq-shaped "
                f"({self.model.nq},), got {tuple(bounds.lower.shape)}"
            )
        protected = ~self.box_mask.to(device=bounds.lower.device)
        valid_lower = torch.isneginf(bounds.lower[protected])
        valid_upper = torch.isposinf(bounds.upper[protected])
        if not bool(valid_lower.all() and valid_upper.all()):
            indices = torch.nonzero(protected, as_tuple=False).flatten().tolist()
            raise ValueError(
                f"RobotConfig VarSpec {name!r} bounds must be (-inf, +inf) on "
                f"non-box manifold coordinates {indices}; quaternion/unit-circle "
                "coordinates are never clamped"
            )

    def project(self, x: torch.Tensor, bounds: Bounds | None) -> torch.Tensor:
        if bounds is None:
            return x
        self.validate_bounds(bounds, name="<unknown>")
        return self._project_prevalidated(x, bounds)

    def _project_prevalidated(self, x: torch.Tensor, bounds: Bounds | None) -> torch.Tensor:
        """Project valid RobotConfig bounds without tensor-to-host predicates.

        :meth:`validate_bounds` establishes that every non-box coordinate has
        ``(-inf, +inf)`` bounds. Clamping the complete state is therefore
        equivalent to masking box coordinates, while avoiding construction and
        device transfer of a fresh mask in a solver update.
        """
        if bounds is None:
            return x
        return torch.maximum(torch.minimum(x, bounds.upper), bounds.lower)
