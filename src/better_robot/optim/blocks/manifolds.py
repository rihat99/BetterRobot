"""State manifolds and feasible state-space retraction for variable blocks.

There are two different kinds of bounds and they must never be conflated.
``Bounds`` in this module constrain *state/configuration* coordinates and are
enforced after retraction.  Trust regions and step clamps constrain tangent
coordinates and belong to solvers (M2b); no tangent-space box is accepted here.

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
    "(rotation prior / swing-twist, roadmap M3), or use RobotConfig with joint "
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


def _joint_box_mask(joint) -> torch.Tensor:
    """Return which state coordinates of one joint admit box projection."""
    kind = joint.kind
    if joint.nq == 0:
        result = torch.zeros(0, dtype=torch.bool)
    elif kind == "free_flyer":
        result = torch.tensor([True, True, True, False, False, False, False])
    elif kind == "spherical":
        result = torch.zeros(4, dtype=torch.bool)
    elif kind == "planar":
        result = torch.tensor([True, True, False, False])
    elif kind == "revolute_unbounded":
        result = torch.zeros(2, dtype=torch.bool)
    elif kind == "composite":
        parts = [_joint_box_mask(child) for child in joint.sub_joints]
        result = torch.cat(parts) if parts else torch.zeros(0, dtype=torch.bool)
    elif kind.startswith(("revolute", "prismatic")) or kind in {"helical", "translation"}:
        result = torch.ones(joint.nq, dtype=torch.bool)
    else:
        # An out-of-tree manifold is not assumed box-projectable merely because
        # nq == nv. Its coordinates stay unconstrained until it declares semantics.
        result = torch.zeros(joint.nq, dtype=torch.bool)
    return result


def _joint_unit_ranges(joint, offset: int = 0) -> tuple[tuple[int, int], ...]:
    """Return local q-coordinate ranges that must have unit Euclidean norm."""
    kind = joint.kind
    if kind == "free_flyer":
        return ((offset + 3, offset + 7),)
    if kind == "spherical":
        return ((offset, offset + 4),)
    if kind in {"planar", "revolute_unbounded"}:
        start = offset + (2 if kind == "planar" else 0)
        return ((start, start + 2),)
    if kind == "composite":
        ranges: list[tuple[int, int]] = []
        child_offset = offset
        for child in joint.sub_joints:
            ranges.extend(_joint_unit_ranges(child, child_offset))
            child_offset += child.nq
        return tuple(ranges)
    return ()


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
        for joint, iq in zip(self.model.joint_models, self.model.idx_qs):
            if joint.nq:
                mask[iq : iq + joint.nq] = _joint_box_mask(joint)
        return mask

    @property
    def unit_coordinate_slices(self) -> tuple[slice, ...]:
        """Configuration slices representing quaternions or unit circles."""
        ranges: list[tuple[int, int]] = []
        for joint, iq in zip(self.model.joint_models, self.model.idx_qs):
            ranges.extend(_joint_unit_ranges(joint, iq))
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
