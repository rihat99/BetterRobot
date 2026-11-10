"""``JointModel``: the abstract joint kind, and what every final kind declares."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import ClassVar

import torch

from ...exceptions import ArgumentError
from ..manifolds import JointManifold


class JointModel(ABC):
    """A joint kind: the map from its coordinates to the child placement, and the manifold those coordinates live on.

    A kind declares ``nq`` (the width of its configuration slice), ``nv`` (the width of its tangent slice),
    ``manifold`` (how ``kinematics.integrate`` and ``kinematics.difference`` step its coordinates) and ``axis`` (the
    axis in the joint frame of a revolute, prismatic or helical joint, a tuple of three finite floats of unit norm
    checked by `check_axis` at construction, ``None`` for the other kinds). The kinds are final frozen dataclasses
    whose fields are the joint's data (record 0013): two joints with equal fields are equal.
    ``q`` and ``v`` below are the joint's own slices, ``(B..., nq)`` and ``(B..., nv)``; every method broadcasts over
    the leading axes and returns tensors on the dtype and device of ``q``.
    """

    nq: ClassVar[int]
    nv: ClassVar[int]
    manifold: ClassVar[JointManifold]
    axis: tuple[float, float, float] | None

    @abstractmethod
    def placement(self, q: torch.Tensor) -> torch.Tensor:
        """Pose of the child frame in the joint frame, ``(B..., nq) -> (B..., 7)``."""

    @abstractmethod
    def motion_subspace(self, q: torch.Tensor) -> torch.Tensor:
        """``S``, ``(B..., nq) -> (B..., 6, nv)``: the child's twist in its own frame is ``S @ v``.

        ``S`` is constant in the child frame: the kinematics and dynamics plans read it once, at ``neutral()``, and
        `better_robot.testing.check_joint_model` asserts ``motion_subspace(q) == motion_subspace(neutral())``. A
        motion whose subspace moves with ``q`` in the child frame is not a joint kind of this library.
        """

    @abstractmethod
    def neutral(self) -> torch.Tensor:
        """The configuration ``(nq,)`` with the identity placement, float32 on the CPU."""

    @abstractmethod
    def random(self, generator: torch.Generator, *, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
        """A configuration ``(nq,)`` in the limits' dtype, drawn on their device with ``generator`` (which lives
        there) inside ``lower, upper: (nq,)`` where the kind has limits; rotations and angles without finite limits
        are drawn uniformly over their full range."""


def check_axis(axis: object) -> None:
    """Raise ``ArgumentError`` unless ``axis`` is a tuple of three finite floats with unit norm (within ``1e-6``).

    A list or a tensor is rejected too: the kinds are hashable values, and a non-unit axis would scale the placement
    and the Jacobian column it builds.
    """
    if not (isinstance(axis, tuple) and len(axis) == 3) or not all(
        isinstance(c, (int, float)) and not isinstance(c, bool) and math.isfinite(c) for c in axis
    ):
        raise ArgumentError(f"axis must be a tuple of three finite floats, got {axis!r}; pass tuple(axis)")
    norm = math.sqrt(sum(c * c for c in axis))
    if abs(norm - 1.0) > 1e-6:
        raise ArgumentError(f"axis {axis!r} has norm {norm:.8g}; pass a unit axis (divide by its norm)")


def _axis_column(axis: torch.Tensor, batch: torch.Size, *, row: int) -> torch.Tensor:
    """``(B..., 6, 1)`` motion subspace with ``axis: (3,)`` in rows ``row:row + 3`` and zeros elsewhere."""
    column = axis.new_zeros(6)
    column[row : row + 3] = axis
    return column.expand(*batch, 6).unsqueeze(-1)


__all__ = ["JointModel"]
