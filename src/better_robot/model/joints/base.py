"""``JointModel``: the abstract joint kind, and what every final kind declares."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import torch

from ..manifolds import Manifold


class JointModel(ABC):
    """A joint kind: the map from its coordinates to the child placement, and the manifold those coordinates live on.

    A kind declares ``nq`` (the width of its configuration slice), ``nv`` (the width of its tangent slice),
    ``manifold`` (how ``kinematics.integrate`` and ``kinematics.difference`` step its coordinates) and ``axis`` (the
    axis in the joint frame of a revolute, prismatic or helical joint, ``None`` for the other kinds). The kinds are
    final frozen dataclasses whose fields are the joint's data (record 0013): two joints with equal fields are equal.
    ``q`` and ``v`` below are the joint's own slices, ``(B..., nq)`` and ``(B..., nv)``; every method broadcasts over
    the leading axes and returns tensors on the dtype and device of ``q``.
    """

    nq: ClassVar[int]
    nv: ClassVar[int]
    manifold: ClassVar[Manifold]
    axis: tuple[float, float, float] | None

    @abstractmethod
    def placement(self, q: torch.Tensor) -> torch.Tensor:
        """Pose of the child frame in the joint frame, ``(B..., nq) -> (B..., 7)``."""

    @abstractmethod
    def motion_subspace(self, q: torch.Tensor) -> torch.Tensor:
        """``S(q)``, ``(B..., nq) -> (B..., 6, nv)``: the child's twist in its own frame is ``S(q) @ v``."""

    @abstractmethod
    def neutral(self) -> torch.Tensor:
        """The configuration ``(nq,)`` with the identity placement, float32 on the CPU."""

    @abstractmethod
    def random(self, generator: torch.Generator | None, *, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
        """A configuration ``(nq,)`` drawn inside ``lower, upper: (nq,)`` where the kind has limits; rotations and
        angles without finite limits are drawn uniformly over their full range."""


def _axis_column(axis: torch.Tensor, batch: torch.Size, *, row: int) -> torch.Tensor:
    """``(B..., 6, 1)`` motion subspace with ``axis: (3,)`` in rows ``row:row + 3`` and zeros elsewhere."""
    column = axis.new_zeros(6)
    column[row : row + 3] = axis
    return column.expand(*batch, 6).unsqueeze(-1)


__all__ = ["JointModel"]
