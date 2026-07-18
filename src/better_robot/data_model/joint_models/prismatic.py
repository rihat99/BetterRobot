"""Prismatic joint family: axis-aligned ``JointPX``/``JointPY``/``JointPZ``
and unaligned ``JointPrismaticUnaligned``.

See ``docs/concepts/joints_bodies_frames.md §5``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import torch

from ...lie import se3 as _se3


def _prismatic_transform(axis: torch.Tensor, q_slice: torch.Tensor) -> torch.Tensor:
    """Pure-translation SE3. q_slice: (B..., 1) → (B..., 7)."""
    disp = q_slice[..., 0]
    return _se3.from_translation(axis.to(q_slice.device, q_slice.dtype), disp)


def _prismatic_subspace(axis: torch.Tensor, q_slice: torch.Tensor) -> torch.Tensor:
    """Motion subspace S. (B..., 6, 1) — linear part along axis."""
    *batch, _ = q_slice.shape
    S = torch.zeros(*batch, 6, 1, dtype=q_slice.dtype, device=q_slice.device)
    a = axis.to(q_slice.device, q_slice.dtype)
    S[..., 0, 0] = a[0]
    S[..., 1, 0] = a[1]
    S[..., 2, 0] = a[2]
    return S


def _prismatic_velocity(axis: torch.Tensor, q_slice, v_slice) -> torch.Tensor:
    """Spatial velocity = S * v. (B..., 6)."""
    *batch, _ = v_slice.shape
    out = torch.zeros(*batch, 6, dtype=v_slice.dtype, device=v_slice.device)
    a = axis.to(v_slice.device, v_slice.dtype)
    out[..., 0] = a[0] * v_slice[..., 0]
    out[..., 1] = a[1] * v_slice[..., 0]
    out[..., 2] = a[2] * v_slice[..., 0]
    return out


_AXIS_X = torch.tensor([1.0, 0.0, 0.0])
_AXIS_Y = torch.tensor([0.0, 1.0, 0.0])
_AXIS_Z = torch.tensor([0.0, 0.0, 1.0])


class _Prismatic:
    _joint_axis: ClassVar[torch.Tensor]

    def _axis_for_ops(self):
        return self._joint_axis

    def joint_transform(self, q_slice):
        return _prismatic_transform(self._axis_for_ops(), q_slice)

    def joint_motion_subspace(self, q_slice):
        return _prismatic_subspace(self._axis_for_ops(), q_slice)

    def joint_velocity(self, q_slice, v_slice):
        return _prismatic_velocity(self._axis_for_ops(), q_slice, v_slice)

    def integrate(self, q_slice, v_slice):
        return q_slice + v_slice

    def difference(self, q0_slice, q1_slice):
        return q1_slice - q0_slice

    def random_configuration(self, generator, lower, upper):
        return lower + (upper - lower) * torch.rand(1, generator=generator)

    def neutral(self):
        return torch.zeros(1)


@dataclass(frozen=True)
class JointPX(_Prismatic):
    kind: str = "prismatic_px"
    nq: int = 1
    nv: int = 1
    axis: torch.Tensor | None = None
    _joint_axis: ClassVar[torch.Tensor] = _AXIS_X


@dataclass(frozen=True)
class JointPY(_Prismatic):
    kind: str = "prismatic_py"
    nq: int = 1
    nv: int = 1
    axis: torch.Tensor | None = None
    _joint_axis: ClassVar[torch.Tensor] = _AXIS_Y


@dataclass(frozen=True)
class JointPZ(_Prismatic):
    kind: str = "prismatic_pz"
    nq: int = 1
    nv: int = 1
    axis: torch.Tensor | None = None
    _joint_axis: ClassVar[torch.Tensor] = _AXIS_Z


@dataclass(frozen=True)
class JointPrismaticUnaligned(_Prismatic):
    """Prismatic joint with an arbitrary 3-vector axis."""

    axis: torch.Tensor = field(default_factory=lambda: torch.tensor([1.0, 0.0, 0.0]))
    kind: str = "prismatic_unaligned"
    nq: int = 1
    nv: int = 1

    def _axis_for_ops(self):
        return self.axis
