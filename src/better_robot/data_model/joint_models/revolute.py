"""Revolute joint family: ``JointRX``/``JointRY``/``JointRZ`` (axis-aligned),
``JointRevoluteUnaligned`` (arbitrary 3-vector axis), and
``JointRevoluteUnbounded`` (continuous, ``(cos θ, sin θ)`` parameterisation).

See ``docs/concepts/joints_bodies_frames.md §5``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ...lie import se3 as _se3
from ...lie import so3 as _so3


def _revolute_transform(axis: torch.Tensor, q_slice: torch.Tensor) -> torch.Tensor:
    """Pure-rotation SE3 for any revolute axis. q_slice: (B..., 1) → (B..., 7)."""
    angle = q_slice[..., 0]   # (B...)
    return _se3.from_axis_angle(axis.to(q_slice.device, q_slice.dtype), angle)


def _revolute_subspace(axis: torch.Tensor, q_slice: torch.Tensor) -> torch.Tensor:
    """Motion subspace S. (B..., 6, 1) — angular part along axis."""
    # S = [0, 0, 0, ax, ay, az]^T
    *batch, _ = q_slice.shape
    S = torch.zeros(*batch, 6, 1, dtype=q_slice.dtype, device=q_slice.device)
    S[..., 3, 0] = axis[0].to(q_slice.device, q_slice.dtype)
    S[..., 4, 0] = axis[1].to(q_slice.device, q_slice.dtype)
    S[..., 5, 0] = axis[2].to(q_slice.device, q_slice.dtype)
    return S


def _revolute_velocity(axis: torch.Tensor, q_slice, v_slice) -> torch.Tensor:
    """Spatial velocity = S * v. (B..., 6)."""
    *batch, _ = v_slice.shape
    out = torch.zeros(*batch, 6, dtype=v_slice.dtype, device=v_slice.device)
    a = axis.to(v_slice.device, v_slice.dtype)
    out[..., 3] = a[0] * v_slice[..., 0]
    out[..., 4] = a[1] * v_slice[..., 0]
    out[..., 5] = a[2] * v_slice[..., 0]
    return out


def _revolute_integrate(q_slice, v_slice) -> torch.Tensor:
    """q ⊕ v = q + v (simple angle addition)."""
    return q_slice + v_slice


def _revolute_difference(q0_slice, q1_slice) -> torch.Tensor:
    """q1 ⊖ q0 = q1 - q0."""
    return q1_slice - q0_slice


def _revolute_random(generator, lower, upper) -> torch.Tensor:
    """Uniform sample in [lower, upper]."""
    return lower + (upper - lower) * torch.rand(1, generator=generator)


# ──────────────────────────── axis-aligned ────────────────────────────

_AXIS_X = torch.tensor([1.0, 0.0, 0.0])
_AXIS_Y = torch.tensor([0.0, 1.0, 0.0])
_AXIS_Z = torch.tensor([0.0, 0.0, 1.0])


@dataclass(frozen=True)
class JointRX:
    kind: str = "revolute_rx"
    nq: int = 1
    nv: int = 1
    axis: torch.Tensor | None = None

    def joint_transform(self, q_slice):
        return _revolute_transform(_AXIS_X, q_slice)

    def joint_motion_subspace(self, q_slice):
        return _revolute_subspace(_AXIS_X, q_slice)

    def joint_velocity(self, q_slice, v_slice):
        return _revolute_velocity(_AXIS_X, q_slice, v_slice)

    def integrate(self, q_slice, v_slice):
        return _revolute_integrate(q_slice, v_slice)

    def difference(self, q0_slice, q1_slice):
        return _revolute_difference(q0_slice, q1_slice)

    def random_configuration(self, generator, lower, upper):
        return _revolute_random(generator, lower, upper)

    def neutral(self):
        return torch.zeros(1)


@dataclass(frozen=True)
class JointRY:
    kind: str = "revolute_ry"
    nq: int = 1
    nv: int = 1
    axis: torch.Tensor | None = None

    def joint_transform(self, q_slice):
        return _revolute_transform(_AXIS_Y, q_slice)

    def joint_motion_subspace(self, q_slice):
        return _revolute_subspace(_AXIS_Y, q_slice)

    def joint_velocity(self, q_slice, v_slice):
        return _revolute_velocity(_AXIS_Y, q_slice, v_slice)

    def integrate(self, q_slice, v_slice):
        return _revolute_integrate(q_slice, v_slice)

    def difference(self, q0_slice, q1_slice):
        return _revolute_difference(q0_slice, q1_slice)

    def random_configuration(self, generator, lower, upper):
        return _revolute_random(generator, lower, upper)

    def neutral(self):
        return torch.zeros(1)


@dataclass(frozen=True)
class JointRZ:
    kind: str = "revolute_rz"
    nq: int = 1
    nv: int = 1
    axis: torch.Tensor | None = None

    def joint_transform(self, q_slice):
        return _revolute_transform(_AXIS_Z, q_slice)

    def joint_motion_subspace(self, q_slice):
        return _revolute_subspace(_AXIS_Z, q_slice)

    def joint_velocity(self, q_slice, v_slice):
        return _revolute_velocity(_AXIS_Z, q_slice, v_slice)

    def integrate(self, q_slice, v_slice):
        return _revolute_integrate(q_slice, v_slice)

    def difference(self, q0_slice, q1_slice):
        return _revolute_difference(q0_slice, q1_slice)

    def random_configuration(self, generator, lower, upper):
        return _revolute_random(generator, lower, upper)

    def neutral(self):
        return torch.zeros(1)


# ──────────────────────────── unaligned ────────────────────────────


@dataclass(frozen=True)
class JointRevoluteUnaligned:
    """Revolute joint with an arbitrary 3-vector axis."""

    axis: torch.Tensor = field(default_factory=lambda: torch.tensor([1.0, 0.0, 0.0]))
    kind: str = "revolute_unaligned"
    nq: int = 1
    nv: int = 1

    def joint_transform(self, q_slice):
        return _revolute_transform(self.axis, q_slice)

    def joint_motion_subspace(self, q_slice):
        return _revolute_subspace(self.axis, q_slice)

    def joint_velocity(self, q_slice, v_slice):
        return _revolute_velocity(self.axis, q_slice, v_slice)

    def integrate(self, q_slice, v_slice):
        return _revolute_integrate(q_slice, v_slice)

    def difference(self, q0_slice, q1_slice):
        return _revolute_difference(q0_slice, q1_slice)

    def random_configuration(self, generator, lower, upper):
        return _revolute_random(generator, lower, upper)

    def neutral(self):
        return torch.zeros(1)


# ──────────────────────────── unbounded ────────────────────────────


@dataclass(frozen=True)
class JointRevoluteUnbounded:
    """Continuous revolute (no joint limit); ``q`` = ``[cos θ, sin θ]``.

    ``nq = 2``, ``nv = 1``. No position limit (unit circle).
    """

    axis: torch.Tensor = field(default_factory=lambda: torch.tensor([0.0, 0.0, 1.0]))
    kind: str = "revolute_unbounded"
    nq: int = 2
    nv: int = 1

    def joint_transform(self, q_slice: torch.Tensor) -> torch.Tensor:
        """q_slice = (B..., 2) = [cos θ, sin θ] → SE3 rotation by θ."""
        cos_t = q_slice[..., 0]
        sin_t = q_slice[..., 1]
        # angle = atan2(sin_t, cos_t)
        angle = torch.atan2(sin_t, cos_t)
        return _revolute_transform(self.axis, angle.unsqueeze(-1))

    def joint_motion_subspace(self, q_slice):
        # Same 6×1 angular subspace
        angle = torch.atan2(q_slice[..., 1], q_slice[..., 0])
        return _revolute_subspace(self.axis, angle.unsqueeze(-1))

    def joint_velocity(self, q_slice, v_slice):
        angle = torch.atan2(q_slice[..., 1], q_slice[..., 0])
        return _revolute_velocity(self.axis, angle.unsqueeze(-1), v_slice)

    def integrate(self, q_slice: torch.Tensor, v_slice: torch.Tensor) -> torch.Tensor:
        """Retraction on unit circle: new_angle = old_angle + dtheta."""
        cos_t = q_slice[..., 0]
        sin_t = q_slice[..., 1]
        dth = v_slice[..., 0]
        theta = torch.atan2(sin_t, cos_t)
        new_theta = theta + dth
        return torch.stack([torch.cos(new_theta), torch.sin(new_theta)], dim=-1)

    def difference(self, q0_slice: torch.Tensor, q1_slice: torch.Tensor) -> torch.Tensor:
        """θ1 - θ0 extracted from (cos,sin) representation."""
        th0 = torch.atan2(q0_slice[..., 1], q0_slice[..., 0])
        th1 = torch.atan2(q1_slice[..., 1], q1_slice[..., 0])
        return (th1 - th0).unsqueeze(-1)

    def random_configuration(self, generator, lower, upper) -> torch.Tensor:
        """Uniform angle on the full circle [−π, π]."""
        theta = (torch.rand(1, generator=generator) * 2 * 3.14159265358979 - 3.14159265358979)
        return torch.cat([torch.cos(theta), torch.sin(theta)], dim=-1)

    def neutral(self) -> torch.Tensor:
        """Identity: θ = 0 → [1, 0]."""
        return torch.tensor([1.0, 0.0])
