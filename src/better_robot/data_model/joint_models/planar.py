"""``JointPlanar`` — SE2 + planar position. ``nq = 4`` (x, y, cosθ, sinθ),
``nv = 3``.

See ``docs/concepts/joints_bodies_frames.md §5``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class JointPlanar:
    """SE2 + planar position. q = [x, y, cos θ, sin θ], v = [vx, vy, dθ]."""

    kind: str = "planar"
    nq: int = 4
    nv: int = 3
    axis: torch.Tensor | None = None

    def joint_transform(self, q_slice: torch.Tensor) -> torch.Tensor:
        """SE3 with translation [x, y, 0] and rotation θ about Z.

        q_slice: (B..., 4) = [x, y, cos θ, sin θ] → (B..., 7).
        """
        x = q_slice[..., 0]
        y = q_slice[..., 1]
        cos_t = q_slice[..., 2]
        sin_t = q_slice[..., 3]
        # Quaternion for rotation by θ about Z: [0, 0, sin(θ/2), cos(θ/2)]
        # cos(θ/2) = sqrt((1+cos_t)/2), sin(θ/2) = sqrt((1-cos_t)/2)*sign(sin_t)
        half_cos = torch.sqrt(((1.0 + cos_t) / 2.0).clamp(min=0.0))
        half_sin = torch.sqrt(((1.0 - cos_t) / 2.0).clamp(min=0.0)) * torch.sign(sin_t)
        z = torch.zeros_like(x)
        zeros3 = torch.zeros(*x.shape, 3, dtype=q_slice.dtype, device=q_slice.device)
        # SE3: [tx, ty, tz=0, qx=0, qy=0, qz=half_sin, qw=half_cos]
        t = torch.stack([x, y, z], dim=-1)
        q = torch.stack([z, z, half_sin, half_cos], dim=-1)
        return torch.cat([t, q], dim=-1)

    def joint_motion_subspace(self, q_slice: torch.Tensor) -> torch.Tensor:
        """(B..., 6, 3) — 2 linear (x,y) + 1 angular (z)."""
        *batch, _ = q_slice.shape
        S = torch.zeros(*batch, 6, 3, dtype=q_slice.dtype, device=q_slice.device)
        S[..., 0, 0] = 1.0  # linear x
        S[..., 1, 1] = 1.0  # linear y
        S[..., 5, 2] = 1.0  # angular z
        return S

    def joint_velocity(self, q_slice, v_slice) -> torch.Tensor:
        """[vx, vy, 0, 0, 0, dθ]. (B..., 6)."""
        *batch, _ = v_slice.shape
        out = torch.zeros(*batch, 6, dtype=v_slice.dtype, device=v_slice.device)
        out[..., 0] = v_slice[..., 0]
        out[..., 1] = v_slice[..., 1]
        out[..., 5] = v_slice[..., 2]
        return out

    def integrate(self, q_slice: torch.Tensor, v_slice: torch.Tensor) -> torch.Tensor:
        """Planar retraction. v = [vx, vy, dθ]."""
        x   = q_slice[..., 0] + v_slice[..., 0]
        y   = q_slice[..., 1] + v_slice[..., 1]
        cos_t = q_slice[..., 2]
        sin_t = q_slice[..., 3]
        theta = torch.atan2(sin_t, cos_t) + v_slice[..., 2]
        return torch.stack([x, y, torch.cos(theta), torch.sin(theta)], dim=-1)

    def difference(self, q0_slice: torch.Tensor, q1_slice: torch.Tensor) -> torch.Tensor:
        """[x1-x0, y1-y0, θ1-θ0]. (B..., 3)."""
        dx = q1_slice[..., 0] - q0_slice[..., 0]
        dy = q1_slice[..., 1] - q0_slice[..., 1]
        th0 = torch.atan2(q0_slice[..., 3], q0_slice[..., 2])
        th1 = torch.atan2(q1_slice[..., 3], q1_slice[..., 2])
        return torch.stack([dx, dy, th1 - th0], dim=-1)

    def random_configuration(self, generator, lower, upper) -> torch.Tensor:
        """Random planar config. lower/upper only apply to (x,y); θ is full circle."""
        x = lower[0] + (upper[0] - lower[0]) * torch.rand(1, generator=generator)
        y = lower[1] + (upper[1] - lower[1]) * torch.rand(1, generator=generator)
        theta = torch.rand(1, generator=generator) * 2 * 3.14159265 - 3.14159265
        return torch.cat([x, y, torch.cos(theta), torch.sin(theta)])

    def neutral(self) -> torch.Tensor:
        """Zero displacement, θ=0."""
        return torch.tensor([0.0, 0.0, 1.0, 0.0])
