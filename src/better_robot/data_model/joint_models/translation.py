"""``JointTranslation`` — 3-DOF pure translation. ``nq = nv = 3``.

See ``docs/concepts/joints_bodies_frames.md §5``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class JointTranslation:
    """3-DOF translation joint. q = [x, y, z], v = [vx, vy, vz]."""

    kind: str = "translation"
    nq: int = 3
    nv: int = 3
    axis: torch.Tensor | None = None

    def joint_transform(self, q_slice: torch.Tensor) -> torch.Tensor:
        """Pure translation SE3. (B..., 3) → (B..., 7)."""
        zeros3 = torch.zeros(*q_slice.shape[:-1], 3, dtype=q_slice.dtype, device=q_slice.device)
        ones   = torch.ones(*q_slice.shape[:-1], 1,  dtype=q_slice.dtype, device=q_slice.device)
        return torch.cat([q_slice, zeros3, ones], dim=-1)

    def joint_motion_subspace(self, q_slice: torch.Tensor) -> torch.Tensor:
        """(B..., 6, 3) — linear 3 DOF."""
        *batch, _ = q_slice.shape
        S = torch.zeros(*batch, 6, 3, dtype=q_slice.dtype, device=q_slice.device)
        S[..., 0, 0] = 1.0
        S[..., 1, 1] = 1.0
        S[..., 2, 2] = 1.0
        return S

    def joint_velocity(self, q_slice, v_slice) -> torch.Tensor:
        """[vx, vy, vz, 0, 0, 0]. (B..., 6)."""
        *batch, _ = v_slice.shape
        out = torch.zeros(*batch, 6, dtype=v_slice.dtype, device=v_slice.device)
        out[..., :3] = v_slice
        return out

    def integrate(self, q_slice, v_slice) -> torch.Tensor:
        return q_slice + v_slice

    def difference(self, q0_slice, q1_slice) -> torch.Tensor:
        return q1_slice - q0_slice

    def random_configuration(self, generator, lower, upper) -> torch.Tensor:
        return lower + (upper - lower) * torch.rand(3, generator=generator)

    def neutral(self) -> torch.Tensor:
        return torch.zeros(3)
