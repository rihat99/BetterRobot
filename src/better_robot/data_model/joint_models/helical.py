"""``JointHelical`` — pitch-coupled rotation + translation. ``nq = nv = 1``.

See ``docs/concepts/joints_bodies_frames.md §5``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass(frozen=True)
class JointHelical:
    """Pitch-coupled rotation + translation. nq = nv = 1.

    T(q) = rotation(q, axis) composed with translation(q * pitch, axis).
    """

    axis: torch.Tensor = field(default_factory=lambda: torch.tensor([0.0, 0.0, 1.0]))
    pitch: float = 0.0  # translation per radian
    kind: str = "helical"
    nq: int = 1
    nv: int = 1

    def joint_transform(self, q_slice: torch.Tensor) -> torch.Tensor:
        """Combined rotation + translation. q_slice: (B..., 1) → (B..., 7)."""
        from ...lie import se3 as _se3
        angle = q_slice[..., 0]
        axis = self.axis.to(q_slice.device, q_slice.dtype)
        # rotation part
        T_rot = _se3.from_axis_angle(axis, angle)
        # translation part: disp = pitch * angle along axis
        if self.pitch != 0.0:
            disp = angle * float(self.pitch)
            T_trans = _se3.from_translation(axis, disp)
            return _se3.compose(T_trans, T_rot)
        return T_rot

    def joint_motion_subspace(self, q_slice: torch.Tensor) -> torch.Tensor:
        """S = [pitch*axis, axis]^T. (B..., 6, 1)."""
        *batch, _ = q_slice.shape
        S = torch.zeros(*batch, 6, 1, dtype=q_slice.dtype, device=q_slice.device)
        a = self.axis.to(q_slice.device, q_slice.dtype)
        # linear part = pitch * axis
        S[..., 0, 0] = float(self.pitch) * a[0]
        S[..., 1, 0] = float(self.pitch) * a[1]
        S[..., 2, 0] = float(self.pitch) * a[2]
        # angular part = axis
        S[..., 3, 0] = a[0]
        S[..., 4, 0] = a[1]
        S[..., 5, 0] = a[2]
        return S

    def joint_velocity(self, q_slice, v_slice) -> torch.Tensor:
        """S * v. (B..., 6)."""
        *batch, _ = v_slice.shape
        out = torch.zeros(*batch, 6, dtype=v_slice.dtype, device=v_slice.device)
        a = self.axis.to(v_slice.device, v_slice.dtype)
        p = float(self.pitch)
        out[..., 0] = p * a[0] * v_slice[..., 0]
        out[..., 1] = p * a[1] * v_slice[..., 0]
        out[..., 2] = p * a[2] * v_slice[..., 0]
        out[..., 3] = a[0] * v_slice[..., 0]
        out[..., 4] = a[1] * v_slice[..., 0]
        out[..., 5] = a[2] * v_slice[..., 0]
        return out

    def integrate(self, q_slice, v_slice) -> torch.Tensor:
        return q_slice + v_slice

    def difference(self, q0_slice, q1_slice) -> torch.Tensor:
        return q1_slice - q0_slice

    def random_configuration(self, generator, lower, upper) -> torch.Tensor:
        return lower + (upper - lower) * torch.rand(1, generator=generator)

    def neutral(self) -> torch.Tensor:
        return torch.zeros(1)
