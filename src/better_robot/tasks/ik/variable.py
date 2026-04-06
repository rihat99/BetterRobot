"""IKVariable: unified variable for fixed and floating-base IK."""
from __future__ import annotations
from dataclasses import dataclass
import torch


@dataclass
class IKVariable:
    """Represents the optimization variables for IK.

    For fixed-base IK: q is (n,), base_pose is None.
    For floating-base IK: q is (n,), base_pose is (7,).

    The flat representation concatenates [q (n,), base_tangent (6,)] for floating-base,
    or just [q (n,)] for fixed-base.
    """

    q: torch.Tensor
    """Shape (n,). Joint configuration."""

    base_pose: torch.Tensor | None = None
    """Shape (7,). Optional floating base SE3 pose [tx,ty,tz,qx,qy,qz,qw]."""

    def is_floating(self) -> bool:
        """True if this is a floating-base variable."""
        return self.base_pose is not None

    def to_flat(self) -> torch.Tensor:
        """Flatten to optimization variable.

        Fixed-base: returns q (n,)
        Floating-base: returns cat([q, zeros(6)]) — base tangent starts at zero
        """
        if self.base_pose is None:
            return self.q
        return torch.cat([self.q, torch.zeros(6, dtype=self.q.dtype, device=self.q.device)])

    @staticmethod
    def from_flat(
        flat: torch.Tensor,
        base_pose: torch.Tensor | None,
    ) -> "IKVariable":
        """Reconstruct IKVariable from flat vector.

        For fixed-base: flat is (n,), base_pose is None.
        For floating-base: flat is (n+6,), base_pose is (7,).
            The base is updated by SE3 retraction: new_base = exp(flat[n:]) @ base_pose
        """
        if base_pose is None:
            return IKVariable(q=flat, base_pose=None)
        from ...math.se3 import se3_compose, se3_exp
        n = flat.shape[0] - 6
        q = flat[:n]
        delta_base = flat[n:]
        new_base = se3_compose(se3_exp(delta_base), base_pose)
        # Normalize quaternion
        quat = new_base[3:7]
        quat_norm = quat / quat.norm().clamp(min=1e-8)
        new_base = torch.cat([new_base[:3], quat_norm])
        return IKVariable(q=q, base_pose=new_base)
