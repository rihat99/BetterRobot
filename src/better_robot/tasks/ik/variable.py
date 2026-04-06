"""IKVariable: unified variable for fixed and floating-base IK."""
from __future__ import annotations
from dataclasses import dataclass
import torch


@dataclass
class IKVariable:
    """Represents the optimization variables for IK.

    For fixed-base IK: cfg is (n,), base_pose is None.
    For floating-base IK: cfg is (n,), base_pose is (7,).

    The flat representation concatenates [cfg (n,), base_tangent (6,)] for floating-base,
    or just [cfg (n,)] for fixed-base.
    """

    cfg: torch.Tensor
    """Shape (n,). Joint configuration."""

    base_pose: torch.Tensor | None = None
    """Shape (7,). Optional floating base SE3 pose [tx,ty,tz,qx,qy,qz,qw]."""

    def is_floating(self) -> bool:
        """True if this is a floating-base variable."""
        return self.base_pose is not None

    def to_flat(self) -> torch.Tensor:
        """Flatten to optimization variable.

        Fixed-base: returns cfg (n,)
        Floating-base: returns cat([cfg, zeros(6)]) — base tangent starts at zero
        """
        if self.base_pose is None:
            return self.cfg
        return torch.cat([self.cfg, torch.zeros(6, dtype=self.cfg.dtype, device=self.cfg.device)])

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
            return IKVariable(cfg=flat, base_pose=None)
        from ...math.se3 import se3_compose, se3_exp
        n = flat.shape[0] - 6
        cfg = flat[:n]
        delta_base = flat[n:]
        new_base = se3_compose(se3_exp(delta_base), base_pose)
        # Normalize quaternion
        q = new_base[3:7]
        q_norm = q / q.norm().clamp(min=1e-8)
        new_base = torch.cat([new_base[:3], q_norm])
        return IKVariable(cfg=cfg, base_pose=new_base)
