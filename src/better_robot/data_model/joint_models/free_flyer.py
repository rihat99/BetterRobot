"""``JointFreeFlyer`` — SE3 floating base. ``nq = 7`` (``[x,y,z,qx,qy,qz,qw]``),
``nv = 6`` (``[vx,vy,vz,wx,wy,wz]``).

This replaces the legacy fixed-vs-floating base split. A humanoid's free
base becomes ``joint_models[1] = JointFreeFlyer()`` and its seven SE3
components occupy ``q[0:7]`` — the rest of the library never sees it as a
special case.

See ``docs/concepts/joints_bodies_frames.md §5`` and ``docs/concepts/tasks.md §1``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ...lie import se3 as _se3
from ...lie import so3 as _so3


@dataclass(frozen=True)
class JointFreeFlyer:
    """SE3 free-flyer root joint — 6-DOF floating base on a 7-component manifold."""

    kind: str = "free_flyer"
    nq: int = 7
    nv: int = 6
    axis: torch.Tensor | None = None

    def joint_transform(self, q_slice: torch.Tensor) -> torch.Tensor:
        """SE3 tensor from the 7-component configuration slice.

        q_slice = (B..., 7) = [tx, ty, tz, qx, qy, qz, qw] → (B..., 7) (same, normalized).
        """
        return _se3.normalize(q_slice)

    def joint_motion_subspace(self, q_slice: torch.Tensor) -> torch.Tensor:
        """(B..., 6, 6) identity — all 6 DOFs are free."""
        *batch, _ = q_slice.shape
        return torch.eye(6, dtype=q_slice.dtype, device=q_slice.device).expand(*batch, 6, 6)

    def joint_velocity(self, q_slice, v_slice) -> torch.Tensor:
        """Spatial velocity = v_slice directly. (B..., 6)."""
        return v_slice.clone()

    def integrate(self, q_slice: torch.Tensor, v_slice: torch.Tensor) -> torch.Tensor:
        """SE3 retraction: q_new = q ⊕ v = compose(q, exp(v))."""
        delta = _se3.exp(v_slice)       # (B..., 7)
        q_new = _se3.compose(q_slice, delta)
        return _se3.normalize(q_new)

    def difference(self, q0_slice: torch.Tensor, q1_slice: torch.Tensor) -> torch.Tensor:
        """SE3 log difference: log(q0^{-1} * q1). (B..., 6)."""
        q0_inv = _se3.inverse(q0_slice)
        diff = _se3.compose(q0_inv, q1_slice)
        return _se3.log(diff)

    def random_configuration(self, generator, lower, upper) -> torch.Tensor:
        """Random SE3: unit Gaussian translation + random unit quaternion."""
        t = torch.randn(3, generator=generator)
        q = torch.randn(4, generator=generator)
        q = q / q.norm().clamp(min=1e-8)
        return torch.cat([t, q])

    def neutral(self) -> torch.Tensor:
        """Identity SE3: zero translation, identity quaternion."""
        return torch.tensor([0., 0., 0., 0., 0., 0., 1.])
