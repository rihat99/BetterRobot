"""``JointSpherical`` — SO3 ball joint. ``nq = 4`` (``[qx,qy,qz,qw]``),
``nv = 3``.

See ``docs/02_DATA_MODEL.md §5``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ...lie import so3 as _so3


@dataclass(frozen=True)
class JointSpherical:
    """SO3 ball joint — 3-DOF rotation on a 4-component quaternion manifold."""

    kind: str = "spherical"
    nq: int = 4
    nv: int = 3
    axis: torch.Tensor | None = None

    def joint_transform(self, q_slice: torch.Tensor) -> torch.Tensor:
        """q_slice = (B..., 4) quaternion [qx,qy,qz,qw] → SE3 (B..., 7).

        Pure rotation: translation = 0.
        """
        # Normalize to be safe
        q_n = _so3.normalize(q_slice)
        zeros = torch.zeros(*q_slice.shape[:-1], 3,
                             dtype=q_slice.dtype, device=q_slice.device)
        return torch.cat([zeros, q_n], dim=-1)

    def joint_motion_subspace(self, q_slice: torch.Tensor) -> torch.Tensor:
        """(B..., 6, 3) — angular 3 DOF (bottom 3 rows are identity)."""
        *batch, _ = q_slice.shape
        S = torch.zeros(*batch, 6, 3, dtype=q_slice.dtype, device=q_slice.device)
        S[..., 3, 0] = 1.0
        S[..., 4, 1] = 1.0
        S[..., 5, 2] = 1.0
        return S

    def joint_velocity(self, q_slice, v_slice) -> torch.Tensor:
        """Spatial velocity = [0, 0, 0, vx, vy, vz]. (B..., 6)."""
        *batch, _ = v_slice.shape
        out = torch.zeros(*batch, 6, dtype=v_slice.dtype, device=v_slice.device)
        out[..., 3:] = v_slice
        return out

    def integrate(self, q_slice: torch.Tensor, v_slice: torch.Tensor) -> torch.Tensor:
        """Quaternion retraction: q_new = normalize(q * exp_so3(v))."""
        dq = _so3.exp(v_slice)                    # (B..., 4)
        q_new = _so3.compose(q_slice, dq)         # (B..., 4)
        return _so3.normalize(q_new)

    def difference(self, q0_slice: torch.Tensor, q1_slice: torch.Tensor) -> torch.Tensor:
        """Log difference: log_so3(q0^{-1} * q1). (B..., 3)."""
        q0_inv = _so3.inverse(q0_slice)
        dq = _so3.compose(q0_inv, q1_slice)
        return _so3.log(_so3.normalize(dq))

    def random_configuration(self, generator, lower, upper) -> torch.Tensor:
        """Random unit quaternion via Gaussian sampling."""
        q = torch.randn(4, generator=generator)
        return q / q.norm().clamp(min=1e-8)

    def neutral(self) -> torch.Tensor:
        """Identity rotation: [0, 0, 0, 1]."""
        return torch.tensor([0.0, 0.0, 0.0, 1.0])
