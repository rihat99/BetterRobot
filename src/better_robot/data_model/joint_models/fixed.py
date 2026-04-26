"""``JointFixed`` / ``JointUniverse`` — zero-DOF joints.

See ``docs/concepts/joints_bodies_frames.md §5``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class JointFixed:
    """Zero-DOF rigid link to parent. ``nq = nv = 0``."""

    kind: str = "fixed"
    nq: int = 0
    nv: int = 0
    axis: torch.Tensor | None = None

    def joint_transform(self, q_slice: torch.Tensor) -> torch.Tensor:
        """Returns identity SE3 (the fixed offset is in model.joint_placements)."""
        # q_slice is empty (nq=0); return a scalar identity SE3
        return torch.tensor([0., 0., 0., 0., 0., 0., 1.],
                             dtype=torch.float32)

    def joint_motion_subspace(self, q_slice: torch.Tensor) -> torch.Tensor:
        """Returns (6, 0) empty subspace."""
        return torch.zeros(6, 0)

    def joint_velocity(self, q_slice, v_slice) -> torch.Tensor:
        """Returns zero 6-vector (no DOF)."""
        return torch.zeros(6)

    def integrate(self, q_slice, v_slice) -> torch.Tensor:
        """No DOF — returns empty tensor."""
        return q_slice.clone()

    def difference(self, q0_slice, q1_slice) -> torch.Tensor:
        """No DOF — returns empty tensor."""
        return torch.zeros(0, dtype=q0_slice.dtype, device=q0_slice.device)

    def random_configuration(self, generator, lower, upper) -> torch.Tensor:
        """Returns empty tensor."""
        return torch.zeros(0)

    def neutral(self) -> torch.Tensor:
        """Returns empty tensor."""
        return torch.zeros(0)


@dataclass(frozen=True)
class JointUniverse:
    """Root placeholder joint (joint 0). Zero DOF, identity transform."""

    kind: str = "universe"
    nq: int = 0
    nv: int = 0
    axis: torch.Tensor | None = None

    def joint_transform(self, q_slice: torch.Tensor) -> torch.Tensor:
        return torch.tensor([0., 0., 0., 0., 0., 0., 1.], dtype=torch.float32)

    def joint_motion_subspace(self, q_slice: torch.Tensor) -> torch.Tensor:
        return torch.zeros(6, 0)

    def joint_velocity(self, q_slice, v_slice) -> torch.Tensor:
        return torch.zeros(6)

    def integrate(self, q_slice, v_slice) -> torch.Tensor:
        return q_slice.clone()

    def difference(self, q0_slice, q1_slice) -> torch.Tensor:
        return torch.zeros(0, dtype=q0_slice.dtype, device=q0_slice.device)

    def random_configuration(self, generator, lower, upper) -> torch.Tensor:
        return torch.zeros(0)

    def neutral(self) -> torch.Tensor:
        return torch.zeros(0)
