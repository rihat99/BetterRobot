"""``JointMimic`` — legacy zero-DOF placeholder.

Real mimic relationships retain the target's concrete scalar joint model and
use the model-level reduced coordinate map. The loader rejects this placeholder
because it does not encode the target motion semantics.

See ``docs/concepts/joints_bodies_frames.md §5``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class JointMimic:
    """Zero-DOF placeholder retained for import compatibility only."""

    kind: str = "mimic"
    nq: int = 0
    nv: int = 0
    axis: torch.Tensor | None = None

    def joint_transform(self, q_slice: torch.Tensor) -> torch.Tensor:
        """Identity — the mimic offset is encoded in model.joint_placements."""
        return torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=torch.float32)

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
