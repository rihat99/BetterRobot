"""Yoshikawa manipulability residual."""

from __future__ import annotations

import torch

from ..core._robot import Robot


def manipulability_residual(
    cfg: torch.Tensor,
    robot: Robot,
    target_link_index: int,
    weight: float = 1.0,
) -> torch.Tensor:
    """Penalize low manipulability (inverse Yoshikawa measure).

    Minimizing this residual maximizes manipulability.
    Manipulability = sqrt(det(J @ J^T)) where J is the translation Jacobian.

    Args:
        cfg: Shape (num_actuated_joints,). Current config.
        robot: Robot instance.
        target_link_index: Index of the link to measure manipulability for.
        weight: Scalar weight.

    Returns:
        Shape (1,). Weighted inverse manipulability.
    """
    raise NotImplementedError
