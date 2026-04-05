"""Yoshikawa manipulability residual."""
from __future__ import annotations

import torch

from ..models.robot_model import RobotModel
from .cost_term import CostTerm


def manipulability_residual(
    cfg: torch.Tensor,
    robot: RobotModel,
    target_link_index: int,
    weight: float = 1.0,
) -> torch.Tensor:
    """Penalize low manipulability (inverse Yoshikawa measure).

    Minimizing this residual maximizes manipulability.
    Manipulability = sqrt(det(J @ J^T)) where J is the translation Jacobian.

    Returns:
        Shape (1,). Weighted inverse manipulability.
    """
    raise NotImplementedError
