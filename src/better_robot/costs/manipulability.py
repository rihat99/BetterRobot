"""Yoshikawa manipulability residual."""
from __future__ import annotations

import functools

import torch

from ..models.robot_model import RobotModel
from ..algorithms.kinematics.jacobian import compute_jacobian
from .cost_term import CostTerm


def manipulability_residual(
    q: torch.Tensor,
    model: RobotModel,
    target_link_index: int,
    weight: float = 1.0,
) -> torch.Tensor:
    """Penalize low manipulability (inverse Yoshikawa measure).

    Minimizing this residual maximizes manipulability.
    Manipulability = sqrt(det(J_pos @ J_pos^T)) where J_pos is the 3xN position Jacobian.

    Returns:
        Shape (1,). Weighted inverse manipulability.
    """
    # Compute Jacobian with unit weights so we get the geometric Jacobian
    J = compute_jacobian(
        model, q, target_link_index,
        target_pose=torch.zeros(7, dtype=q.dtype, device=q.device),
        pos_weight=1.0,
        ori_weight=0.0,
    )  # (6, n) — but with ori_weight=0, rows 3:6 are zero
    J_pos = J[:3, :]  # (3, n) — position Jacobian only
    JJt = J_pos @ J_pos.T  # (3, 3)
    manip = torch.sqrt(torch.linalg.det(JJt).clamp(min=1e-10))
    return torch.tensor([weight / manip], dtype=q.dtype, device=q.device)


def manipulability_cost(
    model: RobotModel,
    target_link_index: int,
    weight: float = 1.0,
) -> CostTerm:
    """Factory: returns CostTerm penalizing low manipulability."""
    return CostTerm(
        residual_fn=functools.partial(
            manipulability_residual,
            model=model,
            target_link_index=target_link_index,
            weight=weight,
        ),
        weight=1.0,
    )
