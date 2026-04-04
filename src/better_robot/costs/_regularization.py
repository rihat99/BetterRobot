"""Rest pose and smoothness regularization residuals."""

from __future__ import annotations

import torch


def rest_residual(
    cfg: torch.Tensor,
    rest_pose: torch.Tensor,
    weight: float = 1.0,
) -> torch.Tensor:
    """Penalize deviation from a rest/default pose.

    Args:
        cfg: Shape (num_actuated_joints,). Current config.
        rest_pose: Shape (num_actuated_joints,). Target rest configuration.
        weight: Scalar weight.

    Returns:
        Shape (num_actuated_joints,). Weighted deviation from rest.
    """
    raise NotImplementedError


def smoothness_residual(
    cfg: torch.Tensor,
    cfg_prev: torch.Tensor,
    weight: float = 1.0,
) -> torch.Tensor:
    """Penalize large configuration changes between timesteps.

    Args:
        cfg: Shape (num_actuated_joints,). Current config.
        cfg_prev: Shape (num_actuated_joints,). Previous config.
        weight: Scalar weight.

    Returns:
        Shape (num_actuated_joints,). Weighted config difference.
    """
    raise NotImplementedError
