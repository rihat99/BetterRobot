"""Rest pose and smoothness regularization residuals."""
from __future__ import annotations

import functools

import torch

from .cost_term import CostTerm


def rest_residual(
    cfg: torch.Tensor,
    rest_pose: torch.Tensor,
    weight: float = 1.0,
) -> torch.Tensor:
    """Penalize deviation from a rest/default pose.

    Args:
        cfg: Shape (num_actuated_joints,).
        rest_pose: Shape (num_actuated_joints,). Target rest configuration.
        weight: Scalar weight.

    Returns:
        Shape (num_actuated_joints,). Weighted deviation from rest.
    """
    return (cfg - rest_pose.to(device=cfg.device, dtype=cfg.dtype)) * weight


def rest_cost(rest_pose: torch.Tensor, weight: float = 0.01) -> CostTerm:
    """Create a rest pose regularization cost term."""
    return CostTerm(
        residual_fn=functools.partial(rest_residual, rest_pose=rest_pose),
        weight=weight,
        kind="soft",
    )


def smoothness_residual(
    cfg: torch.Tensor,
    cfg_prev: torch.Tensor,
    weight: float = 1.0,
) -> torch.Tensor:
    """Penalize large configuration changes between timesteps."""
    return (cfg - cfg_prev.to(device=cfg.device, dtype=cfg.dtype)) * weight
