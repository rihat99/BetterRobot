"""Rest pose and smoothness regularization residuals."""
from __future__ import annotations

import functools

import torch

from .cost_term import CostTerm

__all__ = [
    "rest_residual",
    "rest_cost",
    "smoothness_residual",
]


def rest_residual(
    q: torch.Tensor,
    q_rest: torch.Tensor,
    weight: float = 1.0,
) -> torch.Tensor:
    """Penalize deviation from a rest/default pose.

    Args:
        q: Shape (num_actuated_joints,).
        q_rest: Shape (num_actuated_joints,). Target rest configuration.
        weight: Scalar weight.

    Returns:
        Shape (num_actuated_joints,). Weighted deviation from rest.
    """
    return (q - q_rest.to(device=q.device, dtype=q.dtype)) * weight


def rest_cost(q_rest: torch.Tensor, weight: float = 0.01) -> CostTerm:
    """Create a rest pose regularization cost term."""
    return CostTerm(
        residual_fn=functools.partial(rest_residual, q_rest=q_rest),
        weight=weight,
        kind="soft",
    )


def smoothness_residual(
    q: torch.Tensor,
    q_prev: torch.Tensor,
    weight: float = 1.0,
) -> torch.Tensor:
    """Penalize large configuration changes between timesteps."""
    return (q - q_prev.to(device=q.device, dtype=q.dtype)) * weight
