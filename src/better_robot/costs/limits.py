"""Joint limit, velocity, acceleration, and jerk residuals."""
from __future__ import annotations

import functools

import torch

from ..models.robot_model import RobotModel
from .cost_term import CostTerm

__all__ = [
    "limit_residual",
    "limit_cost",
    "velocity_residual",
    "acceleration_residual",
    "jerk_residual",
]


def limit_residual(
    q: torch.Tensor,
    model: RobotModel,
    weight: float = 1.0,
) -> torch.Tensor:
    """Compute joint limit violation residual.

    Uses clamp(min=0) so residual is zero within limits.
    CRITICAL: do not remove the clamp — without it, residuals are nonzero
    everywhere and act as a centering force that overwhelms pose costs.

    Args:
        q: Shape (num_actuated_joints,). Current configuration.
        model: RobotModel instance.
        weight: Scalar weight.

    Returns:
        Shape (2 * num_actuated_joints,). Upper and lower violations.
    """
    lo = model.joints.lower_limits.to(device=q.device, dtype=q.dtype)
    hi = model.joints.upper_limits.to(device=q.device, dtype=q.dtype)
    lower_viol = torch.clamp(lo - q, min=0.0) * weight
    upper_viol = torch.clamp(q - hi, min=0.0) * weight
    return torch.cat([lower_viol, upper_viol], dim=-1)


def limit_cost(model: RobotModel, weight: float = 0.1) -> CostTerm:
    """Create a joint limit cost term."""
    return CostTerm(
        residual_fn=functools.partial(limit_residual, model=model),
        weight=weight,
        kind="soft",
    )


def velocity_residual(
    q: torch.Tensor,
    q_prev: torch.Tensor,
    model: RobotModel,
    dt: float,
    weight: float = 1.0,
) -> torch.Tensor:
    """Joint velocity limit violation residual."""
    raise NotImplementedError


def acceleration_residual(
    q: torch.Tensor,
    q_tp2: torch.Tensor,
    q_tp1: torch.Tensor,
    q_tm1: torch.Tensor,
    q_tm2: torch.Tensor,
    dt: float,
    weight: float = 1.0,
) -> torch.Tensor:
    """Joint acceleration using 5-point stencil."""
    raise NotImplementedError


def jerk_residual(
    q_tp3: torch.Tensor,
    q_tp2: torch.Tensor,
    q_tp1: torch.Tensor,
    q_tm1: torch.Tensor,
    q_tm2: torch.Tensor,
    q_tm3: torch.Tensor,
    dt: float,
    weight: float = 1.0,
) -> torch.Tensor:
    """Joint jerk using 7-point stencil."""
    raise NotImplementedError
