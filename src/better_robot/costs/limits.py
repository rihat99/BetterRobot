"""Joint limit, velocity, acceleration, and jerk residuals."""
from __future__ import annotations

import functools

import torch

from ..models.robot_model import RobotModel
from .cost_term import CostTerm


def limit_residual(
    cfg: torch.Tensor,
    robot: RobotModel,
    weight: float = 1.0,
) -> torch.Tensor:
    """Compute joint limit violation residual.

    Uses clamp(min=0) so residual is zero within limits.
    CRITICAL: do not remove the clamp — without it, residuals are nonzero
    everywhere and act as a centering force that overwhelms pose costs.

    Args:
        cfg: Shape (num_actuated_joints,). Current configuration.
        robot: RobotModel instance.
        weight: Scalar weight.

    Returns:
        Shape (2 * num_actuated_joints,). Upper and lower violations.
    """
    lo = robot.joints.lower_limits.to(device=cfg.device, dtype=cfg.dtype)
    hi = robot.joints.upper_limits.to(device=cfg.device, dtype=cfg.dtype)
    lower_viol = torch.clamp(lo - cfg, min=0.0) * weight
    upper_viol = torch.clamp(cfg - hi, min=0.0) * weight
    return torch.cat([lower_viol, upper_viol], dim=-1)


def limit_cost(robot: RobotModel, weight: float = 0.1) -> CostTerm:
    """Create a joint limit cost term."""
    return CostTerm(
        residual_fn=functools.partial(limit_residual, robot=robot),
        weight=weight,
        kind="soft",
    )


def velocity_residual(
    cfg: torch.Tensor,
    cfg_prev: torch.Tensor,
    robot: RobotModel,
    dt: float,
    weight: float = 1.0,
) -> torch.Tensor:
    """Joint velocity limit violation residual."""
    raise NotImplementedError


def acceleration_residual(
    cfg: torch.Tensor,
    cfg_tp2: torch.Tensor,
    cfg_tp1: torch.Tensor,
    cfg_tm1: torch.Tensor,
    cfg_tm2: torch.Tensor,
    dt: float,
    weight: float = 1.0,
) -> torch.Tensor:
    """Joint acceleration using 5-point stencil."""
    raise NotImplementedError


def jerk_residual(
    cfg_tp3: torch.Tensor,
    cfg_tp2: torch.Tensor,
    cfg_tp1: torch.Tensor,
    cfg_tm1: torch.Tensor,
    cfg_tm2: torch.Tensor,
    cfg_tm3: torch.Tensor,
    dt: float,
    weight: float = 1.0,
) -> torch.Tensor:
    """Joint jerk using 7-point stencil."""
    raise NotImplementedError
