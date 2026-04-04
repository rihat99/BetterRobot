"""Joint limit, velocity, acceleration, and jerk residuals."""

from __future__ import annotations

import torch

from ..core._robot import Robot


def limit_residual(
    cfg: torch.Tensor,
    robot: Robot,
    weight: float = 1.0,
) -> torch.Tensor:
    """Compute joint limit violation residual.

    Positive values indicate violation (for use as constraint_leq_zero).

    Args:
        cfg: Shape (num_actuated_joints,). Current joint configuration.
        robot: Robot instance.
        weight: Scalar weight.

    Returns:
        Shape (2 * num_actuated_joints,). Upper and lower violations concatenated.
    """
    lo = robot.joints.lower_limits.to(device=cfg.device, dtype=cfg.dtype)
    hi = robot.joints.upper_limits.to(device=cfg.device, dtype=cfg.dtype)

    # Lower violation: positive only when cfg < lo, zero within limits
    lower_viol = torch.clamp(lo - cfg, min=0.0) * weight
    # Upper violation: positive only when cfg > hi, zero within limits
    upper_viol = torch.clamp(cfg - hi, min=0.0) * weight

    return torch.cat([lower_viol, upper_viol], dim=-1)


def velocity_residual(
    cfg: torch.Tensor,
    cfg_prev: torch.Tensor,
    robot: Robot,
    dt: float,
    weight: float = 1.0,
) -> torch.Tensor:
    """Compute joint velocity limit violation residual.

    Args:
        cfg: Shape (num_actuated_joints,). Current config.
        cfg_prev: Shape (num_actuated_joints,). Previous config.
        robot: Robot instance.
        dt: Timestep in seconds.
        weight: Scalar weight.

    Returns:
        Shape (num_actuated_joints,). Velocity violations.
    """
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
    """Compute joint acceleration using 5-point stencil.

    Uses the 5-point stencil: (-q_{t+2} + 16*q_{t+1} - 30*q_t + 16*q_{t-1} - q_{t-2}) / (12*dt^2)

    Args:
        cfg: Shape (num_actuated_joints,). Config at time t (center point).
        cfg_tp2, cfg_tp1: Configs at t+2, t+1.
        cfg_tm1, cfg_tm2: Configs at t-1, t-2.
        dt: Timestep in seconds.
        weight: Scalar weight.

    Returns:
        Shape (num_actuated_joints,). Weighted acceleration magnitude.
    """
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
    """Compute joint jerk using 7-point stencil.

    Uses the 7-point stencil: (-q_{t+3} + 8*q_{t+2} - 13*q_{t+1} + 13*q_{t-1} - 8*q_{t-2} + q_{t-3}) / (8*dt^3)

    Args:
        cfg_tp3, cfg_tp2, cfg_tp1: Configs at t+3, t+2, t+1.
        cfg_tm1, cfg_tm2, cfg_tm3: Configs at t-1, t-2, t-3.
        dt: Timestep in seconds.
        weight: Scalar weight.

    Returns:
        Shape (num_actuated_joints,). Weighted jerk magnitude.
    """
    raise NotImplementedError
