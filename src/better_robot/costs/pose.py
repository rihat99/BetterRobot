"""Pose residuals for end-effector target matching."""
from __future__ import annotations

import functools

import torch

from ..models.robot_model import RobotModel
from ..math.se3 import se3_compose, se3_inverse, se3_log
from .cost_term import CostTerm

__all__ = [
    "pose_residual",
    "pose_cost",
]


def pose_residual(
    q: torch.Tensor,
    model: RobotModel,
    target_link_index: int,
    target_pose: torch.Tensor,
    pos_weight: float = 1.0,
    ori_weight: float = 1.0,
    base_pose: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute SE3 log-space error between actual and target link pose.

    Args:
        q: Shape (*batch, num_actuated_joints). Current joint configuration.
        model: RobotModel instance.
        target_link_index: Index into model.links.names for the target link.
        target_pose: Shape (7,). Target SE3 pose [tx, ty, tz, qx, qy, qz, qw].
        pos_weight: Weight on position error (first 3 dims of log).
        ori_weight: Weight on orientation error (last 3 dims of log).
        base_pose: Shape (*batch, 7). Optional floating base SE3 transform.

    Returns:
        Shape (*batch, 6). Weighted SE3 log error [pos*w, ori*w].
    """
    fk = model.forward_kinematics(q, base_pose=base_pose)
    actual_pose = fk[..., target_link_index, :]
    target_pose = target_pose.to(device=q.device, dtype=q.dtype)
    T_err = se3_compose(se3_inverse(target_pose), actual_pose)
    log_err = se3_log(T_err)
    pos_err = log_err[..., :3] * pos_weight
    ori_err = log_err[..., 3:] * ori_weight
    return torch.cat([pos_err, ori_err], dim=-1)


def pose_cost(
    model: RobotModel,
    target_link_index: int,
    target_pose: torch.Tensor,
    pos_weight: float = 1.0,
    ori_weight: float = 1.0,
    pose_weight: float = 1.0,
    base_pose: torch.Tensor | None = None,
) -> CostTerm:
    """Create a pose cost term for use in a Problem.

    Returns a ready-to-use CostTerm with functools.partial already applied.

    Args:
        model: RobotModel instance.
        target_link_index: Index of the target link.
        target_pose: Shape (7,). Target SE3 pose.
        pos_weight: Weight on position error.
        ori_weight: Weight on orientation error.
        pose_weight: Overall cost weight.
        base_pose: Optional floating base SE3 transform.

    Returns:
        CostTerm ready to add to a Problem.
    """
    return CostTerm(
        residual_fn=functools.partial(
            pose_residual,
            model=model,
            target_link_index=target_link_index,
            target_pose=target_pose,
            pos_weight=pos_weight,
            ori_weight=ori_weight,
            base_pose=base_pose,
        ),
        weight=pose_weight,
        kind="soft",
    )
