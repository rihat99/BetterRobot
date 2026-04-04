"""Pose residuals for end-effector target matching."""

from __future__ import annotations

import torch

from ..core._robot import Robot


def pose_residual(
    cfg: torch.Tensor,
    robot: Robot,
    target_link_index: int,
    target_pose: torch.Tensor,
    pos_weight: float = 1.0,
    ori_weight: float = 1.0,
) -> torch.Tensor:
    """Compute SE3 log-space error between actual and target link pose.

    Args:
        cfg: Shape (num_actuated_joints,). Current joint configuration.
        robot: Robot instance.
        target_link_index: Index into robot.links.names for the target link.
        target_pose: Shape (7,). Target SE3 pose as wxyz+xyz.
        pos_weight: Weight on position error (first 3 dims of log).
        ori_weight: Weight on orientation error (last 3 dims of log).

    Returns:
        Shape (6,). Weighted SE3 log error [pos*w, ori*w].
    """
    raise NotImplementedError
