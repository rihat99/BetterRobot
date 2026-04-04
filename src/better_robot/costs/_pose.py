"""Pose residuals for end-effector target matching."""

from __future__ import annotations

import torch

from ..core._robot import Robot
from ..core._lie_ops import se3_compose, se3_inverse, se3_log


def pose_residual(
    cfg: torch.Tensor,
    robot: Robot,
    target_link_index: int,
    target_pose: torch.Tensor,
    pos_weight: float = 1.0,
    ori_weight: float = 1.0,
    base_pose: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute SE3 log-space error between actual and target link pose.

    Args:
        cfg: Shape (*batch, num_actuated_joints). Current joint configuration.
        robot: Robot instance.
        target_link_index: Index into robot.links.names for the target link.
        target_pose: Shape (7,). Target SE3 pose as [tx, ty, tz, qx, qy, qz, qw].
        pos_weight: Weight on position error (first 3 dims of log).
        ori_weight: Weight on orientation error (last 3 dims of log).
        base_pose: Shape (*batch, 7). Optional floating base SE3 transform.
            Passed through to robot.forward_kinematics. Default None (fixed base).

    Returns:
        Shape (*batch, 6). Weighted SE3 log error [pos*w, ori*w].
    """
    # Run FK: (*batch, num_links, 7)
    fk = robot.forward_kinematics(cfg, base_pose=base_pose)
    # Extract target link pose: (*batch, 7)
    actual_pose = fk[..., target_link_index, :]

    # Error transform: T_err = target_pose.inv() @ actual_pose
    T_err = se3_compose(se3_inverse(target_pose), actual_pose)  # (*batch, 7)

    # se3_log returns [tx, ty, tz, rx, ry, rz]
    log_err = se3_log(T_err)          # (*batch, 6)
    pos_err = log_err[..., :3] * pos_weight   # translation part
    ori_err = log_err[..., 3:] * ori_weight   # rotation part
    return torch.cat([pos_err, ori_err], dim=-1)   # (*batch, 6)
