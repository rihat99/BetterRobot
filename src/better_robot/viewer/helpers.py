"""Shared viser conversion utilities for robot examples."""
from __future__ import annotations

import torch

from ..models.robot_model import RobotModel


def wxyz_pos_to_se3(wxyz: tuple, pos: tuple | list) -> torch.Tensor:
    """Convert viser wxyz + position to SE3 [tx, ty, tz, qx, qy, qz, qw].

    Args:
        wxyz: Viser scalar-first quaternion (w, x, y, z).
        pos: Position (x, y, z).

    Returns:
        Shape (7,) SE3 tensor [tx, ty, tz, qx, qy, qz, qw].
    """
    w, x, y, z = wxyz
    return torch.tensor([pos[0], pos[1], pos[2], x, y, z, w], dtype=torch.float32)


def qxyzw_to_wxyz(q: torch.Tensor) -> tuple:
    """Convert [qx, qy, qz, qw] tensor to viser (w, x, y, z) tuple.

    Args:
        q: Shape (4,) quaternion tensor [qx, qy, qz, qw].

    Returns:
        Viser scalar-first quaternion (w, x, y, z).
    """
    return (q[3].item(), q[0].item(), q[1].item(), q[2].item())


def build_cfg_dict(model: RobotModel, cfg: torch.Tensor) -> dict[str, float]:
    """Build joint_name→angle dict for ViserUrdf.update_cfg.

    Filters to actuated joints (revolute, continuous, prismatic) in
    the same BFS order used by forward_kinematics.

    Args:
        model: RobotModel instance.
        cfg: Shape (num_actuated_joints,) joint configuration tensor.

    Returns:
        Dict mapping joint name to float angle value.
    """
    names = [
        name
        for name, jtype in zip(model.joints.names, model._fk_joint_types)
        if jtype in ("revolute", "continuous", "prismatic")
    ]
    return {name: float(v) for name, v in zip(names, cfg.detach().cpu())}
