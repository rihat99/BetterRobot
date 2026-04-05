"""URDF model loading: load_urdf(path_or_urdf) -> RobotModel."""
from __future__ import annotations

import torch
import yourdfpy

from ..robot_model import RobotModel
from ._urdf_impl import RobotURDFParser


def load_urdf(
    urdf: "str | yourdfpy.URDF",
    default_joint_cfg: torch.Tensor | None = None,
) -> RobotModel:
    """Load a robot model from a URDF file or yourdfpy.URDF object.

    Args:
        urdf: Either a file path string or a loaded yourdfpy.URDF instance.
        default_joint_cfg: Optional (num_actuated_joints,) default configuration.
            Defaults to midpoint of joint limits.

    Returns:
        RobotModel instance with all FK data precomputed.
    """
    if isinstance(urdf, str):
        urdf = yourdfpy.URDF.load(urdf)
    return RobotModel.from_urdf(urdf, default_joint_cfg=default_joint_cfg)
