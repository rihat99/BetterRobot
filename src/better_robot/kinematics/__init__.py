"""``better_robot.kinematics`` — forward kinematics, frame updates, Jacobians.

See ``docs/concepts/kinematics.md``.
"""

from __future__ import annotations

from enum import Enum

from .forward import (
    forward_kinematics,
    forward_kinematics_raw,
    update_frame_placements,
)
from .jacobian import (
    compute_joint_jacobians,
    get_frame_jacobian,
    get_joint_jacobian,
    residual_jacobian,
)
from .jacobian_strategy import JacobianStrategy


class ReferenceFrame(str, Enum):
    """Pinocchio-style reference frame for spatial Jacobians.

    The string values match the ``reference=`` keyword accepted by
    :func:`get_frame_jacobian` and :func:`get_joint_jacobian`. Plain strings
    are still accepted (``str`` subclass), so existing call sites keep
    working.
    """

    WORLD = "world"
    LOCAL = "local"
    LOCAL_WORLD_ALIGNED = "local_world_aligned"


__all__ = [
    "forward_kinematics",
    "forward_kinematics_raw",
    "update_frame_placements",
    "compute_joint_jacobians",
    "get_joint_jacobian",
    "get_frame_jacobian",
    "residual_jacobian",
    "JacobianStrategy",
    "ReferenceFrame",
]
