"""``better_robot.kinematics`` — forward kinematics, frame updates, Jacobians.

See ``docs/05_KINEMATICS.md``.
"""

from __future__ import annotations

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

__all__ = [
    "forward_kinematics",
    "forward_kinematics_raw",
    "update_frame_placements",
    "compute_joint_jacobians",
    "get_joint_jacobian",
    "get_frame_jacobian",
    "residual_jacobian",
    "JacobianStrategy",
]
