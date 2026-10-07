"""``better_robot.kinematics`` — forward kinematics and Jacobians on a ``KinematicState``.

See ``docs/concepts/kinematics_and_jacobians.md``.
"""

from __future__ import annotations

from .state import (
    KinematicState,
    forward_kinematics_state,
    frame_jacobian,
    frame_jacobian_time_variation,
    joint_jacobian,
    joint_jacobian_time_variation,
)

# Public-internal (C-DEP-1): the joint-Jacobian column plan shared with ``dynamics``.
# Importable as ``better_robot.kinematics.jacobian_columns``; not part of ``__all__``.
from . import _jacobian_columns

jacobian_columns = _jacobian_columns

__all__ = [
    "KinematicState",
    "forward_kinematics_state",
    "frame_jacobian",
    "frame_jacobian_time_variation",
    "joint_jacobian",
    "joint_jacobian_time_variation",
]
