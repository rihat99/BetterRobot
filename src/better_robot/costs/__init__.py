"""Costs layer: differentiable residual functions.

Each function takes a joint config tensor and returns a residual vector.
No solver dependency — pure PyTorch functions.
"""

from ._pose import pose_residual as pose_residual
from ._limits import (
    limit_residual as limit_residual,
    velocity_residual as velocity_residual,
    acceleration_residual as acceleration_residual,
    jerk_residual as jerk_residual,
)
from ._regularization import (
    rest_residual as rest_residual,
    smoothness_residual as smoothness_residual,
)
from ._collision import (
    self_collision_residual as self_collision_residual,
    world_collision_residual as world_collision_residual,
)
from ._manipulability import manipulability_residual as manipulability_residual
from ._jacobian import (
    pose_jacobian as pose_jacobian,
    limit_jacobian as limit_jacobian,
    rest_jacobian as rest_jacobian,
)

__all__ = [
    "pose_residual",
    "limit_residual",
    "velocity_residual",
    "acceleration_residual",
    "jerk_residual",
    "rest_residual",
    "smoothness_residual",
    "self_collision_residual",
    "world_collision_residual",
    "manipulability_residual",
    "pose_jacobian",
    "limit_jacobian",
    "rest_jacobian",
]
