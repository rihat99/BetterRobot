"""``better_robot.residuals`` — residual classes + registry.

Every residual is a callable object with an optional analytic
``.jacobian()``. Compose them into a ``CostStack``, and the stack is what
the solver sees.

See ``docs/07_RESIDUALS_COSTS_SOLVERS.md``.
"""

from __future__ import annotations

from .base import Residual, ResidualState
from .collision import SelfCollisionResidual, WorldCollisionResidual
from .limits import JointAccelLimit, JointPositionLimit, JointVelocityLimit
from .manipulability import YoshikawaResidual
from .pose import OrientationResidual, PoseResidual, PositionResidual
from .regularization import NullspaceResidual, RestResidual
from .registry import get_residual, register_residual, registered_residuals
from .smoothness import Accel5pt, Jerk5pt, Velocity5pt

__all__ = [
    "Residual",
    "ResidualState",
    "register_residual",
    "get_residual",
    "registered_residuals",
    "PoseResidual",
    "PositionResidual",
    "OrientationResidual",
    "JointPositionLimit",
    "JointVelocityLimit",
    "JointAccelLimit",
    "RestResidual",
    "NullspaceResidual",
    "Velocity5pt",
    "Accel5pt",
    "Jerk5pt",
    "YoshikawaResidual",
    "SelfCollisionResidual",
    "WorldCollisionResidual",
]
