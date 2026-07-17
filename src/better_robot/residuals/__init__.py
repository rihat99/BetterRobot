"""``better_robot.residuals`` — residual classes.

Every residual is a callable object with an optional analytic
``.jacobian()``. Compose them into a ``CostStack``, and the stack is what
the solver sees.

See ``docs/concepts/residuals_and_costs.md``.
"""

from __future__ import annotations

from .base import Residual, ResidualState
from .collision import SelfCollisionResidual, WorldCollisionResidual
from .limits import JointAccelLimit, JointPositionLimit, JointVelocityLimit
from .manipulability import YoshikawaResidual
from .human import SwingTwistLimitResidual
from .pose import OrientationResidual, PoseResidual, PositionResidual
from .contact import ContactConsistencyResidual
from .regularization import JointRotationPrior, NullspaceResidual, ReferenceTrajectoryResidual, RestResidual
from .temporal import TimeIndexedResidual
from .smoothness import AccelerationResidual, JerkResidual, VelocityResidual

__all__ = [
    "Residual",
    "ResidualState",
    "PoseResidual",
    "PositionResidual",
    "OrientationResidual",
    "JointPositionLimit",
    "JointVelocityLimit",
    "JointAccelLimit",
    "SwingTwistLimitResidual",
    "RestResidual",
    "JointRotationPrior",
    "NullspaceResidual",
    "ReferenceTrajectoryResidual",
    "ContactConsistencyResidual",
    "TimeIndexedResidual",
    "VelocityResidual",
    "AccelerationResidual",
    "JerkResidual",
    "YoshikawaResidual",
    "SelfCollisionResidual",
    "WorldCollisionResidual",
]
