"""``better_robot.residuals`` — residual classes.

Every residual is a callable object with an optional analytic
``.jacobian()``. Compose them into a ``CostStack``, and the stack is what
the solver sees.

See ``docs/concepts/residuals_and_costs.md``.
"""

from __future__ import annotations

from .base import Residual, ResidualState
from .chamfer import MaskedChamferResidual
from .collision import SelfCollisionResidual, WorldCollisionResidual
from .limits import JointAccelLimit, JointPositionLimit, JointVelocityLimit
from .manipulability import YoshikawaResidual
from .human import SwingTwistLimitResidual
from .pose import OrientationResidual, PoseResidual, PositionResidual
from .projection import ProjectionResidual
from .contact import ContactConsistencyResidual
from .regularization import JointRotationPrior, NullspaceResidual, ReferenceTrajectoryResidual, RestResidual
from .temporal import TimeIndexedResidual
from .smoothness import AccelerationResidual, JerkResidual, VelocityResidual
from .scene_sdf import (
    SceneAttractionResidual,
    SceneClearanceResidual,
    ScenePenetrationResidual,
    SceneSDFProvider,
    SceneSDFResult,
)

__all__ = [
    "Residual",
    "ResidualState",
    "PoseResidual",
    "PositionResidual",
    "OrientationResidual",
    "ProjectionResidual",
    "MaskedChamferResidual",
    "SceneSDFProvider",
    "SceneSDFResult",
    "ScenePenetrationResidual",
    "SceneAttractionResidual",
    "SceneClearanceResidual",
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
