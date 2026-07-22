"""Object-referenced residuals, weights, and shared evaluation nodes."""

from __future__ import annotations

from .base import DiagonalWeight, Difference, Residual, ScalarCost, ScaleWeight, Weight, residual
from .nodes import Node, RobotState
from .chamfer import MaskedChamferResidual
from .limits import JointPositionLimit, JointVelocityLimit
from .human import SwingTwistLimitResidual
from .pose import OrientationResidual, PoseResidual, PositionResidual
from .projection import ProjectionResidual
from .contact import ContactConsistencyResidual
from .regularization import JointRotationPrior, ReferenceTrajectoryResidual, RestResidual
from .temporal import TimeIndexedResidual
from .smoothness import AccelerationResidual, VelocityResidual
from .structure import TemporalPattern
from .scene_sdf import (
    SceneAttractionResidual,
    SceneClearanceResidual,
    ScenePenetrationResidual,
    SceneSDFResult,
    SceneSDFState,
)

__all__ = [
    "Residual",
    "Weight",
    "ScaleWeight",
    "DiagonalWeight",
    "Difference",
    "ScalarCost",
    "residual",
    "Node",
    "RobotState",
    "PoseResidual",
    "PositionResidual",
    "OrientationResidual",
    "ProjectionResidual",
    "MaskedChamferResidual",
    "SceneSDFState",
    "SceneSDFResult",
    "ScenePenetrationResidual",
    "SceneAttractionResidual",
    "SceneClearanceResidual",
    "JointPositionLimit",
    "JointVelocityLimit",
    "SwingTwistLimitResidual",
    "RestResidual",
    "JointRotationPrior",
    "ReferenceTrajectoryResidual",
    "ContactConsistencyResidual",
    "TimeIndexedResidual",
    "VelocityResidual",
    "AccelerationResidual",
    "TemporalPattern",
]
