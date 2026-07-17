"""``better_robot.data_model`` — ``Model`` / ``Data`` / ``Frame`` / ``Body`` / ``Joint``.

This package is the Pinocchio-style universal data model. The fixed vs
floating-base split is gone: a floating base is simply a ``JointFreeFlyer``
root joint.

See ``docs/concepts/model_and_data.md``.
"""

from __future__ import annotations

from ._kinematics_level import KinematicsLevel
from .body import Body
from .data import Data
from .frame import Frame, FrameType
from .joint import JOINT_DIMENSIONS, Joint
from .model import Model
from .model_structure import ModelStructure
from .model_values import ModelValues

__all__ = [
    "Model",
    "ModelStructure",
    "ModelValues",
    "Data",
    "Frame",
    "FrameType",
    "Body",
    "Joint",
    "JOINT_DIMENSIONS",
    "KinematicsLevel",
]
