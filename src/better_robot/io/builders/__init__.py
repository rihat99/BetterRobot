"""``better_robot.io.builders`` — programmatic example builders.

See ``docs/design/04_PARSERS.md §6``.
"""

from __future__ import annotations

from .kinematic_tree import build_kinematic_tree_body, build_kinematic_tree_model
from .smpl_like import JOINT_NAMES, PARENTS, make_smpl_like_body, make_smpl_like_model

__all__ = [
    "JOINT_NAMES",
    "PARENTS",
    "build_kinematic_tree_body",
    "build_kinematic_tree_model",
    "make_smpl_like_body",
    "make_smpl_like_model",
]
