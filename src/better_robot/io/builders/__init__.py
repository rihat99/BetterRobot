"""``better_robot.io.builders`` — programmatic model builders.

See ``docs/concepts/parsers_and_ir.md``.
"""

from __future__ import annotations

from .kinematic_tree import build_kinematic_tree_body, build_kinematic_tree_model

__all__ = [
    "build_kinematic_tree_body",
    "build_kinematic_tree_model",
]
