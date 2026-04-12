"""``better_robot.dynamics.action`` — Crocoddyl-style 3-layer action models.

Skeleton only. See ``docs/06_DYNAMICS.md §6``.
"""

from __future__ import annotations

from .action import ActionModel
from .differential import DifferentialActionModel, DifferentialActionModelFreeFwd
from .integrated import IntegratedActionModelEuler, IntegratedActionModelRK4

__all__ = [
    "ActionModel",
    "DifferentialActionModel",
    "DifferentialActionModelFreeFwd",
    "IntegratedActionModelEuler",
    "IntegratedActionModelRK4",
]
