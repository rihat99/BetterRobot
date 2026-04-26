"""``better_robot.dynamics.action`` — Crocoddyl-style 3-layer action models.

Forward dynamics + autograd-derived Jacobians wired into a state-manifold
state. The DDP / iLQR solver itself is left for a follow-on release.
See ``docs/design/06_DYNAMICS.md §6``.
"""

from __future__ import annotations

from .action import ActionData, ActionModel
from .differential import DifferentialActionModel, DifferentialActionModelFreeFwd
from .integrated import IntegratedActionModelEuler, IntegratedActionModelRK4

__all__ = [
    "ActionData",
    "ActionModel",
    "DifferentialActionModel",
    "DifferentialActionModelFreeFwd",
    "IntegratedActionModelEuler",
    "IntegratedActionModelRK4",
]
