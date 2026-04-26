"""``better_robot.costs`` — ``CostStack`` composition over residuals.

See ``docs/design/07_RESIDUALS_COSTS_SOLVERS.md §3``.
"""

from __future__ import annotations

from .factory import factory
from .stack import CostItem, CostStack

__all__ = ["CostStack", "CostItem", "factory"]
