"""``better_robot.costs`` — ``CostStack`` composition over residuals.

See ``docs/concepts/residuals_and_costs.md §3``.
"""

from __future__ import annotations

from .stack import CostItem, CostStack

__all__ = ["CostStack", "CostItem"]
