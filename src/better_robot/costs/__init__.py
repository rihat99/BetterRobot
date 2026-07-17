"""Compatibility imports for :mod:`better_robot.optim` cost composition.

``CostStack`` and ``CostItem`` are canonically defined under
:mod:`better_robot.optim`. This package remains only while BetterRobot's
legacy flat-problem paths migrate to named blocks.
"""

from __future__ import annotations

from .stack import CostItem, CostStack

__all__ = ["CostStack", "CostItem"]
