"""Compatibility imports for :mod:`better_robot.optim` cost composition.

``CostStack`` and ``CostItem`` are canonically defined under
:mod:`better_robot.optim`. This package keeps identity-preserving import
compatibility for direct flat-problem callers.
"""

from __future__ import annotations

from .stack import CostItem, CostStack

__all__ = ["CostStack", "CostItem"]
