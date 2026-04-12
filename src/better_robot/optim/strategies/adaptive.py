"""Adaptive damping strategy — the legacy BetterRobot LM behaviour.

See ``docs/07_RESIDUALS_COSTS_SOLVERS.md §5``.
"""

from __future__ import annotations


class Adaptive:
    """Adaptive ``lambda`` — shrink on accept, grow on reject."""

    def __init__(self, *, lam0: float = 1e-4, factor: float = 2.0) -> None:
        self.lam0 = lam0
        self.factor = factor

    def init(self, problem) -> float:
        return self.lam0

    def accept(self, lam: float) -> float:
        """Shrink lambda on a successful step."""
        return max(lam / self.factor, 1e-10)

    def reject(self, lam: float) -> float:
        """Grow lambda on a failed step."""
        return min(lam * self.factor, 1e8)
