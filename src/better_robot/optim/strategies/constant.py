"""Constant damping strategy — pins ``lambda`` to a fixed value.

See ``docs/design/07_RESIDUALS_COSTS_SOLVERS.md §5``.
"""

from __future__ import annotations


class Constant:
    """Constant ``lambda`` throughout the optimisation."""

    def __init__(self, *, lam: float = 1e-4) -> None:
        self.lam = lam

    def init(self, problem) -> float:
        return self.lam

    def accept(self, lam: float) -> float:
        return lam

    def reject(self, lam: float) -> float:
        return lam
