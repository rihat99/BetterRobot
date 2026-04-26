"""Trust-region damping strategy.

See ``docs/concepts/solver_stack.md §5``.
"""

from __future__ import annotations


class TrustRegion:
    """Trust-region variant — updates ``lambda`` based on step-acceptance ratio."""

    def __init__(self, *, radius: float = 1.0) -> None:
        self.radius = radius

    def init(self, problem) -> float:
        raise NotImplementedError("see docs/concepts/solver_stack.md §5")

    def accept(self, lam: float) -> float:
        raise NotImplementedError("see docs/concepts/solver_stack.md §5")

    def reject(self, lam: float) -> float:
        raise NotImplementedError("see docs/concepts/solver_stack.md §5")
