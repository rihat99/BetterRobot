"""``better_robot.optim.strategies`` — damping / trust-region strategies.

See ``docs/design/07_RESIDUALS_COSTS_SOLVERS.md §5``.
"""

from __future__ import annotations

from .adaptive import Adaptive
from .constant import Constant
from .trust_region import TrustRegion

__all__ = ["Constant", "Adaptive", "TrustRegion"]
