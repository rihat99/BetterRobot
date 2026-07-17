"""``better_robot.optim.solvers`` — linear solvers.

See ``docs/concepts/solver_stack.md §5``.
"""

from __future__ import annotations

from .cholesky import Cholesky
from .lstsq import LSTSQ

__all__ = ["Cholesky", "LSTSQ"]
