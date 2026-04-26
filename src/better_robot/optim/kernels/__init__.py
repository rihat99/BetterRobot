"""``better_robot.optim.kernels`` — robust kernels.

See ``docs/concepts/solver_stack.md §5``.
"""

from __future__ import annotations

from .cauchy import Cauchy
from .huber import Huber
from .l2 import L2
from .tukey import Tukey

__all__ = ["L2", "Huber", "Cauchy", "Tukey"]
