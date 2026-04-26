"""``better_robot.optim.solvers`` — linear solvers.

See ``docs/concepts/solver_stack.md §5``.
"""

from __future__ import annotations

from .cg import CG
from .cholesky import Cholesky
from .lstsq import LSTSQ
from .sparse_cholesky import SparseCholesky

__all__ = ["Cholesky", "LSTSQ", "CG", "SparseCholesky"]
