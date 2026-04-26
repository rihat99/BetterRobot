"""``better_robot.optim.solvers`` — linear solvers.

See ``docs/design/07_RESIDUALS_COSTS_SOLVERS.md §5``.
"""

from __future__ import annotations

from .cg import CG
from .cholesky import Cholesky
from .lstsq import LSTSQ
from .sparse_cholesky import SparseCholesky

__all__ = ["Cholesky", "LSTSQ", "CG", "SparseCholesky"]
