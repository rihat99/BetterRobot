"""``better_robot.optim.optimizers`` — Gauss-Newton / LM / Adam / LBFGS.

See ``docs/07_RESIDUALS_COSTS_SOLVERS.md §5``.
"""

from __future__ import annotations

from .adam import Adam
from .base import OptimizationResult, Optimizer
from .composite import LMThenLBFGS
from .gauss_newton import GaussNewton
from .lbfgs import LBFGS
from .levenberg_marquardt import LevenbergMarquardt

__all__ = [
    "Optimizer",
    "OptimizationResult",
    "LevenbergMarquardt",
    "GaussNewton",
    "Adam",
    "LBFGS",
    "LMThenLBFGS",
]
