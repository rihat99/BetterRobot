"""``better_robot.optim.optimizers`` — Gauss-Newton / LM / Adam / LBFGS.

See ``docs/design/07_RESIDUALS_COSTS_SOLVERS.md §5``.
"""

from __future__ import annotations

from .adam import Adam
from .base import OptimizationResult, Optimizer
from .gauss_newton import GaussNewton
from .lbfgs import LBFGS
from .levenberg_marquardt import LevenbergMarquardt
from .lm_then_lbfgs import LMThenLBFGS
from .multi_stage import MultiStageOptimizer, OptimizerStage

__all__ = [
    "Optimizer",
    "OptimizationResult",
    "LevenbergMarquardt",
    "GaussNewton",
    "Adam",
    "LBFGS",
    "LMThenLBFGS",
    "MultiStageOptimizer",
    "OptimizerStage",
]
