"""Object-referenced least-squares problems, optimizers, and linear solvers."""

from __future__ import annotations

from ..residuals.base import DiagonalWeight, Difference, Residual, ScaleWeight, Weight, residual
from ..residuals.structure import TemporalPattern
from .implicit import ImplicitDiffConfig, ImplicitDifferentiationError
from .kernels import Cauchy, GemanMcClure, Huber, L2, RobustKernel, Tukey
from .lm import GaussNewton, LevenbergMarquardt, LinearizationDecision, LinearizationMode
from .manifolds import Bounds
from .optimizers import Optimizer, OptimizerInfo, OptimizerStatus, TorchOptimizer
from .problem import JacobianStrategy, Problem
from .solvers import (
    BandedCholesky,
    Cholesky,
    InformativeLinearSolver,
    LinearSolveResult,
    LinearSolveStatus,
    LinearSolver,
    LinearSystem,
    LU,
)
from .temporal import BlockBandedMatrix, LinearizationReason, StructuredNormal, TemporalAnalysis
from .variables import RobotVariable, SE3Variable, SO3Variable, Variable

__all__ = [
    "Bounds",
    "Residual",
    "Weight",
    "ScaleWeight",
    "DiagonalWeight",
    "Difference",
    "residual",
    "RobustKernel",
    "L2",
    "Huber",
    "Cauchy",
    "Tukey",
    "GemanMcClure",
    "LinearSystem",
    "LinearSolver",
    "InformativeLinearSolver",
    "LinearSolveResult",
    "LinearSolveStatus",
    "Cholesky",
    "LU",
    "BandedCholesky",
    "BlockBandedMatrix",
    "Variable",
    "SO3Variable",
    "SE3Variable",
    "RobotVariable",
    "TemporalPattern",
    "TemporalAnalysis",
    "StructuredNormal",
    "LinearizationMode",
    "LinearizationReason",
    "LinearizationDecision",
    "Problem",
    "JacobianStrategy",
    "Optimizer",
    "OptimizerInfo",
    "OptimizerStatus",
    "TorchOptimizer",
    "LevenbergMarquardt",
    "GaussNewton",
    "ImplicitDiffConfig",
    "ImplicitDifferentiationError",
]
