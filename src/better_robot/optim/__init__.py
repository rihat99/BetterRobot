"""Named-block problems, manifolds, and optimization algorithms."""

from __future__ import annotations

from ..residuals.structure import TemporalPattern
from .first_order import FirstOrderResult, run_first_order
from .implicit import ImplicitDiffConfig, ImplicitDifferentiationError
from .kernels import Cauchy, GemanMcClure, Huber, L2, RobustKernel, Tukey
from .lm import GaussNewton, LevenbergMarquardt, LinearizationDecision, LinearizationMode, LMState, LMStatus
from .manifolds import Bounds, Euclidean, RobotConfig, SE3Manifold, SO3Manifold
from .problem import JacobianStrategy, Problem, ResidualItem
from .providers import RobotStateProvider
from .solvers import (
    BandedCholesky,
    Cholesky,
    InformativeLinearSolver,
    LinearSolveResult,
    LinearSolveStatus,
    LinearSolver,
    LinearSystem,
)
from .temporal import BlockBandedMatrix, LinearizationReason, StructuredNormal, TemporalAnalysis
from .variables import Values, VarSpec, detach_values

__all__ = [
    # Named-block evaluation and solver API. Deliberately qualified under ``optim``;
    # the package root keeps its existing Lie ``SE3`` identity.
    "Bounds",
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
    "BandedCholesky",
    "BlockBandedMatrix",
    "Euclidean",
    "SO3Manifold",
    "SE3Manifold",
    "RobotConfig",
    "Values",
    "VarSpec",
    "TemporalPattern",
    "TemporalAnalysis",
    "StructuredNormal",
    "LinearizationMode",
    "LinearizationReason",
    "LinearizationDecision",
    "Problem",
    "JacobianStrategy",
    "ResidualItem",
    "RobotStateProvider",
    "detach_values",
    "FirstOrderResult",
    "run_first_order",
    "LevenbergMarquardt",
    "GaussNewton",
    "ImplicitDiffConfig",
    "ImplicitDifferentiationError",
    "LMState",
    "LMStatus",
]
