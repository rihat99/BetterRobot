"""``better_robot.optim`` — optimization problems and solver components.

Named-block :class:`Problem` values use batched :class:`Adam`,
:class:`LevenbergMarquardt`, or :class:`GaussNewton` directly and may be
orchestrated with :class:`Phase`. Legacy :class:`LeastSquaresProblem` callers
instantiate an optimizer from :mod:`better_robot.optim.optimizers` and call
its ``minimize`` method directly.

See ``docs/concepts/solver_stack.md``.
"""

from __future__ import annotations

from .blocks import (
    Adam,
    AdamState,
    AdamStatus,
    BlockBandedMatrix,
    Bounds,
    Euclidean,
    GaussNewton,
    LevenbergMarquardt,
    LMState,
    LMStatus,
    LinearizationDecision,
    LinearizationMode,
    LinearizationReason,
    LinearSystemKind,
    NormalOperator,
    ObjectiveItem,
    Phase,
    PhaseResult,
    Problem,
    ResidualItem,
    RobotConfig,
    RobotStateProvider,
    SE3Manifold,
    SO3Manifold,
    StructuredNormal,
    TemporalAnalysis,
    TemporalPattern,
    Values,
    VarSpec,
    detach_values,
    run_phases,
)
from .cost_stack import CostItem, CostKind, CostStack
from .optimizers.base import OptimizationResult, Optimizer
from .problem import LeastSquaresProblem
from .state import SolverState

__all__ = [
    "LeastSquaresProblem",
    "Optimizer",
    "OptimizationResult",  # deprecated alias for SolverState
    "SolverState",
    "CostKind",
    "CostItem",
    "CostStack",
    # Named-block evaluation and solver API. Deliberately qualified under ``optim``;
    # the package root keeps its existing Lie ``SE3`` identity.
    "Bounds",
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
    "NormalOperator",
    "LinearizationMode",
    "LinearSystemKind",
    "LinearizationReason",
    "LinearizationDecision",
    "Problem",
    "ResidualItem",
    "ObjectiveItem",
    "RobotStateProvider",
    "detach_values",
    "Phase",
    "PhaseResult",
    "run_phases",
    "Adam",
    "AdamState",
    "AdamStatus",
    "LevenbergMarquardt",
    "GaussNewton",
    "LMState",
    "LMStatus",
]
