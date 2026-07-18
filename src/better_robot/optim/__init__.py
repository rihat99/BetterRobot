"""``better_robot.optim`` — named-block problems and solver components.

:class:`Problem` values use batched :class:`Adam`,
:class:`LevenbergMarquardt`, or :class:`GaussNewton` directly and may be
orchestrated with :class:`Phase`.

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
    ImplicitDiffConfig,
    ImplicitDifferentiationError,
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

__all__ = [
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
    "ImplicitDiffConfig",
    "ImplicitDifferentiationError",
    "LMState",
    "LMStatus",
]
