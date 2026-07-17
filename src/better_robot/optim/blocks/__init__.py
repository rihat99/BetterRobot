"""Implementation package for named-block optimization evaluation.

Public consumers import the frozen façade from :mod:`better_robot.optim`;
deep modules in this package are not a compatibility surface.
"""

from __future__ import annotations

from ...residuals.structure import TemporalPattern as TemporalPattern
from ..structure import BlockBandedMatrix as BlockBandedMatrix
from ..structure import LinearizationDecision as LinearizationDecision
from ..structure import LinearizationMode as LinearizationMode
from ..structure import LinearizationReason as LinearizationReason
from ..structure import LinearSystemKind as LinearSystemKind
from ..structure import NormalOperator as NormalOperator
from ..structure import TemporalAnalysis as TemporalAnalysis
from .implicit import ImplicitDiffConfig as ImplicitDiffConfig
from .implicit import ImplicitDifferentiationError as ImplicitDifferentiationError
from .manifolds import (
    Bounds as Bounds,
    Euclidean as Euclidean,
    RobotConfig as RobotConfig,
    SE3Manifold as SE3Manifold,
    SO3Manifold as SO3Manifold,
)
from .problem import ObjectiveItem as ObjectiveItem
from .problem import Problem as Problem
from .problem import ResidualItem as ResidualItem
from .providers import RobotStateProvider as RobotStateProvider
from .phase import Phase as Phase
from .phase import PhaseResult as PhaseResult
from .phase import run_phases as run_phases
from .solver_adam import Adam as Adam
from .solver_adam import AdamState as AdamState
from .solver_adam import AdamStatus as AdamStatus
from .solver_lm import GaussNewton as GaussNewton
from .solver_lm import LevenbergMarquardt as LevenbergMarquardt
from .solver_lm import LMState as LMState
from .solver_lm import LMStatus as LMStatus
from .temporal import StructuredNormal as StructuredNormal
from .variables import Values as Values
from .variables import VarSpec as VarSpec
from .variables import detach_values as detach_values
