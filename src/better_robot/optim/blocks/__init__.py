"""Implementation package for named-block optimization evaluation.

Public consumers import the frozen façade from :mod:`better_robot.optim`;
deep modules in this package are not a compatibility surface.
"""

from __future__ import annotations

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
from .solver_lm import GaussNewton as GaussNewton
from .solver_lm import LevenbergMarquardt as LevenbergMarquardt
from .solver_lm import LMState as LMState
from .solver_lm import LMStatus as LMStatus
from .variables import Values as Values
from .variables import VarSpec as VarSpec
from .variables import detach_values as detach_values
