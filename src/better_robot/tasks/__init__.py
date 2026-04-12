"""``better_robot.tasks`` — thin facades over ``optim``.

Every entry point here is a short facade that picks frames by name,
builds a cost stack out of residuals, wraps it in a
``LeastSquaresProblem``, calls an optimiser, and returns a clean result.

No Jacobian code, no solver loops, no fixed-vs-floating base branching
lives here — that all belongs one layer down.

See ``docs/08_TASKS.md``.
"""

from __future__ import annotations

from .ik import IKCostConfig, IKResult, OptimizerConfig, solve_ik
from .retarget import RetargetCostConfig, retarget
from .trajectory import Trajectory
from .trajopt import TrajOptCostConfig, TrajOptResult, solve_trajopt

__all__ = [
    "solve_ik",
    "solve_trajopt",
    "retarget",
    "Trajectory",
    "IKCostConfig",
    "IKResult",
    "OptimizerConfig",
    "TrajOptCostConfig",
    "TrajOptResult",
    "RetargetCostConfig",
]
