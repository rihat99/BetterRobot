"""``better_robot.tasks`` — task facades and trajectory utilities.

Optimisation entry points are short facades that construct variables and
residual objects, assemble a ``Problem``, run an optimizer, and return a clean
task result. Trajectory helpers remain plain, differentiable tensor operations.

No Jacobian code, no solver loops, no fixed-vs-floating base branching
lives here — that all belongs one layer down.

See ``docs/concepts/residuals_costs_and_solvers.md``.
"""

from __future__ import annotations

from .contact_forces import ContactForceResult, ContactForceWeights, solve_contact_forces
from .ik import IKCostConfig, IKResult, OptimizerConfig, solve_ik
from .smoothing import smooth_trajectory
from .trajectory import Trajectory
from .trajopt import TrajOptResult, solve_trajopt

__all__ = [
    "solve_ik",
    "solve_trajopt",
    "solve_contact_forces",
    "smooth_trajectory",
    "Trajectory",
    "IKCostConfig",
    "IKResult",
    "OptimizerConfig",
    "TrajOptResult",
    "ContactForceResult",
    "ContactForceWeights",
]
