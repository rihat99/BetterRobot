"""``solve_trajopt`` — kinematic trajectory optimisation facade.

Skeleton only in v1; implementation arrives once ``Trajectory`` threads
cleanly through ``ResidualState`` + ``CostStack``. See
``docs/08_TASKS.md §3``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..data_model.model import Model
from .ik import OptimizerConfig
from .trajectory import Trajectory


@dataclass
class TrajOptCostConfig:
    """Weights for the built-in trajopt cost stack."""

    pose_weight: float = 1.0
    limit_weight: float = 0.1
    smoothness_weight: float = 0.01
    collision_margin: float = 0.02
    collision_weight: float = 1.0


@dataclass
class TrajOptResult:
    """Return type of ``solve_trajopt``."""

    trajectory: Trajectory
    residual: torch.Tensor
    iters: int
    converged: bool
    model: Model


def solve_trajopt(
    model: Model,
    *,
    horizon: int,
    dt: float,
    keyframes: dict[int, dict[str, torch.Tensor]] | None = None,
    initial_traj: Trajectory | None = None,
    cost_cfg: TrajOptCostConfig | None = None,
    optimizer_cfg: OptimizerConfig | None = None,
    robot_collision=None,
) -> TrajOptResult:
    """Kinematic trajectory optimisation.

    Variable: ``Trajectory.q`` of shape ``(B, T, nq)``.

    See docs/08_TASKS.md §3.
    """
    raise NotImplementedError("see docs/08_TASKS.md §3")
