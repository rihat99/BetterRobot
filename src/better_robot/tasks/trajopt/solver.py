"""Trajectory optimization solver (stub)."""
from __future__ import annotations

import torch

from ...models.robot_model import RobotModel
from .config import TrajOptConfig


def solve_trajopt(
    model: RobotModel,
    targets: dict[str, torch.Tensor],
    cfg: TrajOptConfig | None = None,
    initial_traj: torch.Tensor | None = None,
    max_iter: int = 100,
) -> torch.Tensor:
    """Solve trajectory optimization for one or more end-effector targets.

    Args:
        model: RobotModel instance.
        targets: {link_name: (7,) SE3 target}.
        cfg: Trajectory optimization configuration.
        initial_traj: (horizon, num_joints) initial trajectory.
        max_iter: Solver iterations.

    Returns:
        (horizon, num_joints) optimized joint trajectory.

    Raises:
        NotImplementedError: This solver is not yet implemented.
    """
    raise NotImplementedError(
        "solve_trajopt is not yet implemented. "
        "See docs/02_target_architecture.md for the planned design."
    )
