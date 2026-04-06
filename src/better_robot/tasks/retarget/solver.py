"""Motion retargeting solver (stub)."""
from __future__ import annotations

import torch

from ...models.robot_model import RobotModel
from .config import RetargetConfig


def retarget(
    model: RobotModel,
    source_motion: torch.Tensor,
    config: RetargetConfig | None = None,
    max_iter: int = 100,
) -> torch.Tensor:
    """Retarget a motion sequence to a robot.

    Args:
        model: RobotModel instance.
        source_motion: (T, num_joints) source motion sequence.
        cfg: Retargeting configuration.
        max_iter: Solver iterations per frame.

    Returns:
        (T, num_joints) retargeted joint trajectory.

    Raises:
        NotImplementedError: This solver is not yet implemented.
    """
    raise NotImplementedError(
        "retarget is not yet implemented. "
        "See docs/02_target_architecture.md for the planned design."
    )
