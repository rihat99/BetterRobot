"""Motion retargeting task."""

from __future__ import annotations

from typing import Literal

import torch

from ..core._robot import Robot


def retarget(
    source_robot: Robot,
    target_robot: Robot,
    source_motion: torch.Tensor,
    link_map: dict[str, str] | None = None,
    solver: Literal["lm", "gn", "adam", "lbfgs"] = "lm",
    weights: dict[str, float] | None = None,
    max_iter: int = 100,
) -> torch.Tensor:
    """Retarget a motion sequence from source robot to target robot.

    Matches specified link poses from source to target robot,
    while respecting target robot's joint limits.

    Args:
        source_robot: Robot whose motion is being retargeted.
        target_robot: Robot to retarget the motion onto.
        source_motion: Shape (T, source_num_actuated_joints). Source trajectory.
        link_map: Dict mapping source link names to target link names for pose matching.
            Example: {'source_hand': 'target_hand', 'source_foot': 'target_foot'}.
            If None, attempts to match links by name.
        solver: Which solver to use per frame. Default 'lm'.
        weights: Cost weights. Keys: 'pose', 'limits', 'smoothness'.
        max_iter: Maximum solver iterations per frame.

    Returns:
        Shape (T, target_num_actuated_joints). Retargeted trajectory.
    """
    raise NotImplementedError
