"""Kinematic chain computation as a free function."""
from __future__ import annotations
from ...models.robot_model import RobotModel


def get_chain(model: RobotModel, link_idx: int) -> list[int]:
    """Joint indices (actuated only) on path from root to link_idx.

    Args:
        model: Robot model.
        link_idx: Target link index.

    Returns:
        List of actuated joint indices in root->EE order.
    """
    return model.get_chain(link_idx)
