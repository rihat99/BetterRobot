"""Kinematic chain computation as a free function."""
from __future__ import annotations
from ...models.robot_model import RobotModel

__all__ = [
    "get_chain",
]


def get_chain(model: RobotModel, link_idx: int) -> list[int]:
    """Joint indices (actuated only) on path from root to link_idx.

    Args:
        model: Robot model.
        link_idx: Target link index.

    Returns:
        List of actuated joint indices in root->EE order.
    """
    child_to_joint: dict[int, int] = {
        child: j for j, child in enumerate(model._fk_joint_child_link)
    }
    chain: list[int] = []
    current = link_idx
    while current != model._root_link_idx:
        if current not in child_to_joint:
            raise ValueError(
                f"Link index {link_idx} is not reachable from root "
                f"(stuck at link {current}). Possible orphaned link."
            )
        j = child_to_joint[current]
        if model._fk_cfg_indices[j] >= 0:
            chain.append(j)
        current = model._fk_joint_parent_link[j]
    chain.reverse()
    return chain
