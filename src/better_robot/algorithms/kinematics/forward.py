"""Forward kinematics as a free function."""
from __future__ import annotations
import torch
from ...models.robot_model import RobotModel


def forward_kinematics(
    model: RobotModel,
    q: torch.Tensor,
    base_pose: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute world poses of all links.

    Args:
        model: Robot model (immutable kinematic structure).
        q: Joint configuration (*batch, num_actuated_joints).
        base_pose: Optional base transform (*batch, 7). Default None.

    Returns:
        Link poses (*batch, num_links, 7) in [tx, ty, tz, qx, qy, qz, qw].
    """
    return model.forward_kinematics(q, base_pose=base_pose)
