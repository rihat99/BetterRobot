"""Forward kinematics as a free function."""
from __future__ import annotations
import torch
from ...models.robot_model import RobotModel
from ...math.se3 import se3_compose, se3_identity, se3_apply_base

__all__ = [
    "forward_kinematics",
]


def _revolute_transform(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Pure rotation SE3: [0, 0, 0, sin*ax, sin*ay, sin*az, cos(a/2)]."""
    half = angle / 2.0
    cos_h = torch.cos(half)
    sin_h = torch.sin(half)
    qxyz = sin_h.unsqueeze(-1) * axis
    zeros = torch.zeros(*angle.shape, 3, device=angle.device, dtype=angle.dtype)
    return torch.cat([zeros, qxyz, cos_h.unsqueeze(-1)], dim=-1)


def _prismatic_transform(axis: torch.Tensor, displacement: torch.Tensor) -> torch.Tensor:
    """Pure translation SE3: [d*ax, d*ay, d*az, 0, 0, 0, 1]."""
    batch_shape = displacement.shape
    device, dtype = displacement.device, displacement.dtype
    trans = displacement.unsqueeze(-1) * axis.to(device=device, dtype=dtype)
    qxyz = torch.zeros(*batch_shape, 3, device=device, dtype=dtype)
    qw = torch.ones(*batch_shape, 1, device=device, dtype=dtype)
    return torch.cat([trans, qxyz, qw], dim=-1)


def _fk_impl(
    model: RobotModel,
    q: torch.Tensor,
    base_pose: torch.Tensor | None = None,
) -> torch.Tensor:
    """Core FK implementation operating on raw tensors."""
    batch_shape = q.shape[:-1]
    device, dtype = q.device, q.dtype

    link_world: dict[int, torch.Tensor] = {}
    link_world[model._root_link_idx] = (
        se3_identity()
        .to(device=device, dtype=dtype)
        .expand(*batch_shape, 7)
        .clone()
    )

    for j_idx in model._fk_joint_order:
        parent_link = model._fk_joint_parent_link[j_idx]
        child_link = model._fk_joint_child_link[j_idx]

        T_parent = link_world[parent_link]
        T_origin = model._fk_joint_origins[j_idx].to(device=device, dtype=dtype)
        T = se3_compose(T_parent, T_origin.expand_as(T_parent))

        cfg_idx = model._fk_cfg_indices[j_idx]
        if cfg_idx >= 0:
            angle = q[..., cfg_idx]
            axis = model._fk_joint_axes[j_idx].to(device=device, dtype=dtype)
            jtype = model._fk_joint_types[j_idx]
            if jtype in ('revolute', 'continuous'):
                T_motion = _revolute_transform(axis, angle)
            else:
                T_motion = _prismatic_transform(axis, angle)
            T = se3_compose(T, T_motion)

        link_world[child_link] = T

    result = torch.stack(
        [link_world[i] for i in range(model.links.num_links)], dim=-2
    )
    if base_pose is not None:
        result = se3_apply_base(base_pose, result)
    return result


def forward_kinematics(
    model: RobotModel,
    q_or_data: "torch.Tensor | RobotData",
    base_pose: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute world poses of all links.

    Args:
        model: Robot model (immutable kinematic structure).
        q_or_data: Joint configuration (*batch, num_actuated_joints), or a
            RobotData instance (uses data.q and data.base_pose).
        base_pose: Optional base transform (*batch, 7). Ignored if q_or_data
            is RobotData. Default None.

    Returns:
        Link poses (*batch, num_links, 7) in [tx, ty, tz, qx, qy, qz, qw].
    """
    from ...models.data import RobotData
    if isinstance(q_or_data, RobotData):
        data = q_or_data
        if data.fk_poses is None:
            data.fk_poses = _fk_impl(model, data.q, data.base_pose)
        return data.fk_poses
    cfg = q_or_data
    return _fk_impl(model, cfg, base_pose)
