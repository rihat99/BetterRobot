"""Forward kinematics as a free function."""
from __future__ import annotations
import torch
from ...models.robot_model import RobotModel
from ...math.se3 import se3_compose, se3_identity, se3_apply_base, se3_from_axis_angle, se3_from_translation

__all__ = [
    "forward_kinematics",
]



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
                T_motion = se3_from_axis_angle(axis, angle)
            else:
                T_motion = se3_from_translation(axis, angle)
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
        if base_pose is not None:
            raise ValueError(
                "base_pose must not be passed when q_or_data is RobotData; "
                "set data.base_pose instead."
            )
        data = q_or_data
        if data._model_id != -1 and data._model_id != id(model):
            raise ValueError(
                "RobotData was created from a different RobotModel. "
                "Use model.create_data() to create data for this model."
            )
        if data.fk_poses is None:
            data.fk_poses = _fk_impl(model, data.q, data.base_pose)
        return data.fk_poses
    q = q_or_data
    return _fk_impl(model, q, base_pose)
