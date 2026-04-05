"""Analytical Jacobian functions for kinematics.

Convention
----------
- compute_jacobian applies pos_weight and ori_weight internally.
- jlog approximated as identity (valid for errors < ~30 degrees).
"""
from __future__ import annotations

import pypose as pp
import torch

from ...models.robot_model import RobotModel
from ...math.se3 import se3_compose


def compute_jacobian(
    model: RobotModel,
    cfg: torch.Tensor,
    target_link_index: int,
    target_pose: torch.Tensor,
    pos_weight: float,
    ori_weight: float,
    base_pose: torch.Tensor | None = None,
    fk: torch.Tensor | None = None,
) -> torch.Tensor:
    """Geometric (body-frame) Jacobian of the pose residual wrt cfg.

    Args:
        model: Robot model.
        cfg: (num_actuated_joints,) current joint config.
        target_link_index: Index of the target link.
        target_pose: (7,) SE3 target. Kept for API symmetry; unused in Jacobian.
        pos_weight: Weight on position rows.
        ori_weight: Weight on orientation rows.
        base_pose: (7,) optional floating base SE3.
        fk: (num_links, 7) pre-computed FK result. Skips FK call if provided.

    Returns:
        (6, num_actuated_joints) Jacobian matrix.
    """
    device, dtype = cfg.device, cfg.dtype
    n = model.joints.num_actuated_joints

    if fk is None:
        fk = model.forward_kinematics(cfg, base_pose=base_pose)
    T_ee = fk[target_link_index]
    p_ee = T_ee[:3]
    R_ee = pp.SO3(T_ee[3:7]).matrix()

    chain = model.get_chain(target_link_index)
    J = torch.zeros(6, n, dtype=dtype, device=device)

    for j in chain:
        cfg_idx = model._fk_cfg_indices[j]
        parent_link = model._fk_joint_parent_link[j]

        T_parent = fk[parent_link]
        T_origin = model._fk_joint_origins[j].to(device=device, dtype=dtype)
        T_j = se3_compose(T_parent, T_origin)

        p_j = T_j[:3]
        local_axis = model._fk_joint_axes[j].to(device=device, dtype=dtype)
        axis_world = pp.SO3(T_j[3:7]).Act(local_axis)

        jtype = model._fk_joint_types[j]
        if jtype in ("revolute", "continuous"):
            lin_world = torch.linalg.cross(axis_world, p_ee - p_j)
            J[:, cfg_idx] = torch.cat([
                R_ee.T @ lin_world * pos_weight,
                R_ee.T @ axis_world * ori_weight,
            ])
        else:  # prismatic
            J[:, cfg_idx] = torch.cat([
                R_ee.T @ axis_world * pos_weight,
                torch.zeros(3, dtype=dtype, device=device),
            ])

    return J


def limit_jacobian(cfg: torch.Tensor, model: RobotModel) -> torch.Tensor:
    """Jacobian of limit_residual wrt cfg.

    Returns:
        (2 * num_actuated_joints, num_actuated_joints) diagonal Jacobian.
    """
    n = model.joints.num_actuated_joints
    lo = model.joints.lower_limits.to(device=cfg.device, dtype=cfg.dtype)
    hi = model.joints.upper_limits.to(device=cfg.device, dtype=cfg.dtype)

    lower_grad = torch.where(cfg < lo, torch.full_like(cfg, -1.0), torch.zeros_like(cfg))
    upper_grad = torch.where(cfg > hi, torch.ones_like(cfg), torch.zeros_like(cfg))

    J = torch.zeros(2 * n, n, dtype=cfg.dtype, device=cfg.device)
    J[:n] = torch.diag(lower_grad)
    J[n:] = torch.diag(upper_grad)
    return J


def rest_jacobian(cfg: torch.Tensor, rest_pose: torch.Tensor) -> torch.Tensor:
    """Jacobian of rest_residual wrt cfg. Returns identity matrix.

    Returns:
        (num_actuated_joints, num_actuated_joints) identity.
    """
    n = len(cfg)
    return torch.eye(n, dtype=cfg.dtype, device=cfg.device)
