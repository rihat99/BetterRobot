"""Analytical Jacobian functions for IK cost terms.

All functions return the Jacobian dR/dcfg where R is the corresponding
residual vector (from _pose.py, _limits.py, _regularization.py).

Convention
----------
- pose_jacobian applies pos_weight and ori_weight internally (matching pose_residual).
- jlog is approximated as identity (valid for errors < ~30°); noted in CLAUDE.md.
"""

from __future__ import annotations

import pypose as pp
import torch

from ..core._robot import Robot
from ..core._lie_ops import se3_compose


def pose_jacobian(
    cfg: torch.Tensor,
    robot: Robot,
    target_link_index: int,
    target_pose: torch.Tensor,
    pos_weight: float,
    ori_weight: float,
    base_pose: torch.Tensor | None = None,
) -> torch.Tensor:
    """Geometric Jacobian of pose_residual wrt cfg.

    Args:
        cfg: (num_actuated_joints,) current joint config.
        robot: Robot instance.
        target_link_index: Index of the target link.
        target_pose: (7,) SE3 target [tx, ty, tz, qx, qy, qz, qw]. Unused here
            (Jacobian at current cfg, not at target); kept for API symmetry.
        pos_weight: Weight on position rows (matches pose_residual).
        ori_weight: Weight on orientation rows (matches pose_residual).
        base_pose: (7,) optional floating base SE3. Passed to forward_kinematics.

    Returns:
        (6, num_actuated_joints) Jacobian matrix.
    """
    device, dtype = cfg.device, cfg.dtype
    n = robot.joints.num_actuated_joints

    fk = robot.forward_kinematics(cfg, base_pose=base_pose)   # (num_links, 7)
    T_ee = fk[target_link_index]
    p_ee = T_ee[:3]
    R_ee = pp.SO3(T_ee[3:7]).matrix()                                    # (3, 3) EE rotation

    chain = robot.get_chain(target_link_index)

    J = torch.zeros(6, n, dtype=dtype, device=device)

    for j in chain:
        cfg_idx = robot._fk_cfg_indices[j]
        parent_link = robot._fk_joint_parent_link[j]

        T_parent = fk[parent_link]                                        # (7,) world
        T_origin = robot._fk_joint_origins[j].to(device=device, dtype=dtype)  # (7,)
        T_j = se3_compose(T_parent, T_origin)                            # joint frame in world

        p_j = T_j[:3]                                                    # joint origin (world)
        local_axis = robot._fk_joint_axes[j].to(device=device, dtype=dtype)  # (3,)
        axis_world = pp.SO3(T_j[3:7]).Act(local_axis)                   # axis in world frame

        jtype = robot._fk_joint_types[j]
        if jtype in ("revolute", "continuous"):
            # Body-frame Jacobian: pose_residual = log(T_target^{-1} @ T_actual) is a
            # RIGHT log error, so its Jacobian at zero error is the body (EE) frame screw.
            # J_body[:3] = R_ee^T @ (ω × (p_ee - p_j))  (EE linear)
            # J_body[3:]  = R_ee^T @ ω                   (EE angular)
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


def limit_jacobian(cfg: torch.Tensor, robot: Robot) -> torch.Tensor:
    """Jacobian of limit_residual wrt cfg.

    limit_residual returns [clamp(lo-cfg, min=0), clamp(cfg-hi, min=0)].
    Derivative: -1 where cfg < lo (lower block), +1 where cfg > hi (upper block).

    Returns:
        (2 * num_actuated_joints, num_actuated_joints) diagonal Jacobian.
    """
    n = robot.joints.num_actuated_joints
    lo = robot.joints.lower_limits.to(device=cfg.device, dtype=cfg.dtype)
    hi = robot.joints.upper_limits.to(device=cfg.device, dtype=cfg.dtype)

    lower_grad = torch.where(cfg < lo, torch.full_like(cfg, -1.0), torch.zeros_like(cfg))
    upper_grad = torch.where(cfg > hi, torch.ones_like(cfg), torch.zeros_like(cfg))

    J = torch.zeros(2 * n, n, dtype=cfg.dtype, device=cfg.device)
    J[:n] = torch.diag(lower_grad)
    J[n:] = torch.diag(upper_grad)
    return J


def rest_jacobian(cfg: torch.Tensor, rest_pose: torch.Tensor) -> torch.Tensor:
    """Jacobian of rest_residual wrt cfg.

    rest_residual = cfg - rest_pose, so J = I.

    Returns:
        (num_actuated_joints, num_actuated_joints) identity matrix.
    """
    n = len(cfg)
    return torch.eye(n, dtype=cfg.dtype, device=cfg.device)
