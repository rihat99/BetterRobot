"""Unified Jacobian entry points.

Pinocchio-style canonical functions — a single dispatch path that replaces
the legacy four-way fixed/floating × analytic/autodiff mess.

See ``docs/concepts/kinematics_and_jacobians.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from ..data_model import KinematicsLevel
from ..data_model.data import Data
from ..data_model.execution_batch import broadcast_to_execution_batch
from ..data_model.model import Model
from ..data_model.model_structure import ModelStructure
from ..data_model.reduced_coordinates import (
    expand_configuration,
    reduce_jacobian,
)
from ..lie import se3, so3
from ..lie.tangents import hat_so3

_ReferenceFrame = Literal["world", "local", "local_world_aligned"]


@dataclass(frozen=True)
class JointJacobiansResult:
    """Fresh world-frame joint Jacobians from :func:`joint_jacobians_raw`."""

    joint_jacobians: torch.Tensor


def joint_jacobians_raw(
    structure: ModelStructure,
    q: torch.Tensor,
    joint_pose_world: torch.Tensor,
) -> JointJacobiansResult:
    """Tensor-only joint-Jacobian primitive with no ``Data`` sequencing.

    Uses the propagation trick: ``J[j] = J[parent[j]]`` then adds the
    contribution of joint ``j`` itself. ``joint_pose_world`` has shape
    ``(B..., njoints, 7)`` and the result field has shape
    ``(B..., njoints, 6, nv)``.
    """
    batch = tuple(joint_pose_world.shape[:-2])
    q = broadcast_to_execution_batch(
        q,
        batch,
        (structure.nq,),
        name="q",
    )
    q_full = expand_configuration(structure, q)
    device, dtype = q.device, q.dtype

    J = torch.zeros(
        *batch,
        structure.njoints,
        6,
        structure.nv_full,
        device=device,
        dtype=dtype,
    )

    for j in structure.topo_order:
        parent = structure.parents[j]
        if parent >= 0:
            J[..., j, :, :] = J[..., parent, :, :]

        nv_j = structure.nvs_full[j]
        v_j = structure.idx_vs_full[j]

        if nv_j == 0:
            continue

        T_j = joint_pose_world[..., j, :]  # (B..., 7)
        p_j = T_j[..., :3]  # (B..., 3)
        R_j = so3.to_matrix(T_j[..., 3:])  # (B..., 3, 3)
        hat_p = hat_so3(p_j)  # (B..., 3, 3)

        nq_j = structure.nqs_full[j]
        q_j = q_full[..., structure.idx_qs_full[j] : structure.idx_qs_full[j] + nq_j]
        S_local = structure.joint_models[j].joint_motion_subspace(q_j)
        S_local = S_local.to(device=device, dtype=dtype)  # (B..., 6, nv_j)

        S_lin = S_local[..., :3, :]  # (B..., 3, nv_j)
        S_ang = S_local[..., 3:, :]  # (B..., 3, nv_j)

        # Ad(joint_pose_world[j]) @ S_local = [R @ S_lin + hat(p) @ R @ S_ang; R @ S_ang]
        R_S_lin = torch.matmul(R_j, S_lin)  # (B..., 3, nv_j)
        R_S_ang = torch.matmul(R_j, S_ang)  # (B..., 3, nv_j)
        hat_p_R_S_ang = torch.matmul(hat_p, R_S_ang)  # (B..., 3, nv_j)

        J[..., j, :3, v_j : v_j + nv_j] = R_S_lin + hat_p_R_S_ang
        J[..., j, 3:, v_j : v_j + nv_j] = R_S_ang

    return JointJacobiansResult(joint_jacobians=reduce_jacobian(structure, J))


def compute_joint_jacobians(model: Model, data: Data) -> Data:
    """Populate ``data.joint_jacobians`` with the spatial Jacobian of every joint.

    Shape: ``data.joint_jacobians = (B..., njoints, 6, nv)``. Requires
    ``data`` to hold at least ``KinematicsLevel.PLACEMENTS`` — call
    :func:`forward_kinematics` first; otherwise raises
    :class:`~better_robot.exceptions.StaleCacheError`.

    See docs/concepts/kinematics_and_jacobians.md.
    """
    data.require(KinematicsLevel.PLACEMENTS)
    assert data.joint_pose_world is not None
    result = joint_jacobians_raw(model.structure, data.q, data.joint_pose_world)
    data.joint_jacobians = result.joint_jacobians
    return data


def get_joint_jacobian(
    model: Model,
    data: Data,
    joint_id: int,
    *,
    reference: _ReferenceFrame = "world",
) -> torch.Tensor:
    """Extract the spatial Jacobian of a single joint from ``data.joint_jacobians``.

    Shape: ``(B..., 6, nv)``. The literal reference strings mirror
    Pinocchio's three reference-frame conventions.

    See docs/concepts/kinematics_and_jacobians.md.
    """
    data.require(KinematicsLevel.PLACEMENTS)
    if data.joint_jacobians is None:
        compute_joint_jacobians(model, data)

    J_j = data.joint_jacobians[..., joint_id, :, :]  # (B..., 6, nv)

    if reference == "world":
        return J_j
    elif reference == "local":
        T_j = data.joint_pose_world[..., joint_id, :]
        return torch.matmul(se3.adjoint_inv(T_j), J_j)
    else:
        raise ValueError(f"Unsupported reference frame: {reference!r}")


def get_frame_jacobian(
    model: Model,
    data: Data,
    frame_id: int,
    *,
    reference: _ReferenceFrame = "local_world_aligned",
) -> torch.Tensor:
    """Spatial Jacobian of an arbitrary frame.

    Three pinocchio-style reference frames — all returning ``(B..., 6, nv)``:

    - ``"local_world_aligned"`` (default): linear rows are the velocity of the
      frame origin expressed in the world frame; angular rows are the angular
      velocity expressed in the world frame. This matches Pinocchio's
      ``LOCAL_WORLD_ALIGNED`` and is the natural basis for position/pose
      residuals.
    - ``"world"``: linear rows are the velocity of the world-coincident point
      of the frame (i.e. the spatial velocity at the world origin); angular
      rows are the angular velocity in world frame. Matches Pinocchio's
      ``WORLD``.
    - ``"local"``: both linear and angular rows expressed in the body-local
      frame of this frame. Matches Pinocchio's ``LOCAL``.

    See docs/concepts/kinematics_and_jacobians.md.
    """
    data.require(KinematicsLevel.PLACEMENTS)
    assert data.joint_pose_world is not None, "call forward_kinematics before get_frame_jacobian"

    frame = model.frames[frame_id]
    parent_joint = frame.parent_joint
    joint_pose_world = data.joint_pose_world
    batch = tuple(joint_pose_world.shape[:-2])
    q = broadcast_to_execution_batch(
        data.q,
        batch,
        (model.nq,),
        name="data.q",
    )
    device, dtype = q.device, q.dtype

    # -- Build the parent joint's world-frame Jacobian --
    # This is the WORLD Jacobian of the parent joint (velocity at world origin).
    # Use data.joint_jacobians if already computed; otherwise compute directly.
    if data.joint_jacobians is not None:
        J_parent = data.joint_jacobians[..., parent_joint, :, :]  # (B..., 6, nv)
    else:
        q_full = expand_configuration(model.structure, q)
        J_parent_full = torch.zeros(*batch, 6, model.nv_full, device=device, dtype=dtype)
        support = model.get_support(parent_joint)
        for j in support:
            nv_j = model.nvs_full[j]
            v_j = model.idx_vs_full[j]
            if nv_j == 0:
                continue
            T_j = joint_pose_world[..., j, :]
            p_j = T_j[..., :3]
            R_j = so3.to_matrix(T_j[..., 3:])
            hat_p = hat_so3(p_j)

            nq_j = model.nqs_full[j]
            q_j = q_full[..., model.idx_qs_full[j] : model.idx_qs_full[j] + nq_j]
            S_local = model.joint_models[j].joint_motion_subspace(q_j)
            S_local = S_local.to(device=device, dtype=dtype)

            S_lin = S_local[..., :3, :]
            S_ang = S_local[..., 3:, :]

            R_S_ang = torch.matmul(R_j, S_ang)
            R_S_lin = torch.matmul(R_j, S_lin)
            hat_p_R_S_ang = torch.matmul(hat_p, R_S_ang)

            J_parent_full[..., :3, v_j : v_j + nv_j] = R_S_lin + hat_p_R_S_ang
            J_parent_full[..., 3:, v_j : v_j + nv_j] = R_S_ang
        J_parent = reduce_jacobian(model.structure, J_parent_full)

    # -- Compute the frame's world pose (needed for LWA and LOCAL adjustments) --
    T_local = broadcast_to_execution_batch(
        model.values.frame_placements[..., frame_id, :],
        batch,
        (7,),
        name="frame_placements row",
    )
    T_parent = joint_pose_world[..., parent_joint, :]
    T_frame = se3.compose(T_parent, T_local)  # (B..., 7) world frame pose
    p_frame = T_frame[..., :3]  # (B..., 3) world position

    if reference == "world":
        # Spatial velocity at world origin: frame's WORLD Jacobian equals its
        # parent joint's (rigid attachment). Pose-offset does not affect it.
        return J_parent

    # Both LWA and LOCAL start from LWA: linear rows = velocity of the frame
    # origin in world frame. v_at_pf = v_at_world_origin + ω × p_f
    #                               = J_parent_lin - hat(p_f) @ J_parent_ang.
    hat_pf = hat_so3(p_frame)  # (B..., 3, 3)
    J_lwa_lin = J_parent[..., :3, :] - torch.matmul(hat_pf, J_parent[..., 3:, :])
    J_lwa = torch.cat([J_lwa_lin, J_parent[..., 3:, :]], dim=-2)  # (B..., 6, nv)

    if reference == "local_world_aligned":
        return J_lwa
    elif reference == "local":
        # Body-frame Jacobian: rotate the LWA rows by R_frame^T.
        # Do NOT apply full Ad(T_frame^{-1}) — that would subtract hat(p) twice.
        R_frame = so3.to_matrix(T_frame[..., 3:])  # (B..., 3, 3)
        J_local_lin = torch.matmul(R_frame.mT, J_lwa[..., :3, :])
        J_local_ang = torch.matmul(R_frame.mT, J_lwa[..., 3:, :])
        return torch.cat([J_local_lin, J_local_ang], dim=-2)
    else:
        raise ValueError(f"Unsupported reference frame: {reference!r}")
