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
from ..data_model.model_values import ModelValues
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


def frame_jacobian_raw(
    structure: ModelStructure,
    values: ModelValues,
    q: torch.Tensor,
    joint_pose_world: torch.Tensor,
    frame_id: int,
    *,
    reference: _ReferenceFrame = "local_world_aligned",
    joint_jacobians: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return one frame Jacobian without reading or mutating :class:`Data`.

    ``joint_pose_world`` normally comes from :func:`forward_kinematics_raw`.
    Supplying a compatible ``joint_jacobians`` tensor reuses an existing
    :func:`joint_jacobians_raw` result; otherwise this pass computes it.
    """
    batch = tuple(joint_pose_world.shape[:-2])
    q = broadcast_to_execution_batch(q, batch, (structure.nq,), name="q")
    if joint_jacobians is None:
        joint_jacobians = joint_jacobians_raw(structure, q, joint_pose_world).joint_jacobians

    parent_index = structure.frame_parent_joints[frame_id : frame_id + 1]
    J_parent = torch.index_select(joint_jacobians, -3, parent_index).squeeze(-3)
    T_local = broadcast_to_execution_batch(
        values.frame_placements[..., frame_id, :],
        batch,
        (7,),
        name="frame_placements row",
    )
    T_parent = torch.index_select(joint_pose_world, -2, parent_index).squeeze(-2)
    T_frame = se3.compose(T_parent, T_local)

    if reference == "world":
        return J_parent

    hat_position = hat_so3(T_frame[..., :3])
    linear = J_parent[..., :3, :] - torch.matmul(hat_position, J_parent[..., 3:, :])
    aligned = torch.cat((linear, J_parent[..., 3:, :]), dim=-2)
    if reference == "local_world_aligned":
        return aligned
    if reference == "local":
        rotation = so3.to_matrix(T_frame[..., 3:])
        return torch.cat(
            (
                torch.matmul(rotation.mT, aligned[..., :3, :]),
                torch.matmul(rotation.mT, aligned[..., 3:, :]),
            ),
            dim=-2,
        )
    raise ValueError(f"Unsupported reference frame: {reference!r}")


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

    Three pinocchio-style reference frames — all returning ``(B..., 6, nv)``:

    - ``"world"`` (default): linear rows are the velocity of the world-coincident
      point of the joint (the spatial velocity at the world origin); angular rows
      are the angular velocity in world frame. This is the cached representation.
      Matches Pinocchio's ``WORLD``.
    - ``"local_world_aligned"``: linear rows are the velocity of the joint origin
      expressed in the world frame; angular rows are the angular velocity in world
      frame. Derived from the world Jacobian by translating the linear rows to the
      joint origin (angular rows unchanged) — a translation only, not a full
      adjoint. Matches Pinocchio's ``LOCAL_WORLD_ALIGNED``.
    - ``"local"``: both linear and angular rows expressed in the body-local frame
      of the joint, via the full inverse adjoint ``Ad(oMj)⁻¹``. Matches Pinocchio's
      ``LOCAL``.

    See docs/concepts/kinematics_and_jacobians.md.
    """
    data.require(KinematicsLevel.PLACEMENTS)
    if data.joint_jacobians is None:
        compute_joint_jacobians(model, data)

    J_j = data.joint_jacobians[..., joint_id, :, :]  # (B..., 6, nv)

    if reference == "world":
        return J_j
    elif reference == "local_world_aligned":
        p_j = data.joint_pose_world[..., joint_id, :3]
        linear = J_j[..., :3, :] - torch.matmul(hat_so3(p_j), J_j[..., 3:, :])
        return torch.cat((linear, J_j[..., 3:, :]), dim=-2)
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

    return frame_jacobian_raw(
        model.structure,
        model.values,
        data.q,
        data.joint_pose_world,
        frame_id,
        reference=reference,
        joint_jacobians=data.joint_jacobians,
    )
