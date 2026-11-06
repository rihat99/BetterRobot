"""Batched column-table lane for joint Jacobians.

The Jacobian kernels in `state` and `jacobian` delegate their per-joint work here. The world Jacobian has a
purely static sparsity: the column block owned by joint ``a`` holds ``Ad(oM_a) @ S_a`` and appears in row joint
``j``'s Jacobian iff ``a`` supports ``j`` (is an ancestor of, or equals, ``j``). Neither the column content nor the
sparsity depends on the row joint, so the whole pass is two batched stages instead of a per-joint Python loop:

1.  A static gather of each column's owner rotation and origin, times the constant joint-frame subspace column,
    builds the shared ``(B..., 6, nv_full)`` column table in one batched pass (no padded ``(njoints, max_nv)`` grid).
2.  A static supports-derived mask broadcasts that shared table into per-joint world Jacobians.

The plan (owner map, subspace columns, support mask) is a pure function of the model's static fields, built once
per model, dtype and device through ``Model._plan``. The Jacobian time-variation pass reuses `world_column_table`
and the plan's owner map.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..lie import so3
from ..model.model import Model
from ..model.topology import build_supports


@dataclass(frozen=True)
class JointJacobianPlan:
    """Static, device-resident schedule for the batched joint-Jacobian lane."""

    max_nv: int  #: widest per-joint subspace.
    column_owner: torch.Tensor  #: ``(nv_full,)`` long: joint owning each full-width v-column.
    column_subspace: torch.Tensor  #: ``(nv_full, 6)``: the constant joint-frame motion subspace column of each column.
    support_mask: torch.Tensor  #: ``(njoints, 1, nv_full)`` bool: True where ``column_owner`` supports the row joint.


def build_plan(model: Model, dtype: torch.dtype, device: torch.device) -> JointJacobianPlan:
    """Derive the static joint-Jacobian schedule from the topology and the joints' neutral motion subspaces."""
    njoints, nv_full = model.njoints, model.nv_full
    max_nv = max(model.nvs_full, default=0)
    owner = [0] * nv_full
    gather = [0] * nv_full
    for joint_id in range(njoints):
        for local_col in range(model.nvs_full[joint_id]):
            column = model.idx_vs_full[joint_id] + local_col
            owner[column] = joint_id
            gather[column] = joint_id * max_nv + local_col
    is_support = torch.zeros((njoints, njoints), dtype=torch.bool, device=device)
    for joint_id, chain in enumerate(build_supports(model.parents)):
        is_support[joint_id, list(chain)] = True
    subspaces = torch.zeros((njoints, 6, max_nv), dtype=dtype, device=device)
    for joint_id, joint in enumerate(model.joints):
        if joint.nv:
            neutral = joint.neutral().to(device=device, dtype=dtype)
            subspaces[joint_id, :, : joint.nv] = joint.motion_subspace(neutral).to(device=device, dtype=dtype)
    column_owner = torch.tensor(owner, dtype=torch.long, device=device)
    column_gather = torch.tensor(gather, dtype=torch.long, device=device)
    return JointJacobianPlan(
        max_nv=max_nv,
        column_owner=column_owner,
        column_subspace=subspaces.transpose(-2, -1).reshape(-1, 6).index_select(0, column_gather).contiguous(),
        support_mask=is_support.index_select(-1, column_owner).unsqueeze(-2).contiguous(),
    )


def jacobian_plan(model: Model, *, dtype: torch.dtype, device: torch.device) -> JointJacobianPlan:
    """The model's `JointJacobianPlan` in ``dtype`` on ``device`` (built once per model, dtype and device)."""
    return model._plan("jacobian", build_plan, dtype=dtype, device=device)


def world_column_table(
    joint_pose_world: torch.Tensor, plan: JointJacobianPlan, *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Shared full-width world column table ``(B..., 6, nv_full)`` from world joint poses ``(B..., njoints, 7)``.

    Column ``c`` holds ``Ad(oM_a) @ S_a[:, k]`` for the owning joint ``a = plan.column_owner[c]`` and its local
    column ``k``: the world Jacobian content of that column, independent of the row joint that reads it. Per-joint
    masking (`JointJacobianPlan.support_mask`) and mimic reduction are the caller's job.
    """
    rotation = so3.to_matrix(joint_pose_world[..., 3:])  # (B..., njoints, 3, 3)
    return world_column_table_from_matrices(rotation, joint_pose_world[..., :3], plan, device=device, dtype=dtype)


def world_column_table_from_matrices(
    rotation: torch.Tensor,
    translation: torch.Tensor,
    plan: JointJacobianPlan,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """`world_column_table` from world rotations ``(B..., njoints, 3, 3)`` and origins ``(B..., njoints, 3)``.

    Each full-width column is built from its owner's rotation and origin directly, so no padded
    ``(njoints, max_nv)`` grid is formed (a free flyer would pad every joint to six columns).
    """
    owner = plan.column_owner
    rotation = rotation.index_select(-3, owner)  # (B..., nv_full, 3, 3)
    translation = translation.index_select(-2, owner)  # (B..., nv_full, 3)
    subspace = plan.column_subspace.to(device=device, dtype=dtype)  # (nv_full, 6)
    angular = (rotation @ subspace[:, 3:, None]).squeeze(-1)
    linear = (rotation @ subspace[:, :3, None]).squeeze(-1) + torch.linalg.cross(translation, angular)
    return torch.cat((linear, angular), dim=-1).transpose(-1, -2)  # (B..., 6, nv_full)


__all__ = ["JointJacobianPlan", "build_plan", "jacobian_plan", "world_column_table", "world_column_table_from_matrices"]
