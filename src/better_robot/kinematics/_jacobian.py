"""Shared world Jacobian column table: one batched pass for every joint and frame.

The world Jacobian has a purely static sparsity: the column block owned by joint ``a`` holds ``Ad(oM_a) @ S_a``
and appears in row joint ``j``'s Jacobian iff ``a`` supports ``j`` (is an ancestor of, or equals, ``j``). Neither
the column content nor the sparsity depends on the row joint, so one static gather of each column's owner
rotation and origin, times the constant joint-frame subspace column (`_plans.JacobianPlan.column_subspace`),
builds the shared ``(B..., 6, nv_full)`` table at once; no padded ``(njoints, max_nv)`` grid is formed, so a free
flyer does not pad every joint to six columns. A joint's Jacobian is the table masked by its support
(`_plans.JacobianPlan.support_mask`) and mimic-reduced last (`configuration.reduced_columns`); the time variation
transports the same table (`jacobians`). The pass is a constant number of launches independent of the joint
count: against the per-joint loop it replaced, a 203-joint model at batch 256 went from 47.3 to 0.84 ms forward
(CPU, one thread). `KinematicState.jacobian_columns` builds the table once per state. Everything stays autograd-
and ``torch.func``-safe; no host syncs.
"""

from __future__ import annotations

import math

import torch

from ._plans import JacobianPlan


def world_column_table(transforms: torch.Tensor, plan: JacobianPlan) -> torch.Tensor:
    """``(B..., 6, nv_full)`` world column table from world joint transforms ``[R | p]: (B..., njoints, 3, 4)``.

    Column ``c`` holds ``Ad(oM_a) @ S_a[:, k]`` for the owning joint ``a = plan.column_owner[c]`` and its local
    column ``k``: the world Jacobian content of that column, independent of the row joint that reads it. Per-joint
    masking and mimic reduction are the caller's job. Trusts validated inputs.
    """
    owners = transforms.index_select(-3, plan.column_owner)  # (B..., nv_full, 3, 4)
    rotated = owners[..., :3] @ plan.column_subspace  # (B..., nv_full, 3, 2): R s_linear, R s_angular
    angular = rotated[..., 1]
    linear = rotated[..., 0] + torch.linalg.cross(owners[..., 3], angular)
    return torch.cat((linear, angular), dim=-1).transpose(-1, -2)  # (B..., 6, nv_full)


def geometry_about_first_joint(
    joint_transforms: torch.Tensor, columns: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(columns, rotations, translations, origin)`` about each sample's first joint, the batch flattened to ``N``.

    From world ``joint_transforms (B..., njoints, 3, 4)`` and the world column table ``columns (B..., 6, nv_full)``:
    the columns moved to ``origin`` ``(N, 6, nv_full)``, the joint rotations ``(N, njoints, 3, 3)``, the joint origins
    relative to ``origin`` ``(N, njoints, 3)`` and ``origin (N, 3)``, the position of joint 1 (joint 0 for a
    one-joint model). A column's linear part is the velocity of the point at the world origin, so at ``origin`` it is
    ``linear + angular × origin``. Trusts validated inputs.
    """
    nj, nv_full = joint_transforms.shape[-3], columns.shape[-1]
    count = math.prod(joint_transforms.shape[:-3])
    transforms = joint_transforms.reshape(count, nj, 3, 4)
    translations = transforms[..., 3]
    origin = translations[:, min(1, nj - 1)]
    columns = columns.reshape(count, 6, nv_full)
    angular = columns[:, 3:]
    linear = columns[:, :3] + torch.linalg.cross(angular, origin.unsqueeze(-1), dim=-2)
    return torch.cat((linear, angular), dim=-2), transforms[..., :3], translations - origin.unsqueeze(-2), origin


__all__ = ["geometry_about_first_joint", "world_column_table"]
