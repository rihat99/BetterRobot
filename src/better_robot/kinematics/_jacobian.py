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

import torch

from ._plans import JacobianPlan


def world_column_table(rotations: torch.Tensor, translations: torch.Tensor, plan: JacobianPlan) -> torch.Tensor:
    """``(B..., 6, nv_full)`` world column table from world joint rotations ``(B..., njoints, 3, 3)`` and origins
    ``(B..., njoints, 3)``.

    Column ``c`` holds ``Ad(oM_a) @ S_a[:, k]`` for the owning joint ``a = plan.column_owner[c]`` and its local
    column ``k``: the world Jacobian content of that column, independent of the row joint that reads it. Per-joint
    masking and mimic reduction are the caller's job. Trusts validated inputs.
    """
    owner = plan.column_owner
    rotation = rotations.index_select(-3, owner)  # (B..., nv_full, 3, 3)
    translation = translations.index_select(-2, owner)  # (B..., nv_full, 3)
    subspace = plan.column_subspace  # (nv_full, 6)
    angular = (rotation @ subspace[:, 3:, None]).squeeze(-1)
    linear = (rotation @ subspace[:, :3, None]).squeeze(-1) + torch.linalg.cross(translation, angular)
    return torch.cat((linear, angular), dim=-1).transpose(-1, -2)  # (B..., 6, nv_full)


__all__ = ["world_column_table"]
