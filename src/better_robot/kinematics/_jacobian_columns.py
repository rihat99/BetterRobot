"""Batched column-table lane for joint Jacobians.

:func:`~better_robot.kinematics.jacobian.joint_jacobians_raw` delegates its
per-joint work here. The world Jacobian has a purely static sparsity: the
column block owned by joint ``a`` holds ``Ad(oM_a) @ S_a`` and appears in row
joint ``j``'s Jacobian iff ``a`` supports ``j`` (is an ancestor of, or equals,
``j``). Neither the column content nor the sparsity depends on the row joint, so
the whole pass is three batched stages instead of a per-joint Python loop:

1.  One batched ``so3.to_matrix`` / ``hat_so3`` over ``joint_pose_world`` and the
    constant :attr:`ModelStructure.joint_motion_subspaces` table build every
    joint's own column block at once.
2.  A static gather scatters the padded ``(njoints, max_nv)`` own blocks into a
    shared ``(B..., 6, nv_full)`` column table.
3.  A static supports-derived mask broadcasts that shared table into per-joint
    world Jacobians.

The plan (owner map, gather index, support mask) is a pure function of
:class:`ModelStructure`, memoised on the structure instance exactly like
``_fk_matrix_plan``; :meth:`ModelStructure.to` mints a fresh instance, so a
moved model rebuilds its plan on the new device. Order 03's Jacobian
time-variation pass reuses :func:`world_column_table` and the plan's owner map.

See ``docs/concepts/kinematics_and_jacobians.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..data_model.model_structure import ModelStructure
from ..lie import so3
from ..lie.tangents import hat_so3


@dataclass(frozen=True)
class JointJacobianPlan:
    """Static, device-resident schedule for the batched joint-Jacobian lane.

    Contract consumed by the time-variation pass (order 03): the shared world
    column table from :func:`world_column_table` plus this owner map are all it
    needs to build ``J̇`` column-wise.
    """

    max_nv: int  #: padded per-joint subspace width (``joint_motion_subspaces.shape[-1]``).
    column_owner: torch.Tensor  #: ``(nv_full,)`` long — joint owning each full-width v-column.
    column_gather: torch.Tensor  #: ``(nv_full,)`` long — flat index into a ``(njoints * max_nv)`` own-column grid selecting each full-width column's padded source.
    support_mask: torch.Tensor  #: ``(njoints, 1, nv_full)`` bool — True where ``column_owner`` supports the row joint; masks the shared column table into per-joint Jacobians.


def build_plan(structure: ModelStructure) -> JointJacobianPlan:
    """Derive the static joint-Jacobian schedule from topology.

    Pure function of ``structure``; device-resident and reusable for every
    configuration. Prefer :func:`get_plan`, which memoises this.
    """

    device = structure.idx_vs_full_tensor.device
    njoints = structure.njoints
    nv_full = structure.nv_full
    max_nv = int(structure.joint_motion_subspaces.shape[-1])

    owner = [0] * nv_full
    gather = [0] * nv_full
    for joint_id in range(njoints):
        base = structure.idx_vs_full[joint_id]
        for local_col in range(structure.nvs_full[joint_id]):
            column = base + local_col
            owner[column] = joint_id
            gather[column] = joint_id * max_nv + local_col

    is_support = torch.zeros((njoints, njoints), dtype=torch.bool, device=device)
    for joint_id in range(njoints):
        for ancestor in structure.supports[joint_id]:
            is_support[joint_id, ancestor] = True

    column_owner = torch.tensor(owner, dtype=torch.long, device=device)
    support_mask = is_support.index_select(-1, column_owner).unsqueeze(-2)  # (njoints, 1, nv_full)
    return JointJacobianPlan(
        max_nv=max_nv,
        column_owner=column_owner,
        column_gather=torch.tensor(gather, dtype=torch.long, device=device),
        support_mask=support_mask.contiguous(),
    )


def get_plan(structure: ModelStructure) -> JointJacobianPlan:
    """Return the memoised joint-Jacobian plan for ``structure``.

    Cached as a private attribute on the (otherwise immutable) structure;
    :meth:`ModelStructure.to` builds a fresh instance, so a moved model
    rebuilds its plan on the new device.
    """

    plan = getattr(structure, "_joint_jacobian_plan", None)
    if plan is None:
        plan = build_plan(structure)
        object.__setattr__(structure, "_joint_jacobian_plan", plan)
    return plan


def world_column_table(
    structure: ModelStructure,
    joint_pose_world: torch.Tensor,
    plan: JointJacobianPlan,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Shared full-width world column table ``(B..., 6, nv_full)``.

    Column ``c`` holds ``Ad(oM_a) @ S_a[:, k]`` for the owning joint
    ``a = plan.column_owner[c]`` and its local column ``k`` — the world Jacobian
    content of that column, independent of the row joint that reads it. Per-joint
    masking (:attr:`JointJacobianPlan.support_mask`) and mimic reduction are the
    caller's job. Order 03's ``J̇`` pass consumes this table plus the owner map.

    ``joint_pose_world`` is ``(B..., njoints, 7)``; ``device``/``dtype`` fix the
    compute precision — the constant subspace table is cast to match, mirroring
    the legacy per-joint promotion.
    """

    rotation = so3.to_matrix(joint_pose_world[..., 3:])  # (B..., njoints, 3, 3)
    hat_position = hat_so3(joint_pose_world[..., :3])  # (B..., njoints, 3, 3)
    subspace = structure.joint_motion_subspaces.to(device=device, dtype=dtype)  # (njoints, 6, max_nv)

    rotated_angular = torch.matmul(rotation, subspace[..., 3:, :])  # (B..., njoints, 3, max_nv)
    linear = torch.matmul(rotation, subspace[..., :3, :]) + torch.matmul(hat_position, rotated_angular)
    own_columns = torch.cat((linear, rotated_angular), dim=-2)  # (B..., njoints, 6, max_nv)

    # Move the 6-row axis ahead of (njoints, max_nv), flatten the owner grid,
    # and select each full-width column's padded source. Static gather → vmap-
    # and compile-safe; padding columns are never selected.
    flattened = own_columns.transpose(-3, -2).flatten(-2, -1)  # (B..., 6, njoints * max_nv)
    return flattened.index_select(-1, plan.column_gather)  # (B..., 6, nv_full)


__all__ = ["JointJacobianPlan", "build_plan", "get_plan", "world_column_table"]
