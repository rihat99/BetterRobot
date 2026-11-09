"""Static schedules of the kinematics kernels, pure functions of a model's static fields (record 0011).

A plan fixes the structure of a computation so that only tensor values vary (principle 5.5): the FK plan holds
the per-joint feature map and the tree levels, the Jacobian plan the column owners, subspace columns and support
mask. Both are built once per model, dtype and device through ``Model._plan`` and read by `_fk` and `_jacobian`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import final

import torch

from ..lie.tangents import hat_so3_raw
from ..model.joints import JointFreeFlyer, JointPrismatic, JointRevolute, JointRevoluteUnbounded, JointSpherical
from ..model.model import Model
from ..model.topology import build_supports

#: kind -> offset of ``[qx, qy, qz, qw]`` inside the joint's configuration block.
_QUATERNION = {JointSpherical: 0, JointFreeFlyer: 3}

# vec(R(q)) = vec(q qᵀ) · C / |q|² for q = [x, y, z, w]; entry (a, b, k) adds k · q_a q_b to R[row].
_QUATERNION_TERMS = {
    0: [(0, 0, 1), (1, 1, -1), (2, 2, -1), (3, 3, 1)],
    1: [(0, 1, 2), (2, 3, -2)],
    2: [(0, 2, 2), (1, 3, 2)],
    3: [(0, 1, 2), (2, 3, 2)],
    4: [(0, 0, -1), (1, 1, 1), (2, 2, -1), (3, 3, 1)],
    5: [(1, 2, 2), (0, 3, -2)],
    6: [(0, 2, 2), (1, 3, -2)],
    7: [(1, 2, 2), (0, 3, 2)],
    8: [(0, 0, -1), (1, 1, -1), (2, 2, 1), (3, 3, 1)],
}


@final
@dataclass(frozen=True)
class FkPlan:
    """Constants of the FK kernel for one topology, dtype and device. Joints are stored in level order."""

    njoints: int
    angle_columns: torch.Tensor  #: ``(n_rev,)`` columns of ``q_full`` holding revolute angles.
    circle_columns: torch.Tensor  #: ``(n_unb, 2)`` columns holding ``(cos, sin)`` of unbounded revolutes.
    quaternion_columns: torch.Tensor  #: ``(n_quat, 4)`` columns holding ``[qx, qy, qz, qw]``.
    linear_columns: torch.Tensor  #: ``(n_lin,)`` columns that enter translations linearly.
    feature_index: torch.Tensor  #: ``(njoints · F,)`` feature read by each joint slot; padding reads feature 0.
    weight: torch.Tensor  #: ``(njoints, F, 4, 4)`` per-joint feature -> ``4x4`` map; zero at padding.
    bias: torch.Tensor  #: ``(njoints, 4, 4)``
    order: torch.Tensor  #: ``(njoints,)`` joint id at each level-order slot.
    inverse_order: torch.Tensor | None  #: slot of each joint id; ``None`` when the order is the identity.
    level_slices: tuple[tuple[int, int], ...]  #: slot range of each tree level.
    parent_position: tuple[
        torch.Tensor | None, ...
    ]  #: per level > 0: index of each joint's parent in the previous level.
    fallbacks: tuple[
        tuple[torch.Tensor, int, int, int], ...
    ]  #: ``(slot (1,), joint id, q start, q width)`` of rare kinds.
    frame_parents: torch.Tensor  #: ``(nframes,)`` long: parent joint of each frame.


@final
@dataclass(frozen=True)
class JacobianPlan:
    """Static, device-resident schedule of the shared Jacobian column table."""

    max_nv: int  #: widest per-joint subspace.
    column_owner: torch.Tensor  #: ``(nv_full,)`` long: joint owning each full-width v-column.
    column_subspace: torch.Tensor  #: ``(nv_full, 6)``: the constant joint-frame motion subspace column of each column.
    support_mask: torch.Tensor  #: ``(njoints, 1, nv_full)`` bool: True where ``column_owner`` supports the row joint.


def _tree_levels(model: Model) -> list[list[int]]:
    """Joint ids grouped by depth in the tree; parents come before children."""
    parents = model.parent_joint_ids
    depth = [0] * model.njoints
    for joint in range(model.njoints):  # parents[i] < i, so index order is a topological order
        depth[joint] = 0 if parents[joint] < 0 else depth[parents[joint]] + 1
    return [[j for j in range(model.njoints) if depth[j] == d] for d in range(max(depth) + 1)]


def _feature_map(
    axes: torch.Tensor, slot: dict[int, int], rotary: list, quaternions: list, linear: list
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-joint map from features to the joint transform.

    Returns ``feature_index (nj, F)``, ``weight (nj, F, 4, 4)`` and ``bias (nj, 4, 4)`` in level
    order, where ``F`` is the largest number of features any joint reads.
    """
    nj, n_rot = len(slot), len(rotary)
    entries: list[list[tuple[int, torch.Tensor]]] = [[] for _ in range(nj)]  # per slot: (feature, 4x4 weight)
    bias = torch.eye(4, dtype=torch.float64).repeat(nj, 1, 1)

    def block(rotation: torch.Tensor | None = None, translation: torch.Tensor | None = None) -> torch.Tensor:
        out = torch.zeros(4, 4, dtype=torch.float64)
        if rotation is not None:
            out[:3, :3] = rotation
        if translation is not None:
            out[:3, 3] = translation
        return out

    for index, (joint, _) in enumerate(rotary):  # R = (I + K²) + sin · K - cos · K²
        K = hat_so3_raw(axes[joint] / axes[joint].norm())  # unit within 1e-6 (check_axis); exact here, in float64
        bias[slot[joint], :3, :3] += K @ K
        entries[slot[joint]] += [(index, block(K)), (n_rot + index, block(-K @ K))]
    base = 2 * n_rot
    products = torch.zeros(16, 9, dtype=torch.float64)
    for entry, terms in _QUATERNION_TERMS.items():
        for a, b, factor in terms:
            products[a * 4 + b, entry] += factor
    for index, (joint, _) in enumerate(quaternions):
        bias[slot[joint], :3, :3] = 0.0
        entries[slot[joint]] += [(base + 16 * index + k, block(products[k].view(3, 3))) for k in range(16)]
    base += 16 * len(quaternions)
    for index, (joint, _, direction) in enumerate(linear):
        entries[slot[joint]].append((base + index, block(translation=direction)))
    width = max(1, *(len(joint_entries) for joint_entries in entries))
    feature_index = torch.zeros(nj, width, dtype=torch.long)
    weight = torch.zeros(nj, width, 4, 4, dtype=torch.float64)
    for position, joint_entries in enumerate(entries):
        for column, (feature, matrix) in enumerate(joint_entries):
            feature_index[position, column] = feature
            weight[position, column] = matrix
    return feature_index, weight, bias


def build_fk_plan(model: Model, dtype: torch.dtype, device: torch.device) -> FkPlan:
    """Derive the FK constants from the static topology and the joint axes."""
    nj, parents = model.njoints, model.parent_joint_ids
    starts, widths = model.idx_qs_full, model.nqs_full
    levels = _tree_levels(model)
    order = [j for level in levels for j in level]
    slot = {joint: index for index, joint in enumerate(order)}
    axes = torch.tensor([joint.axis or (0.0, 0.0, 0.0) for joint in model.joints], dtype=torch.float64).reshape(nj, 3)
    unit = torch.eye(3, dtype=torch.float64)

    angles, circles, quaternions, fallbacks = [], [], [], []
    linear: list[tuple[int, int, torch.Tensor]] = []  # (joint, column, direction)
    for joint, model_joint in enumerate(model.joints):
        kind, start = type(model_joint), starts[joint]
        if model_joint.nq == 0:
            continue
        if kind is JointRevolute:
            angles.append((joint, start))
        elif kind is JointRevoluteUnbounded:
            circles.append((joint, start))
        elif kind is JointPrismatic:
            linear.append((joint, start, axes[joint]))
        elif kind in _QUATERNION:
            quaternions.append((joint, start + _QUATERNION[kind]))
            if _QUATERNION[kind]:
                linear += [(joint, start + k, unit[k]) for k in range(3)]
        else:
            fallbacks.append((torch.tensor([slot[joint]], device=device), joint, start, widths[joint]))

    # Feature layout: [sin | cos] of revolute then unbounded joints, quaternion products, linear coordinates.
    feature_index, weight, bias = _feature_map(axes, slot, angles + circles, quaternions, linear)

    def columns(values: list, shape: tuple[int, ...]) -> torch.Tensor:
        return torch.tensor(values, dtype=torch.long, device=device).reshape(shape)

    level_slices, parent_position = [], []
    start = 0
    for index, level in enumerate(levels):
        level_slices.append((start, start + len(level)))
        start += len(level)
        if index:
            previous = {joint: position for position, joint in enumerate(levels[index - 1])}
            # A single parent broadcasts; no gather needed.
            parent_position.append(
                None if len(previous) == 1 else columns([previous[parents[j]] for j in level], (-1,))
            )
    identity = order == list(range(nj))
    order_tensor = columns(order, (-1,))
    return FkPlan(
        njoints=nj,
        angle_columns=columns([c for _, c in angles], (-1,)),
        circle_columns=columns([[c, c + 1] for _, c in circles], (-1, 2)),
        quaternion_columns=columns([[c + k for k in range(4)] for _, c in quaternions], (-1, 4)),
        linear_columns=columns([c for _, c, _ in linear], (-1,)),
        feature_index=feature_index.reshape(-1).to(device),
        weight=weight.to(dtype=dtype, device=device),
        bias=bias.to(dtype=dtype, device=device),
        order=order_tensor,
        inverse_order=None if identity else torch.argsort(order_tensor),
        level_slices=tuple(level_slices),
        parent_position=tuple(parent_position),
        fallbacks=tuple(fallbacks),
        frame_parents=torch.tensor(model.frame_parent_joint_ids, dtype=torch.long, device=device),
    )


def build_jacobian_plan(model: Model, dtype: torch.dtype, device: torch.device) -> JacobianPlan:
    """Derive the column schedule from the topology and the joints' neutral motion subspaces."""
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
    for joint_id, chain in enumerate(build_supports(model.parent_joint_ids)):
        is_support[joint_id, list(chain)] = True
    subspaces = torch.zeros((njoints, 6, max_nv), dtype=dtype, device=device)
    for joint_id, joint in enumerate(model.joints):
        if joint.nv:
            neutral = joint.neutral().to(device=device, dtype=dtype)
            subspaces[joint_id, :, : joint.nv] = joint.motion_subspace(neutral).to(device=device, dtype=dtype)
    column_owner = torch.tensor(owner, dtype=torch.long, device=device)
    column_gather = torch.tensor(gather, dtype=torch.long, device=device)
    return JacobianPlan(
        max_nv=max_nv,
        column_owner=column_owner,
        column_subspace=subspaces.transpose(-2, -1).reshape(-1, 6).index_select(0, column_gather).contiguous(),
        support_mask=is_support.index_select(-1, column_owner).unsqueeze(-2).contiguous(),
    )


def fk_plan(model: Model, *, dtype: torch.dtype, device: torch.device) -> FkPlan:
    """The model's `FkPlan` in ``dtype`` on ``device`` (built once per model, dtype and device)."""
    return model._plan("fk", build_fk_plan, dtype=dtype, device=device)


def jacobian_plan(model: Model, *, dtype: torch.dtype, device: torch.device) -> JacobianPlan:
    """The model's `JacobianPlan` in ``dtype`` on ``device`` (built once per model, dtype and device)."""
    return model._plan("jacobian", build_jacobian_plan, dtype=dtype, device=device)


__all__ = ["FkPlan", "JacobianPlan", "build_fk_plan", "build_jacobian_plan", "fk_plan", "jacobian_plan"]
