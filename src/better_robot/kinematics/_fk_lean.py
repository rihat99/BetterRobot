"""Forward kinematics as one affine map plus one matmul per tree level.

Every common joint transform is affine in a few features of ``q``: ``sin`` and ``cos`` of a
revolute angle, the products ``q_a q_b`` of a unit quaternion, and the raw value of a
translation coordinate. So all joint transforms of a model come from one batched product of
each joint's own features with a constant per-joint matrix that depends only on the topology
and the joint axes (cost linear in the number of joints). Placements (which stay
differentiable and may be batched) are folded into that matrix or applied with one matmul,
and the world chain needs one matmul per tree level. Transforms are ``[R | p]`` matrices throughout; nothing is converted
to quaternions here.

Rare joint kinds (planar, translation, helical, composite) are evaluated one by one through
the shared joint dispatch and written into the same tensor.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..lie import se3
from ..model.joint_dispatch import joint_axes, joint_pitches
from ..model.joint_models.helical import _helical_transform
from ..model.model import Model

_FIXED = {"universe", "fixed", "mimic"}
_REVOLUTE = {"revolute_rx", "revolute_ry", "revolute_rz", "revolute_unaligned"}
_UNBOUNDED = {"revolute_unbounded"}
_PRISMATIC = {"prismatic_px", "prismatic_py", "prismatic_pz", "prismatic_unaligned"}
#: kind -> offset of ``[qx, qy, qz, qw]`` inside the joint's configuration block.
_QUATERNION = {"spherical": 0, "free_flyer": 3}

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


def _hat(v: torch.Tensor) -> torch.Tensor:
    """Skew matrix of ``v: (3,)``."""
    x, y, z = v.tolist()
    return v.new_tensor([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])  # bench-ok: plan construction, once per model


@dataclass(frozen=True)
class LeanPlan:
    """Constants of the lean FK for one topology, dtype and device. Joints are stored in level order."""

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
    axes: torch.Tensor  #: ``(njoints, 3)`` joint axes in the plan's dtype on its device (the helical fallback).
    pitches: torch.Tensor  #: ``(njoints,)`` helical pitches, likewise.


def _tree_levels(model: Model) -> list[list[int]]:
    """Joint ids grouped by depth in the tree; parents come before children."""
    parents = model.parents
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
        K = _hat(axes[joint] / axes[joint].norm())  # Rodrigues needs a unit axis; builders may pass any length
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


def build_plan(model: Model, dtype: torch.dtype, device: torch.device) -> LeanPlan:
    """Derive the constants from the static topology and the joint axes."""
    nj, parents = model.njoints, model.parents
    kinds, starts, widths = [joint.kind for joint in model.joints], model.idx_qs_full, model.nqs_full
    levels = _tree_levels(model)
    order = [j for level in levels for j in level]
    slot = {joint: index for index, joint in enumerate(order)}
    axes = joint_axes(model.joints)
    unit = torch.eye(3, dtype=torch.float64)

    angles, circles, quaternions, fallbacks = [], [], [], []
    linear: list[tuple[int, int, torch.Tensor]] = []  # (joint, column, direction)
    for joint in range(nj):
        kind, start = kinds[joint], starts[joint]
        if kind in _FIXED:
            continue
        if kind in _REVOLUTE:
            angles.append((joint, start))
        elif kind in _UNBOUNDED:
            circles.append((joint, start))
        elif kind in _PRISMATIC:
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
    return LeanPlan(
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
        axes=axes.to(dtype=dtype, device=device),
        pitches=joint_pitches(model.joints).to(dtype=dtype, device=device),
    )


def fk_plan(model: Model, *, dtype: torch.dtype, device: torch.device) -> LeanPlan:
    """The model's `LeanPlan` in ``dtype`` on ``device`` (built once per model, dtype and device)."""
    return model._plan("fk", build_plan, dtype=dtype, device=device)


def placement_matrices(model: Model) -> tuple[torch.Tensor, torch.Tensor]:
    """Joint and frame placements as ``4x4`` matrices, ``(I..., njoints, 4, 4)`` and ``(I..., nframes, 4, 4)``.

    Design note (record 0011, A3): converted on every call, so each call owns its graph and the model carries
    no cache keyed by tensor versions. Measured at R1 (CPU, one thread, float32, one core, this tree against the
    R0 tree): the two conversions cost 0.17 ms per call, which is 44 % of a batch-1 FK call on the Panda
    (0.19 -> 0.39 ms; G1 0.23 -> 0.44 ms) and under 1.2 % at batch 4096 (Panda 14.3 ms, G1 55 ms, inside the
    baseline spread). That is above the 10 % batch-1 threshold of record 0011, so its option (b), a value-side
    memo on the model keyed by grad mode, is due by its own decision record; until that record lands the
    conversion stays per call.
    """
    return se3.to_matrix(model.joint_placements), se3.to_matrix(model.frame_placements)


def _fallback_pose(model: Model, plan: LeanPlan, joint_id: int, q_slice: torch.Tensor) -> torch.Tensor:
    """Pose ``(N, 7)`` of a rare-kind joint; helical reads the plan's device-resident axis and pitch."""
    if model.joints[joint_id].kind == "helical":
        return _helical_transform(plan.axes[joint_id], plan.pitches[joint_id], q_slice)
    return model.joints[joint_id].joint_placement(q_slice)


def lean_core(model: Model, q_full: torch.Tensor, placement: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """World and parent-relative joint transforms.

    ``q_full: (N, nq_full)`` is the mimic-expanded configuration; ``placement`` holds the joint
    placements as matrices, ``(njoints, 4, 4)`` or ``(N, njoints, 4, 4)``. Returns ``(world, local)``, each
    ``(N, njoints, 3, 4)`` as ``[R | p]`` in joint order.
    """
    plan = fk_plan(model, dtype=placement.dtype, device=placement.device)
    nj, count = plan.njoints, q_full.shape[0]
    features = []
    n_circle = plan.circle_columns.shape[0]
    if plan.angle_columns.numel() or n_circle:
        angle = q_full.index_select(-1, plan.angle_columns)
        sine, cosine = [torch.sin(angle)], [torch.cos(angle)]
        if n_circle:
            pair = q_full[:, plan.circle_columns]  # (N, n, 2) as (cos, sin)
            pair = pair / pair.norm(dim=-1, keepdim=True)
            sine.append(pair[..., 1])
            cosine.append(pair[..., 0])
        features += sine + cosine
    if plan.quaternion_columns.numel():
        quaternion = q_full[:, plan.quaternion_columns]  # (N, n, 4)
        outer = quaternion.unsqueeze(-1) * quaternion.unsqueeze(-2)
        outer = outer / outer.diagonal(dim1=-2, dim2=-1).sum(-1)[..., None, None]
        features.append(outer.flatten(-3))
    if plan.linear_columns.numel():
        features.append(q_full.index_select(-1, plan.linear_columns))
    if plan.inverse_order is not None:
        placement = placement.index_select(-3, plan.order)
    if features:
        gathered = (
            torch.cat(features, dim=-1)
            .index_select(1, plan.feature_index)
            .view(count, nj, plan.feature_index.numel() // nj)
        )
    else:
        gathered = plan.bias.new_zeros(count, nj, 1)
    if placement.ndim == 3 and not plan.fallbacks:
        # The joint transform is affine in the features, so constant-per-batch placements fold into
        # the map: two small products independent of the batch size instead of one N-sized matmul.
        weight = placement.unsqueeze(1) @ plan.weight
        local = (placement @ plan.bias) + torch.einsum("njf,jfab->njab", gathered, weight)
    else:
        joint = plan.bias + torch.einsum("njf,jfab->njab", gathered, plan.weight)
        for slot, joint_id, start, width in plan.fallbacks:
            pose = _fallback_pose(model, plan, joint_id, q_full[:, start : start + width])
            joint = joint.index_copy(1, slot, se3.to_matrix(pose).unsqueeze(1))
        local = placement @ joint  # (N, nj, 4, 4), level order
    first_start, first_stop = plan.level_slices[0]
    levels = [local[:, first_start:first_stop, :3]]
    for (start, stop), position in zip(plan.level_slices[1:], plan.parent_position):
        parent = levels[-1] if position is None else levels[-1].index_select(1, position)
        levels.append(parent @ local[:, start:stop])
    world = torch.cat(levels, dim=1)
    local = local[:, :, :3]
    if plan.inverse_order is not None:
        world = world.index_select(1, plan.inverse_order)
        local = local.index_select(1, plan.inverse_order)
    return world, local


__all__ = ["LeanPlan", "build_plan", "fk_plan", "lean_core", "placement_matrices"]
