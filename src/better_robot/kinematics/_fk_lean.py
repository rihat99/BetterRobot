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

from ..data_model.joint_dispatch import joint_transform
from ..data_model.model_structure import JOINT_KIND_CODES, ModelStructure
from ..data_model.model_values import ModelValues
from ..lie import se3

_K = JOINT_KIND_CODES
_FIXED = {_K["universe"], _K["fixed"], _K["mimic"]}
_REVOLUTE = {_K["revolute_rx"], _K["revolute_ry"], _K["revolute_rz"], _K["revolute_unaligned"]}
_UNBOUNDED = {_K["revolute_unbounded"]}
_PRISMATIC = {_K["prismatic_px"], _K["prismatic_py"], _K["prismatic_pz"], _K["prismatic_unaligned"]}
#: kind code -> offset of ``[qx, qy, qz, qw]`` inside the joint's configuration block.
_QUATERNION = {_K["spherical"]: 0, _K["free_flyer"]: 3}

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
    parent_position: tuple[torch.Tensor | None, ...]  #: per level > 0: index of each joint's parent in the previous level.
    fallbacks: tuple[tuple[torch.Tensor, int, int, int], ...]  #: ``(slot (1,), joint id, q start, q width)`` of rare kinds.


def _tree_levels(structure: ModelStructure) -> list[list[int]]:
    """Joint ids grouped by depth in the tree; parents come before children."""
    parents = structure.parents
    depth = [0] * structure.njoints
    for joint in structure.topo_order:
        depth[joint] = 0 if parents[joint] < 0 else depth[parents[joint]] + 1
    return [[j for j in range(structure.njoints) if depth[j] == d] for d in range(max(depth) + 1)]


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


def build_plan(structure: ModelStructure, dtype: torch.dtype, device: torch.device) -> LeanPlan:
    """Derive the constants from static topology and joint axes."""
    nj, parents = structure.njoints, structure.parents
    codes, starts, widths = structure.joint_kind_codes, structure.idx_qs_full, structure.nqs_full
    levels = _tree_levels(structure)
    order = [j for level in levels for j in level]
    slot = {joint: index for index, joint in enumerate(order)}
    axes = structure.joint_axes.detach().to(dtype=torch.float64, device="cpu")
    unit = torch.eye(3, dtype=torch.float64)

    angles, circles, quaternions, fallbacks = [], [], [], []
    linear: list[tuple[int, int, torch.Tensor]] = []  # (joint, column, direction)
    for joint in range(nj):
        code, start = codes[joint], starts[joint]
        if code in _FIXED:
            continue
        if code in _REVOLUTE:
            angles.append((joint, start))
        elif code in _UNBOUNDED:
            circles.append((joint, start))
        elif code in _PRISMATIC:
            linear.append((joint, start, axes[joint]))
        elif code in _QUATERNION:
            quaternions.append((joint, start + _QUATERNION[code]))
            if _QUATERNION[code]:
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
            parent_position.append(None if len(previous) == 1 else columns([previous[parents[j]] for j in level], (-1,)))
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
    )


def get_plan(structure: ModelStructure, dtype: torch.dtype, device: torch.device) -> LeanPlan:
    """Memoised :func:`build_plan`; the cache lives on the structure instance."""
    cache = getattr(structure, "_fk_lean_plans", None)
    if cache is None:
        cache = {}
        object.__setattr__(structure, "_fk_lean_plans", cache)
    key = (dtype, device)
    if key not in cache:
        with torch.inference_mode(False):  # inference tensors in the plan would break later autograd calls
            cache[key] = build_plan(structure, dtype, device)
    return cache[key]


def placement_matrices(values: ModelValues) -> tuple[torch.Tensor, torch.Tensor]:
    """Joint and frame placements as ``4x4`` matrices, ``(..., njoints, 4, 4)`` and ``(..., nframes, 4, 4)``.

    Constant placements are converted once and kept on the values object; an in-place edit of
    the tensors invalidates the copy. Placements that require grad, or that are being
    transformed by ``torch.func``, or calls in inference mode, convert on every call so each
    call owns its graph.
    """
    joints, frames = values.joint_placements, values.frame_placements
    wrapped = torch._C._functorch.is_functorch_wrapped_tensor
    if joints.requires_grad or frames.requires_grad or wrapped(joints) or wrapped(frames):
        return se3.to_matrix(joints), se3.to_matrix(frames)
    if torch.is_inference_mode_enabled() or joints.is_inference() or frames.is_inference():
        return se3.to_matrix(joints), se3.to_matrix(frames)
    versions = (joints._version, frames._version)
    cached = getattr(values, "_fk_placement_matrices", None)
    if cached is None or cached[0] != versions:
        cached = (versions, se3.to_matrix(joints), se3.to_matrix(frames))
        object.__setattr__(values, "_fk_placement_matrices", cached)
    return cached[1], cached[2]


def lean_core(
    structure: ModelStructure, q_full: torch.Tensor, placement: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """World and parent-relative joint transforms.

    ``q_full: (N, nq_full)`` is the mimic-expanded configuration; ``placement`` holds the joint
    placements as matrices, ``(njoints, 4, 4)`` or ``(N, njoints, 4, 4)``. Returns ``(world, local)``, each
    ``(N, njoints, 3, 4)`` as ``[R | p]`` in joint order.
    """
    plan = get_plan(structure, placement.dtype, placement.device)
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
        gathered = torch.cat(features, dim=-1).index_select(1, plan.feature_index).view(count, nj, -1)
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
            pose = joint_transform(
                structure.joint_models[joint_id],
                structure.joint_kind_codes[joint_id],
                structure.joint_axes[joint_id],
                structure.joint_pitches[joint_id],
                q_full[:, start : start + width],
            )
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


__all__ = ["LeanPlan", "build_plan", "get_plan", "lean_core", "placement_matrices"]
