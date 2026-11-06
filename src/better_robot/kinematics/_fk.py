"""Forward kinematics kernel: one affine map plus one matmul per tree level.

Every common joint transform is affine in a few features of ``q``: ``sin`` and ``cos`` of a revolute angle, the
products ``q_a q_b`` of a unit quaternion, and the raw value of a translation coordinate. So all joint transforms
of a model come from one batched product of each joint's own features with a constant per-joint matrix that
depends only on the topology and the joint axes (`_plans.FkPlan`, built once per model, dtype and device; cost
linear in the number of joints). Placements, which stay differentiable and may be batched, are folded into that
matrix when they carry no batch (two small products independent of the batch size) or applied with one matmul,
and the world chain needs one matmul per tree level. Transforms are ``[R | p]`` matrices throughout; nothing is
converted to quaternions here. Rare joint kinds (planar, translation, helical, custom) are evaluated one by one
through their own ``placement`` and written into the same tensor. Everything stays autograd- and ``torch.func``-safe
(jacrev and jacfwd); no host syncs; the public wrapper traces as one graph under ``torch.compile(fullgraph=True)``
(``tests/kinematics/test_fk_compile.py``).

The ``(njoints, 7)`` and ``(nframes, 7)`` placements are converted to ``4x4`` matrices by `placement_matrices` and
memoised on the model by grad mode (record 0040): converting per call cost 0.17 ms, 44 % of a batch-1 Panda FK
call on CPU (0.19 -> 0.39 ms; one thread, float32, torch 2.13, AMD EPYC 9654) and under 1.2 % at batch 4096. With
the memo batch 1 is back at 0.18 ms. Placements that carry a graph are converted per call so that each
differentiable call owns its graph.
"""

from __future__ import annotations

import torch

from ..lie import se3
from ..model.model import Model
from ._plans import fk_plan


def placement_matrices(model: Model) -> tuple[torch.Tensor, torch.Tensor]:
    """Joint and frame placements as ``4x4`` matrices, ``(I..., njoints, 4, 4)`` and ``(I..., nframes, 4, 4)``.

    Memoised on the model by grad mode (record 0040); placements that carry a graph are converted per call.
    """
    joints = model._value("joint_matrices", lambda m: se3.to_matrix(m.joint_placements))
    frames = model._value("frame_matrices", lambda m: se3.to_matrix(m.frame_placements))
    return joints, frames


def world_transforms(
    model: Model, q_full: torch.Tensor, placements: torch.Tensor, frames: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """World ``[R | p]`` of every joint and frame, ``(N, njoints, 3, 4)`` and ``(N, nframes, 3, 4)``, in joint order.

    ``q_full: (N, nq_full)`` is the mimic-expanded configuration; ``placements`` and ``frames`` are the placement
    matrices ``(njoints, 4, 4)`` and ``(nframes, 4, 4)``, or ``(N, njoints, 4, 4)`` and ``(N, nframes, 4, 4)`` when
    the model's values are batched. Trusts validated inputs.
    """
    plan = fk_plan(model, dtype=placements.dtype, device=placements.device)
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
        placements = placements.index_select(-3, plan.order)
    if features:
        gathered = (
            torch.cat(features, dim=-1)
            .index_select(1, plan.feature_index)
            .view(count, nj, plan.feature_index.numel() // nj)
        )
    else:
        gathered = plan.bias.new_zeros(count, nj, 1)
    if placements.ndim == 3 and not plan.fallbacks:
        # The joint transform is affine in the features, so constant-per-batch placements fold into
        # the map: two small products independent of the batch size instead of one N-sized matmul.
        weight = placements.unsqueeze(1) @ plan.weight
        local = (placements @ plan.bias) + torch.einsum("njf,jfab->njab", gathered, weight)
    else:
        joint = plan.bias + torch.einsum("njf,jfab->njab", gathered, plan.weight)
        for slot, joint_id, start, width in plan.fallbacks:
            pose = model.joints[joint_id].placement(q_full[:, start : start + width])
            joint = joint.index_copy(1, slot, se3.to_matrix(pose).unsqueeze(1))
        local = placements @ joint  # (N, nj, 4, 4), level order
    first_start, first_stop = plan.level_slices[0]
    levels = [local[:, first_start:first_stop, :3]]
    for (start, stop), position in zip(plan.level_slices[1:], plan.parent_position):
        parent = levels[-1] if position is None else levels[-1].index_select(1, position)
        levels.append(parent @ local[:, start:stop])
    world = torch.cat(levels, dim=1)
    if plan.inverse_order is not None:
        world = world.index_select(1, plan.inverse_order)
    parents = world.index_select(1, plan.frame_parents)
    # einsum is several times faster than a broadcast matmul on CPU when one side has no batch axis.
    frame_world = torch.einsum("nfab,fbc->nfac", parents, frames) if frames.ndim == 3 else parents @ frames
    return world, frame_world


__all__ = ["placement_matrices", "world_transforms"]
