"""Matrix-lane forward kinematics — the batched Torch FK implementation.

:func:`~better_robot.kinematics.forward.forward_kinematics_raw` delegates its
per-joint work here. The strategy trades the legacy per-joint quaternion
composition (~78 kernel launches per joint) for three batched stages:

1.  Group joints by kind and evaluate every kind's local transform for all of
    its joints in one batched call, as rotation matrices scattered into a
    per-joint ``(B..., njoints, 3, 3)`` / ``(B..., njoints, 3)`` pair.
2.  Chain world placements with one batched ``4x4`` matmul per joint over the
    static topological order — the single irreducible sequential dependency.
3.  Convert the world and parent-frame rotations to scalar-last quaternions in
    one batched matrix-to-quaternion pass.

The kind-grouping plan is a pure function of :class:`ModelStructure` and is
memoised on the structure instance, so repeated evaluations skip the Python
rebuild. Everything stays autograd- and ``torch.func``-safe: no in-place writes
on traced tensors, out-of-place :meth:`~torch.Tensor.index_copy`.

The matrix-to-quaternion stage emits canonical-sign quaternions (Shepperd
selection), so a world/local quaternion may differ in overall sign (``q`` vs
``-q``) from the legacy quaternion-composition lane; both represent the same
rotation. Callers that compare raw quaternions across lanes must sign-align.

See ``docs/concepts/kinematics_and_jacobians.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..data_model.execution_batch import broadcast_to_execution_batch
from ..data_model.joint_dispatch import joint_transform
from ..data_model.joint_models.base import JointModel
from ..data_model.model_structure import JOINT_KIND_CODES, ModelStructure
from ..data_model.model_values import ModelValues
from ..data_model.reduced_coordinates import expand_configuration
from ..lie import se3, so3


#: Joint-kind code -> batched-family name. Kinds absent here (planar,
#: translation, helical, composite) use the per-joint fallback path.
_FAMILY_BY_CODE: dict[int, str] = {
    JOINT_KIND_CODES["universe"]: "fixed",
    JOINT_KIND_CODES["fixed"]: "fixed",
    JOINT_KIND_CODES["mimic"]: "fixed",
    JOINT_KIND_CODES["revolute_rx"]: "revolute",
    JOINT_KIND_CODES["revolute_ry"]: "revolute",
    JOINT_KIND_CODES["revolute_rz"]: "revolute",
    JOINT_KIND_CODES["revolute_unaligned"]: "revolute",
    JOINT_KIND_CODES["revolute_unbounded"]: "unbounded",
    JOINT_KIND_CODES["prismatic_px"]: "prismatic",
    JOINT_KIND_CODES["prismatic_py"]: "prismatic",
    JOINT_KIND_CODES["prismatic_pz"]: "prismatic",
    JOINT_KIND_CODES["prismatic_unaligned"]: "prismatic",
    JOINT_KIND_CODES["spherical"]: "spherical",
    JOINT_KIND_CODES["free_flyer"]: "free_flyer",
}

#: Families whose local rotation is parameterised by a per-joint axis.
_AXIS_FAMILIES = frozenset({"revolute", "unbounded", "prismatic"})


@dataclass(frozen=True)
class _KindGroup:
    """One batched joint-kind bucket resolved from static topology."""

    family: str  #: batched-family name (see :data:`_FAMILY_BY_CODE`).
    ids: torch.Tensor  #: ``(n,)`` long joint ids in this group.
    q_index: torch.Tensor | None  #: ``(n, width)`` long gather columns into the full configuration; ``None`` for fixed.
    axes: torch.Tensor | None  #: ``(n, 3)`` per-joint axis for axis families, else ``None``.


@dataclass(frozen=True)
class _FallbackJoint:
    """One rare-kind joint evaluated per-joint through the shared dispatch."""

    joint_id: torch.Tensor  #: ``(1,)`` long id for scatter.
    q_start: int  #: first full-configuration column.
    q_width: int  #: number of full-configuration columns.
    joint_model: JointModel  #: concrete joint model for dispatch.
    kind_code: int  #: joint-kind code for dispatch.
    axis: torch.Tensor  #: packed joint axis.
    pitch: torch.Tensor  #: packed joint pitch (helical).


@dataclass(frozen=True)
class MatrixFKPlan:
    """Static, device-resident schedule for the matrix FK lane."""

    njoints: int  #: number of joints (including the universe).
    groups: tuple[_KindGroup, ...]  #: batched kind buckets.
    fallbacks: tuple[_FallbackJoint, ...]  #: rare-kind per-joint jobs.
    topo_order: tuple[int, ...]  #: parents-before-children joint order.
    parents: tuple[int, ...]  #: parent joint id per joint (``-1`` for roots).


def build_plan(structure: ModelStructure) -> MatrixFKPlan:
    """Derive the kind-grouping plan from static topology.

    Pure function of ``structure``; the result is device-resident and reusable
    for every configuration. Prefer :func:`get_plan`, which memoises this.
    """

    device = structure.joint_axes.device
    codes = structure.joint_kind_codes
    idx = structure.idx_qs_full
    widths = structure.nqs_full

    buckets: dict[str, list[int]] = {}
    fallback_ids: list[int] = []
    for joint_id in range(structure.njoints):
        family = _FAMILY_BY_CODE.get(codes[joint_id])
        if family is None:
            fallback_ids.append(joint_id)
        else:
            buckets.setdefault(family, []).append(joint_id)

    groups: list[_KindGroup] = []
    for family, ids in buckets.items():
        id_tensor = torch.tensor(ids, dtype=torch.long, device=device)
        if family == "fixed":
            groups.append(_KindGroup(family, id_tensor, None, None))
            continue
        width = widths[ids[0]]
        columns = torch.tensor(
            [list(range(idx[j], idx[j] + width)) for j in ids],
            dtype=torch.long,
            device=device,
        )
        axes = structure.joint_axes.index_select(0, id_tensor) if family in _AXIS_FAMILIES else None
        groups.append(_KindGroup(family, id_tensor, columns, axes))

    fallbacks = tuple(
        _FallbackJoint(
            joint_id=torch.tensor([j], dtype=torch.long, device=device),
            q_start=idx[j],
            q_width=widths[j],
            joint_model=structure.joint_models[j],
            kind_code=codes[j],
            axis=structure.joint_axes[j],
            pitch=structure.joint_pitches[j],
        )
        for j in fallback_ids
    )
    return MatrixFKPlan(
        njoints=structure.njoints,
        groups=tuple(groups),
        fallbacks=fallbacks,
        topo_order=structure.topo_order,
        parents=structure.parents,
    )


def get_plan(structure: ModelStructure) -> MatrixFKPlan:
    """Return the memoised kind-grouping plan for ``structure``.

    The plan is cached as a private attribute on the (otherwise immutable)
    structure. :meth:`ModelStructure.to` builds a fresh instance, so a moved
    model naturally rebuilds its plan on the new device/dtype.
    """

    plan = getattr(structure, "_fk_matrix_plan", None)
    if plan is None:
        plan = build_plan(structure)
        object.__setattr__(structure, "_fk_matrix_plan", plan)
    return plan


def _local_matrices(
    plan: MatrixFKPlan,
    q_full: torch.Tensor,
    placements: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-joint local rotation ``(B..., nj, 3, 3)`` and translation ``(B..., nj, 3)``.

    These are the *joint* transforms only; the parent placement is composed in
    :func:`matrix_core`.
    """

    device = placements.device
    dtype = placements.dtype
    batch_shape = tuple(placements.shape[:-2])
    nj = plan.njoints

    eye3 = torch.eye(3, device=device, dtype=dtype)
    rotations = eye3.expand(*batch_shape, nj, 3, 3).contiguous()
    translations = torch.zeros(*batch_shape, nj, 3, device=device, dtype=dtype)

    for group in plan.groups:
        if group.family == "fixed":
            continue
        q_group = q_full[..., group.q_index]  # (B..., n, width)
        group_rotation: torch.Tensor | None = None
        group_translation: torch.Tensor | None = None
        if group.family == "spherical":
            group_rotation = so3.to_matrix(so3.normalize(q_group))
        elif group.family == "free_flyer":
            group_translation = q_group[..., :3]
            group_rotation = so3.to_matrix(so3.normalize(q_group[..., 3:7]))
        elif group.family == "revolute":
            group_rotation = so3.to_matrix(so3.from_axis_angle(group.axes, q_group[..., 0]))
        elif group.family == "unbounded":
            angle = torch.atan2(q_group[..., 1], q_group[..., 0])
            group_rotation = so3.to_matrix(so3.from_axis_angle(group.axes, angle))
        else:  # prismatic
            group_translation = q_group[..., 0:1] * group.axes
        if group_rotation is not None:
            rotations = rotations.index_copy(-3, group.ids, group_rotation)
        if group_translation is not None:
            translations = translations.index_copy(-2, group.ids, group_translation)

    for fallback in plan.fallbacks:
        q_joint = q_full[..., fallback.q_start : fallback.q_start + fallback.q_width]
        transform = joint_transform(
            fallback.joint_model,
            fallback.kind_code,
            fallback.axis,
            fallback.pitch,
            q_joint,
        )
        rotation = so3.to_matrix(transform[..., 3:7]).unsqueeze(-3)  # (B..., 1, 3, 3)
        translation = transform[..., :3].unsqueeze(-2)  # (B..., 1, 3)
        rotations = rotations.index_copy(-3, fallback.joint_id, rotation)
        translations = translations.index_copy(-2, fallback.joint_id, translation)

    return rotations, translations


def matrix_core(
    plan: MatrixFKPlan,
    q_full: torch.Tensor,
    placements: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the batched matrix FK on prepared inputs.

    :param q_full: full (mimic-expanded) configuration ``(B..., nq_full)``.
    :param placements: parent-frame joint placements ``(B..., njoints, 7)``.
    :returns: ``(world, local)`` joint placements, each ``(B..., njoints, 7)``
        in ``[tx, ty, tz, qx, qy, qz, qw]`` layout.
    """

    batch_shape = tuple(placements.shape[:-2])
    nj = plan.njoints
    stack_dim = len(batch_shape)

    rotations, translations = _local_matrices(plan, q_full, placements)

    # Assemble the joint transform as a homogeneous 4x4 and compose the parent
    # placement onto it in one batched matmul over all joints.
    top = torch.cat([rotations, translations.unsqueeze(-1)], dim=-1)  # (B..., nj, 3, 4)
    zeros_row = placements.new_zeros(*batch_shape, nj, 1, 3)
    ones_col = placements.new_ones(*batch_shape, nj, 1, 1)
    bottom = torch.cat([zeros_row, ones_col], dim=-1)  # (B..., nj, 1, 4)
    joint_mat = torch.cat([top, bottom], dim=-2)  # (B..., nj, 4, 4)
    local_mat = se3.to_matrix(placements) @ joint_mat  # (B..., nj, 4, 4)

    # World chain: one batched 4x4 matmul per joint along the topological order.
    world: list[torch.Tensor] = [torch.empty(0)] * nj
    for j in plan.topo_order:
        parent = plan.parents[j]
        m_j = local_mat[..., j, :, :]
        world[j] = m_j if parent < 0 else world[parent] @ m_j
    world_mat = torch.stack(world, dim=stack_dim)  # (B..., nj, 4, 4)

    # Single batched matrix -> quaternion for world and local rotations.
    world_translation = world_mat[..., :3, 3]
    local_translation = local_mat[..., :3, 3]
    both_rotation = torch.cat([world_mat[..., :3, :3], local_mat[..., :3, :3]], dim=-3)
    both_quaternion = so3.from_matrix(both_rotation)
    world_quaternion = both_quaternion[..., :nj, :]
    local_quaternion = both_quaternion[..., nj:, :]

    world = torch.cat([world_translation, world_quaternion], dim=-1)
    local = torch.cat([local_translation, local_quaternion], dim=-1)
    return world, local


def prepare_inputs(
    structure: ModelStructure,
    values: ModelValues,
    q: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Broadcast and mimic-expand FK inputs to the execution batch.

    :returns: ``(q_full, placements)`` broadcast to the shared execution batch.
    """

    batch_shape = values._execution_batch_shape(q)
    q = broadcast_to_execution_batch(q, batch_shape, (structure.nq,), name="q")
    q_full = expand_configuration(structure, q)
    placements = broadcast_to_execution_batch(
        values.joint_placements,
        batch_shape,
        (structure.njoints, 7),
        name="joint_placements",
    )
    return q_full, placements


def matrix_forward(
    structure: ModelStructure,
    values: ModelValues,
    q: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Eager matrix FK. Returns ``(world, local)`` placements ``(B..., njoints, 7)``."""

    q_full, placements = prepare_inputs(structure, values, q)
    return matrix_core(get_plan(structure), q_full, placements)


def _compiled_core(structure: ModelStructure):
    """Return the ``torch.compile``d matrix core cached on ``structure``.

    The core closes over the structure's kind-grouping plan and takes only the
    two differentiable tensors, so ``dynamic=True`` lets the batch size vary
    without a recompile. The multi-second first-call compilation is paid once
    per structure (and per dtype/device, since :meth:`ModelStructure.to` mints a
    fresh instance).
    """

    core = getattr(structure, "_fk_matrix_compiled_core", None)
    if core is None:
        plan = get_plan(structure)

        def run(q_full: torch.Tensor, placements: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return matrix_core(plan, q_full, placements)

        core = torch.compile(run, dynamic=True)
        object.__setattr__(structure, "_fk_matrix_compiled_core", core)
    return core


def matrix_forward_compiled(
    structure: ModelStructure,
    values: ModelValues,
    q: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compiled matrix FK. Returns ``(world, local)`` placements ``(B..., njoints, 7)``."""

    q_full, placements = prepare_inputs(structure, values, q)
    return _compiled_core(structure)(q_full, placements)


__all__ = [
    "MatrixFKPlan",
    "build_plan",
    "get_plan",
    "matrix_core",
    "matrix_forward",
    "matrix_forward_compiled",
    "prepare_inputs",
]
