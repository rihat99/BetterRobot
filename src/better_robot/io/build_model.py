"""``build_model(ir, root_joint=...)`` — IR → frozen ``Model`` factory.

See ``docs/concepts/parsers_and_ir.md §3`` for the 10 responsibilities.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import torch

from ..data_model.frame import Frame
from ..exceptions import ModelInconsistencyError
from ..data_model.joint_models import (
    JointFixed,
    JointFreeFlyer,
    JointHelical,
    JointMimic,
    JointPlanar,
    JointPrismaticUnaligned,
    JointPX,
    JointPY,
    JointPZ,
    JointRevoluteUnaligned,
    JointRevoluteUnbounded,
    JointRX,
    JointRY,
    JointRZ,
    JointSpherical,
    JointTranslation,
    JointUniverse,
)
from ..data_model.joint_models.base import JointModel
from ..data_model.model import Model
from ..data_model.model_structure import JOINT_KIND_CODES, ModelStructure, _flatten_rows
from ..data_model.model_values import ModelValues
from ..data_model.topology import build_children, build_subtrees, build_supports, topo_sort
from .ir import IRBody, IRJoint, IRModel, IRError

# ──────────────────────────────── constants ──────────────────────────────────

#: Parent body name that signals "connect to the universe (joint 0)".
_WORLD_SENTINEL = "world"

_EPS = 1e-6
_IDENTITY_SE3_VALS = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
_SUPPORTED_MIMIC_JOINTS = (
    JointRX,
    JointRY,
    JointRZ,
    JointRevoluteUnaligned,
    JointPX,
    JointPY,
    JointPZ,
    JointPrismaticUnaligned,
    JointHelical,
)
_EUCLIDEAN_MANIFOLD_TYPES = frozenset(
    {
        JointRX,
        JointRY,
        JointRZ,
        JointRevoluteUnaligned,
        JointPX,
        JointPY,
        JointPZ,
        JointPrismaticUnaligned,
        JointTranslation,
        JointHelical,
    }
)


@dataclass(frozen=True, slots=True)
class _MimicReduction:
    """Named build-time result for reduced mimic coordinates."""

    nqs: tuple[int, ...]
    nvs: tuple[int, ...]
    idx_qs: tuple[int, ...]
    idx_vs: tuple[int, ...]
    nq: int
    nv: int
    q_expansion: torch.Tensor
    q_offset: torch.Tensor
    v_expansion: torch.Tensor
    lower_pos_limit: torch.Tensor
    upper_pos_limit: torch.Tensor
    velocity_limit: torch.Tensor
    effort_limit: torch.Tensor
    rotor_inertia: torch.Tensor
    armature: torch.Tensor
    friction: torch.Tensor
    damping: torch.Tensor


def _check_topology_invariants(
    *,
    parents: tuple[int, ...],
    nqs: tuple[int, ...],
    nvs: tuple[int, ...],
    idx_qs: tuple[int, ...],
    idx_vs: tuple[int, ...],
    nq_total: int,
    nv_total: int,
) -> None:
    """Validate the four topology invariants from ``docs/conventions/contracts.md §1.5``.

    Raises :class:`~better_robot.exceptions.ModelInconsistencyError` at
    build time so the caller never sees a ``Model`` in an inconsistent
    state.
    """
    # 1. Universe joint is rooted at -1.
    if parents[0] != -1:
        raise ModelInconsistencyError(f"parents[0] must be -1 (universe), got {parents[0]}")
    # 2. Topologically sorted: parents[i] < i for i > 0.
    for i, p in enumerate(parents[1:], start=1):
        if p >= i:
            raise ModelInconsistencyError(f"parents[{i}]={p} >= {i}; topological sort violated")
    # 3. sum(nqs) == nq, sum(nvs) == nv.
    if sum(nqs) != nq_total:
        raise ModelInconsistencyError(f"sum(nqs)={sum(nqs)} != nq={nq_total}")
    if sum(nvs) != nv_total:
        raise ModelInconsistencyError(f"sum(nvs)={sum(nvs)} != nv={nv_total}")
    # 4. Contiguous slicing: idx_qs[i] + nqs[i] == idx_qs[i+1] (and same for v).
    for i in range(len(nqs) - 1):
        if idx_qs[i] + nqs[i] != idx_qs[i + 1]:
            raise ModelInconsistencyError(
                f"idx_qs[{i}] + nqs[{i}] = {idx_qs[i] + nqs[i]} != idx_qs[{i + 1}]={idx_qs[i + 1]} (q-slicing gap)"
            )
        if idx_vs[i] + nvs[i] != idx_vs[i + 1]:
            raise ModelInconsistencyError(
                f"idx_vs[{i}] + nvs[{i}] = {idx_vs[i] + nvs[i]} != idx_vs[{i + 1}]={idx_vs[i + 1]} (v-slicing gap)"
            )


# ─────────────────────────────── helpers ─────────────────────────────────��───


def _axis_near(a: torch.Tensor | None, ref: tuple[float, float, float]) -> bool:
    """True if ``a`` is within ``_EPS`` of ``ref``."""
    if a is None:
        return False
    r = torch.tensor(ref, dtype=a.dtype, device=a.device)
    return bool((a - r).norm() < _EPS)


def _cumulative_layout(widths: tuple[int, ...]) -> tuple[tuple[int, ...], int]:
    indices: list[int] = []
    offset = 0
    for width in widths:
        indices.append(offset)
        offset += width
    return tuple(indices), offset


def _matrix_slice_indices(
    joint_ids: tuple[int, ...],
    starts: tuple[int, ...],
    width: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    indices = [starts[joint_id] + offset for joint_id in joint_ids for offset in range(width)]
    return torch.tensor(indices, device=device, dtype=torch.long).reshape(len(joint_ids), width)


def _flat_slice_indices(
    joint_ids: tuple[int, ...],
    starts: tuple[int, ...],
    widths: tuple[int, ...],
    *,
    device: torch.device,
) -> tuple[tuple[int, ...], torch.Tensor]:
    offsets = [0]
    indices: list[int] = []
    for joint_id in joint_ids:
        indices.extend(range(starts[joint_id], starts[joint_id] + widths[joint_id]))
        offsets.append(len(indices))
    return tuple(offsets), torch.tensor(indices, device=device, dtype=torch.long)


def _joint_axis(joint: JointModel) -> tuple[float, float, float]:
    aligned = {
        "revolute_rx": (1.0, 0.0, 0.0),
        "revolute_ry": (0.0, 1.0, 0.0),
        "revolute_rz": (0.0, 0.0, 1.0),
        "prismatic_px": (1.0, 0.0, 0.0),
        "prismatic_py": (0.0, 1.0, 0.0),
        "prismatic_pz": (0.0, 0.0, 1.0),
    }
    if joint.kind in aligned:
        return aligned[joint.kind]
    axis = getattr(joint, "axis", None)
    if axis is None:
        return (0.0, 0.0, 0.0)
    values = axis.detach().to(device="cpu", dtype=torch.float64).tolist()
    return (float(values[0]), float(values[1]), float(values[2]))


def _build_mimic_reduction(  # noqa: PLR0912, PLR0913, PLR0915 - one build-time policy boundary
    *,
    joint_models: tuple[JointModel, ...],
    joint_names: tuple[str, ...],
    mimic_source: tuple[int, ...],
    mimic_multiplier: torch.Tensor,
    mimic_offset: torch.Tensor,
    mimic_targets: set[int],
    nqs_full: tuple[int, ...],
    nvs_full: tuple[int, ...],
    idx_qs_full: tuple[int, ...],
    idx_vs_full: tuple[int, ...],
    lower_full: torch.Tensor,
    upper_full: torch.Tensor,
    velocity_full: torch.Tensor,
    effort_full: torch.Tensor,
    rotor_full: torch.Tensor,
    armature_full: torch.Tensor,
    friction_full: torch.Tensor,
    damping_full: torch.Tensor,
) -> _MimicReduction:
    """Build public layouts, affine maps, and reduced scalar limits."""

    for target in mimic_targets:
        source = mimic_source[target]
        if source == target:
            raise IRError(f"Mimic cycle detected: joint {joint_names[target]!r} references itself")
        for role, joint_id in (("target", target), ("source", source)):
            joint = joint_models[joint_id]
            if not isinstance(joint, _SUPPORTED_MIMIC_JOINTS):
                raise IRError(
                    f"Mimic {role} joint {joint_names[joint_id]!r} has unsupported "
                    f"kind {joint.kind!r}; reduced mimic coordinates currently "
                    "require concrete scalar Euclidean revolute, prismatic, or "
                    "helical joints (nq=nv=1)"
                )

    public_nqs = tuple(0 if joint_id in mimic_targets else width for joint_id, width in enumerate(nqs_full))
    public_nvs = tuple(0 if joint_id in mimic_targets else width for joint_id, width in enumerate(nvs_full))
    public_idx_qs, nq = _cumulative_layout(public_nqs)
    public_idx_vs, nv = _cumulative_layout(public_nvs)
    nq_full = sum(nqs_full)
    nv_full = sum(nvs_full)

    q_expansion = lower_full.new_zeros((nq_full, nq))
    q_offset_full = lower_full.new_zeros((nq_full,))
    v_expansion = velocity_full.new_zeros((nv_full, nv))

    state = [0] * len(joint_models)
    resolved: list[tuple[int, float, float] | None] = [None] * len(joint_models)

    def resolve(joint_id: int, path: tuple[int, ...] = ()) -> tuple[int, float, float]:
        if joint_id not in mimic_targets:
            return joint_id, 1.0, 0.0
        if state[joint_id] == 1:
            cycle = (*path, joint_id)
            names = " -> ".join(joint_names[index] for index in cycle)
            raise IRError(f"Mimic cycle detected: {names}")
        if state[joint_id] == 2:
            result = resolved[joint_id]
            assert result is not None
            return result
        state[joint_id] = 1
        source = mimic_source[joint_id]
        root, source_scale, source_offset = resolve(source, (*path, joint_id))
        multiplier = float(mimic_multiplier[joint_id])
        offset = float(mimic_offset[joint_id])
        result = (
            root,
            multiplier * source_scale,
            multiplier * source_offset + offset,
        )
        resolved[joint_id] = result
        state[joint_id] = 2
        return result

    for joint_id, (nq_joint, nv_joint) in enumerate(zip(nqs_full, nvs_full)):
        full_iq = idx_qs_full[joint_id]
        full_iv = idx_vs_full[joint_id]
        if joint_id in mimic_targets:
            root, scale, offset = resolve(joint_id)
            reduced_iq = public_idx_qs[root]
            reduced_iv = public_idx_vs[root]
            q_expansion[full_iq, reduced_iq] = scale
            q_offset_full[full_iq] = offset
            v_expansion[full_iv, reduced_iv] = scale
            continue
        reduced_iq = public_idx_qs[joint_id]
        reduced_iv = public_idx_vs[joint_id]
        if nq_joint:
            q_expansion[
                full_iq : full_iq + nq_joint,
                reduced_iq : reduced_iq + nq_joint,
            ] = torch.eye(nq_joint, dtype=lower_full.dtype, device=lower_full.device)
        if nv_joint:
            v_expansion[
                full_iv : full_iv + nv_joint,
                reduced_iv : reduced_iv + nv_joint,
            ] = torch.eye(nv_joint, dtype=velocity_full.dtype, device=velocity_full.device)

    lower = lower_full.new_empty((nq,))
    upper = upper_full.new_empty((nq,))
    velocity = velocity_full.new_empty((nv,))
    effort = effort_full.new_empty((nv,))
    rotor = rotor_full.new_empty((nv,))
    armature = armature_full.new_empty((nv,))
    friction = friction_full.new_empty((nv,))
    damping = damping_full.new_empty((nv,))
    for joint_id, (nq_joint, nv_joint) in enumerate(zip(nqs_full, nvs_full)):
        if joint_id in mimic_targets:
            continue
        full_iq = idx_qs_full[joint_id]
        full_iv = idx_vs_full[joint_id]
        reduced_iq = public_idx_qs[joint_id]
        reduced_iv = public_idx_vs[joint_id]
        lower[reduced_iq : reduced_iq + nq_joint] = lower_full[full_iq : full_iq + nq_joint]
        upper[reduced_iq : reduced_iq + nq_joint] = upper_full[full_iq : full_iq + nq_joint]
        velocity[reduced_iv : reduced_iv + nv_joint] = velocity_full[full_iv : full_iv + nv_joint]
        effort[reduced_iv : reduced_iv + nv_joint] = effort_full[full_iv : full_iv + nv_joint]
        rotor[reduced_iv : reduced_iv + nv_joint] = rotor_full[full_iv : full_iv + nv_joint]
        armature[reduced_iv : reduced_iv + nv_joint] = armature_full[full_iv : full_iv + nv_joint]
        friction[reduced_iv : reduced_iv + nv_joint] = friction_full[full_iv : full_iv + nv_joint]
        damping[reduced_iv : reduced_iv + nv_joint] = damping_full[full_iv : full_iv + nv_joint]

    for target in mimic_targets:
        root, scale, offset = resolve(target)
        full_iq = idx_qs_full[target]
        full_iv = idx_vs_full[target]
        reduced_iq = public_idx_qs[root]
        reduced_iv = public_idx_vs[root]
        lo = float(lower_full[full_iq])
        hi = float(upper_full[full_iq])
        if scale == 0.0:
            if offset < lo or offset > hi:
                raise IRError(
                    f"Constant mimic joint {joint_names[target]!r} has offset "
                    f"{offset}, outside its position limits [{lo}, {hi}]"
                )
            continue
        mapped_a = (lo - offset) / scale
        mapped_b = (hi - offset) / scale
        mapped_lo = min(mapped_a, mapped_b)
        mapped_hi = max(mapped_a, mapped_b)
        lower[reduced_iq] = torch.maximum(lower[reduced_iq], lower.new_tensor(mapped_lo))
        upper[reduced_iq] = torch.minimum(upper[reduced_iq], upper.new_tensor(mapped_hi))
        if float(lower[reduced_iq]) > float(upper[reduced_iq]):
            raise IRError(
                f"Mimic limits for source joint {joint_names[root]!r} are empty "
                f"after applying target {joint_names[target]!r}"
            )
        absolute_scale = abs(scale)
        velocity[reduced_iv] = torch.minimum(velocity[reduced_iv], velocity_full[full_iv] / absolute_scale)
        effort[reduced_iv] = effort[reduced_iv] + absolute_scale * effort_full[full_iv]
        rotor[reduced_iv] = rotor[reduced_iv] + scale**2 * rotor_full[full_iv]
        armature[reduced_iv] = armature[reduced_iv] + scale**2 * armature_full[full_iv]
        friction[reduced_iv] = friction[reduced_iv] + absolute_scale * friction_full[full_iv]
        damping[reduced_iv] = damping[reduced_iv] + scale**2 * damping_full[full_iv]

    return _MimicReduction(
        nqs=public_nqs,
        nvs=public_nvs,
        idx_qs=public_idx_qs,
        idx_vs=public_idx_vs,
        nq=nq,
        nv=nv,
        q_expansion=q_expansion,
        q_offset=q_offset_full,
        v_expansion=v_expansion,
        lower_pos_limit=lower,
        upper_pos_limit=upper,
        velocity_limit=velocity,
        effort_limit=effort,
        rotor_inertia=rotor,
        armature=armature,
        friction=friction,
        damping=damping,
    )


def _kind_to_joint_model(ir_joint: IRJoint) -> JointModel:  # noqa: PLR0911, PLR0912
    """Select a concrete ``JointModel`` from an ``IRJoint``."""
    kind = ir_joint.kind
    axis = ir_joint.axis

    joint_model = getattr(ir_joint, "joint_model", None)
    if joint_model is not None:
        if not isinstance(joint_model, JointModel):
            raise IRError(f"joint_model payload for joint {ir_joint.name!r} does not implement the JointModel protocol")
        if joint_model.kind != kind:
            raise IRError(
                f"joint_model payload kind {joint_model.kind!r} does not match "
                f"IR kind {kind!r} for joint {ir_joint.name!r}"
            )
        if isinstance(joint_model, JointMimic):
            raise NotImplementedError(
                f"Joint {ir_joint.name!r} directly selects JointMimic. "
                "The zero-DOF placeholder cannot describe the target motion. "
                "Use a concrete supported scalar revolute, prismatic, or "
                "helical joint model together with mimic_source, "
                "mimic_multiplier, and mimic_offset."
            )
        return joint_model

    if kind == "mimic":
        raise NotImplementedError(
            f"Joint {ir_joint.name!r} uses the zero-DOF mimic kind, which is "
            "missing the target's concrete motion semantics. Use a supported "
            "scalar revolute, prismatic, or helical kind with mimic_source."
        )

    if kind == "universe":
        return JointUniverse()

    if kind in ("fixed", "world"):
        return JointFixed()

    if kind == "revolute_rx":
        return JointRX()
    if kind == "revolute_ry":
        return JointRY()
    if kind == "revolute_rz":
        return JointRZ()
    if kind == "revolute_unaligned":
        if axis is None:
            raise IRError(f"Joint {ir_joint.name!r} with kind {kind!r} requires an axis")
        return JointRevoluteUnaligned(axis=axis.float())
    if kind == "revolute_unbounded":
        _ax = axis if axis is not None else torch.tensor([0.0, 0.0, 1.0])
        return JointRevoluteUnbounded(axis=_ax.float())

    if kind in ("revolute",):
        if axis is None or _axis_near(axis, (1.0, 0.0, 0.0)):
            return JointRX()
        if _axis_near(axis, (0.0, 1.0, 0.0)):
            return JointRY()
        if _axis_near(axis, (0.0, 0.0, 1.0)):
            return JointRZ()
        return JointRevoluteUnaligned(axis=axis.float())

    if kind in ("continuous",):
        _ax = axis if axis is not None else torch.tensor([0.0, 0.0, 1.0])
        return JointRevoluteUnbounded(axis=_ax.float())

    if kind in ("prismatic",):
        if axis is None or _axis_near(axis, (1.0, 0.0, 0.0)):
            return JointPX()
        if _axis_near(axis, (0.0, 1.0, 0.0)):
            return JointPY()
        if _axis_near(axis, (0.0, 0.0, 1.0)):
            return JointPZ()
        return JointPrismaticUnaligned(axis=axis.float())

    if kind == "prismatic_px":
        return JointPX()
    if kind == "prismatic_py":
        return JointPY()
    if kind == "prismatic_pz":
        return JointPZ()
    if kind == "prismatic_unaligned":
        if axis is None:
            raise IRError(f"Joint {ir_joint.name!r} with kind {kind!r} requires an axis")
        return JointPrismaticUnaligned(axis=axis.float())

    if kind in ("spherical", "ball"):
        return JointSpherical()

    if kind in ("free_flyer", "free", "floating"):
        return JointFreeFlyer()

    if kind in ("planar",):
        return JointPlanar()

    if kind in ("translation",):
        return JointTranslation()

    if kind == "helical":
        _ax = axis if axis is not None else torch.tensor([0.0, 0.0, 1.0])
        return JointHelical(axis=_ax.float(), pitch=ir_joint.pitch)

    if kind == "composite":
        raise IRError(
            f"Composite joint {ir_joint.name!r} is missing its programmatic "
            f"JointComposite payload; construct it with "
            f"ModelBuilder.add_joint(kind=JointComposite(...))"
        )

    raise IRError(f"Unknown joint kind {kind!r} for joint {ir_joint.name!r}")


def _pack_inertia(ir_body: IRBody, dtype: torch.dtype) -> torch.Tensor:
    """Pack ``IRBody`` inertia into [m, cx,cy,cz, Ixx,Iyy,Izz,Ixy,Ixz,Iyz]."""
    m = torch.tensor([float(ir_body.mass)], dtype=dtype)
    com = ir_body.com.to(dtype=dtype)
    inertia = ir_body.inertia.to(dtype=dtype)
    sym6 = torch.stack([inertia[0, 0], inertia[1, 1], inertia[2, 2], inertia[0, 1], inertia[0, 2], inertia[1, 2]])
    return torch.cat([m, com, sym6.to(dtype=dtype)])


def _ir_topo_sort(
    ir: IRModel,
    root_body: str,
    ir_indices: list[int],
) -> list[int]:
    """DFS topological sort of IR joint subset (by body traversal).

    Only joints whose indices are in ``ir_indices`` are considered.
    ``root_body`` is the body from which the traversal starts.
    """
    parent_to_joints: dict[str, list[int]] = {}
    for i in ir_indices:
        j = ir.joints[i]
        parent_to_joints.setdefault(j.parent_body, []).append(i)

    sorted_indices: list[int] = []
    stack: list[tuple[str, int | None]] = [(root_body, None)]
    visited: set[str] = {root_body}

    while stack:
        body, incoming_ji = stack.pop()
        if incoming_ji is not None:
            sorted_indices.append(incoming_ji)
        # Push children in reverse-sorted order so smallest IR-joint-index pops first
        for ji in reversed(sorted(parent_to_joints.get(body, []))):
            child = ir.joints[ji].child_body
            if child in visited:
                raise IRError(f"Cycle detected: body {child!r} reachable via multiple paths")
            visited.add(child)
            stack.append((child, ji))

    if len(sorted_indices) != len(ir_indices):
        disconnected = set(ir_indices) - set(sorted_indices)
        names = [ir.joints[i].name for i in disconnected]
        raise IRError(f"Disconnected joints (not reachable from root): {names}")

    return sorted_indices


def _ir_stable_topo_sort(
    ir: IRModel,
    root_body: str,
    ir_indices: list[int],
) -> list[int]:
    """Stable Kahn sort that preserves IR order whenever it is topological.

    The heap key is the joint's IR index, so newly eligible joints are merged
    with already eligible siblings in source order. Only the selected joint
    subset participates; the explicit world joint is handled separately by
    :func:`build_model`.
    """
    parent_to_joints: dict[str, list[int]] = {}
    for index in ir_indices:
        parent_to_joints.setdefault(ir.joints[index].parent_body, []).append(index)

    ready = list(parent_to_joints.get(root_body, ()))
    heapq.heapify(ready)
    sorted_indices: list[int] = []
    visited_bodies: set[str] = {root_body}

    while ready:
        joint_index = heapq.heappop(ready)
        child_body = ir.joints[joint_index].child_body
        if child_body in visited_bodies:
            raise IRError(f"Cycle detected: body {child_body!r} reachable via multiple paths")
        visited_bodies.add(child_body)
        sorted_indices.append(joint_index)
        for child_index in parent_to_joints.get(child_body, ()):
            heapq.heappush(ready, child_index)

    if len(sorted_indices) != len(ir_indices):
        selected = set(sorted_indices)
        names = [ir.joints[index].name for index in ir_indices if index not in selected]
        raise IRError(f"Disconnected joints (not reachable from root): {names}")

    return sorted_indices


# ──────────────────────────────── main factory ───────────────────────────────


def build_model(  # noqa: PLR0912, PLR0915 - one audited IR packing boundary
    ir: IRModel,
    *,
    root_joint: JointModel | None = None,
    preserve_joint_order: bool = False,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Model:
    """Consume an ``IRModel`` and return a frozen ``Model``.

    Responsibilities (see docs/concepts/parsers_and_ir.md §3):

    1.  Replace the root body's parent joint with ``root_joint`` if supplied
        (default: ``JointFixed``).
    2.  Resolve mimic edges to ``mimic_source/multiplier/offset`` arrays.
    3.  Topologically sort joints so parents precede children.
    4.  Assign ``idx_q``/``idx_v`` by accumulating ``nq_i``/``nv_i``.
    5.  Select concrete ``JointModel`` instances from ``IRJoint.kind`` + axis.
    6.  Pack per-joint numeric buffers.
    7.  Pack per-body inertias into the 10-vector form.
    8.  Build frames (including ``body_<name>`` defaults).
    9.  Build name → id dicts.
    10. Return the frozen model wrapper.

    **World-sentinel convention**: an ``IRJoint`` with ``parent_body="world"``
    signals that the joint connects to the universe (joint 0) directly.  That
    joint becomes model joint 1.  A synthetic ``JointFixed`` is *not* inserted;
    the IR joint's ``kind`` is used instead (unless ``root_joint`` overrides
    it).  This lets programmatic builders embed a ``JointFreeFlyer`` root
    without extra ``load(…, free_flyer=True)`` kwargs.

    Set ``preserve_joint_order=True`` to use a stable Kahn topological sort.
    Already-topological IR joint order is then retained exactly (after the
    universe entry); non-topological input is repaired stably. The default
    remains the historical deterministic DFS order.
    """
    identity_se3 = torch.tensor(_IDENTITY_SE3_VALS, dtype=dtype)

    # ── 1. Identify root structure ────────────────────────────────────────────
    world_ir_idxs = [i for i, j in enumerate(ir.joints) if j.parent_body == _WORLD_SENTINEL]

    if world_ir_idxs:
        # IR has an explicit joint connecting world → root_body.
        if len(world_ir_idxs) > 1:
            raise IRError(
                f"Multiple joints with parent_body='world' is not supported; "
                f"found: {[ir.joints[i].name for i in world_ir_idxs]}"
            )
        world_ir_ji = world_ir_idxs[0]
        world_ir_joint = ir.joints[world_ir_ji]
        root_body = world_ir_joint.child_body

        # Root joint model: IR kind OR user-supplied override
        _root_jm: JointModel = root_joint if root_joint is not None else _kind_to_joint_model(world_ir_joint)
        _root_placement = world_ir_joint.origin.to(dtype=dtype)
        _root_name = world_ir_joint.name

        # Regular joints = all except the world-parent joint
        regular_ir_indices = [i for i in range(len(ir.joints)) if i != world_ir_ji]
    else:
        # Standard: find root body (no incoming joint), insert synthetic JointFixed.
        child_bodies = {j.child_body for j in ir.joints}
        if ir.root_body:
            root_body = ir.root_body
            if root_body not in {b.name for b in ir.bodies}:
                raise IRError(f"root_body {root_body!r} not found in IR bodies")
        else:
            candidates = [b.name for b in ir.bodies if b.name not in child_bodies]
            if len(candidates) != 1:
                raise IRError(f"Expected exactly 1 root body, found {len(candidates)}: {candidates}")
            root_body = candidates[0]

        _root_jm = root_joint if root_joint is not None else JointFixed()
        _root_placement = identity_se3.clone()
        _root_name = "root_joint"
        regular_ir_indices = list(range(len(ir.joints)))

    # ── 2. Topological sort of regular IR joints ──────────────────────────────
    if preserve_joint_order:
        sorted_ir_indices = _ir_stable_topo_sort(
            ir,
            root_body,
            regular_ir_indices,
        )
    else:
        sorted_ir_indices = _ir_topo_sort(ir, root_body, regular_ir_indices)

    # ── 3. Model joint layout ─────────────────────────────────────────────────
    # joint 0 = universe (JointUniverse, body = "universe" placeholder)
    # joint 1 = root joint (body = root_body)
    # joints 2..N = regular IR joints in topo order
    n_model_joints = 2 + len(sorted_ir_indices)

    body_to_mjidx: dict[str, int] = {root_body: 1}
    for offset, ir_ji in enumerate(sorted_ir_indices):
        body_to_mjidx[ir.joints[ir_ji].child_body] = offset + 2

    # ── 4. Names ──────────────────────────────────────────────────────────────
    joint_names_list = ["universe", _root_name]
    body_names_list = ["universe", root_body]
    for ir_ji in sorted_ir_indices:
        joint_names_list.append(ir.joints[ir_ji].name)
        body_names_list.append(ir.joints[ir_ji].child_body)

    # ── 5. Parents ────────────────────────────────────────────────────────────
    parents_list: list[int] = [-1, 0]
    for ir_ji in sorted_ir_indices:
        parent_body = ir.joints[ir_ji].parent_body
        parents_list.append(body_to_mjidx[parent_body])
    parents = tuple(parents_list)

    # ── 6. Topology ───────────────────────────────────────────────────────────
    topo_order = topo_sort(parents)
    children = build_children(parents)
    subtrees = build_subtrees(parents)
    supports = build_supports(parents)

    # ── 7. Joint models ───────────────────────────────────────────────────────
    joint_models_list: list[JointModel] = [JointUniverse(), _root_jm]
    for ir_ji in sorted_ir_indices:
        joint_models_list.append(_kind_to_joint_model(ir.joints[ir_ji]))
    joint_models = tuple(joint_models_list)

    # ── 8. idx_q / idx_v ─────────────────────────────────────────────────────
    nqs_full = tuple(jm.nq for jm in joint_models_list)
    nvs_full = tuple(jm.nv for jm in joint_models_list)
    idx_qs_full, nq_full = _cumulative_layout(nqs_full)
    idx_vs_full, nv_full = _cumulative_layout(nvs_full)

    # ── 9. joint_placements ───────────────────────────────────────────────────
    placements: list[torch.Tensor] = [identity_se3.clone(), _root_placement]
    for ir_ji in sorted_ir_indices:
        placements.append(ir.joints[ir_ji].origin.to(dtype=dtype))
    joint_placements = torch.stack(placements, dim=0)

    # ── 10. Limits ────────────────────────────────────────────────────────────
    lower_pos: list[float] = []
    upper_pos: list[float] = []
    vel_lim: list[float] = []
    eff_lim: list[float] = []

    _INF = float("inf")

    # Helper: get IR joint for model joint index j_idx
    def _ir_joint_for(j_idx: int) -> IRJoint | None:
        if j_idx == 1 and world_ir_idxs:
            return world_ir_joint  # root joint came from an IR joint
        if j_idx >= 2:
            return ir.joints[sorted_ir_indices[j_idx - 2]]
        return None

    for j_idx, jm in enumerate(joint_models_list):
        if jm.nq == 0:
            continue

        kind = jm.kind
        ir_j = _ir_joint_for(j_idx)

        if kind == "free_flyer":
            lower_pos.extend([-_INF] * 7)
            upper_pos.extend([_INF] * 7)
            vel_lim.extend([_INF] * 6)
            eff_lim.extend([_INF] * 6)

        elif kind == "revolute_unbounded":
            lower_pos.extend([-_INF, -_INF])
            upper_pos.extend([_INF, _INF])
            vel = ir_j.velocity_limit if ir_j is not None and ir_j.velocity_limit is not None else 0.0
            eff = ir_j.effort_limit if ir_j is not None and ir_j.effort_limit is not None else 0.0
            vel_lim.append(vel)
            eff_lim.append(eff)

        elif kind == "spherical":
            lower_pos.extend([-1.0] * 4)
            upper_pos.extend([1.0] * 4)
            v = ir_j.velocity_limit if ir_j is not None and ir_j.velocity_limit is not None else 0.0
            e = ir_j.effort_limit if ir_j is not None and ir_j.effort_limit is not None else 0.0
            vel_lim.extend([v] * 3)
            eff_lim.extend([e] * 3)

        elif kind == "planar":
            lower_pos.extend([-_INF, -_INF, -1.0, -1.0])
            upper_pos.extend([_INF, _INF, 1.0, 1.0])
            vel_lim.extend([0.0] * 3)
            eff_lim.extend([0.0] * 3)

        elif kind == "translation":
            lower_pos.extend([-_INF] * 3)
            upper_pos.extend([_INF] * 3)
            vel_lim.extend([0.0] * 3)
            eff_lim.extend([0.0] * 3)

        else:
            # Scalar IR limits apply to every coordinate of generic/custom
            # joints. This is also the fallback for programmatic composites.
            if ir_j is not None:
                lo = ir_j.lower if ir_j.lower is not None else -_INF
                hi = ir_j.upper if ir_j.upper is not None else _INF
                vel = ir_j.velocity_limit if ir_j.velocity_limit is not None else 0.0
                eff = ir_j.effort_limit if ir_j.effort_limit is not None else 0.0
            else:
                lo, hi, vel, eff = -_INF, _INF, _INF, _INF
            lower_pos.extend([lo] * jm.nq)
            upper_pos.extend([hi] * jm.nq)
            vel_lim.extend([vel] * jm.nv)
            eff_lim.extend([eff] * jm.nv)

    lower_pos_limit = torch.tensor(lower_pos, dtype=dtype)
    upper_pos_limit = torch.tensor(upper_pos, dtype=dtype)
    velocity_limit = torch.tensor(vel_lim, dtype=dtype)
    effort_limit = torch.tensor(eff_lim, dtype=dtype)
    rotor_inertia = torch.zeros(nv_full, dtype=dtype)
    armature = torch.zeros(nv_full, dtype=dtype)
    friction = torch.zeros(nv_full, dtype=dtype)
    damping = torch.zeros(nv_full, dtype=dtype)

    # ── 11. Body inertias ─────────────────────────────────────────────────────
    ir_body_map: dict[str, IRBody] = {b.name: b for b in ir.bodies}
    # Universe body: zero inertia placeholder
    inertia_rows: list[torch.Tensor] = [torch.zeros(10, dtype=dtype)]
    # Root body
    if root_body in ir_body_map:
        inertia_rows.append(_pack_inertia(ir_body_map[root_body], dtype))
    else:
        inertia_rows.append(torch.zeros(10, dtype=dtype))
    # Regular bodies
    for ir_ji in sorted_ir_indices:
        bname = ir.joints[ir_ji].child_body
        if bname in ir_body_map:
            inertia_rows.append(_pack_inertia(ir_body_map[bname], dtype))
        else:
            inertia_rows.append(torch.zeros(10, dtype=dtype))
    body_inertias = torch.stack(inertia_rows, dim=0)

    # ── 12. Frames ────────────────────────────────────────────────────────────
    frames_list: list[Frame] = []
    for midx, bname in enumerate(body_names_list):
        frames_list.append(
            Frame(
                name=f"body_{bname}",
                parent_joint=midx,
                joint_placement=identity_se3.clone(),
                frame_type="body",
            )
        )
    for ir_frame in ir.frames:
        parent_midx = body_to_mjidx.get(ir_frame.parent_body, 1)
        frames_list.append(
            Frame(
                name=ir_frame.name,
                parent_joint=parent_midx,
                joint_placement=ir_frame.placement.to(dtype=dtype),
                frame_type=ir_frame.frame_type,  # type: ignore[arg-type]
            )
        )
    frames = tuple(frames_list)
    nframes = len(frames)
    frame_names = tuple(f.name for f in frames)

    # ── 13. Name → id dicts ───────────────────────────────────────────────────
    joint_name_to_id: dict[str, int] = {n: i for i, n in enumerate(joint_names_list)}
    body_name_to_id: dict[str, int] = {n: i for i, n in enumerate(body_names_list)}
    frame_name_to_id: dict[str, int] = {n: i for i, n in enumerate(frame_names)}
    joint_names = tuple(joint_names_list)
    body_names = tuple(body_names_list)

    # ── 14. Mimic ─────────────────────────────────────────────────────────────
    ir_joint_name_to_mjidx: dict[str, int] = {}
    if world_ir_idxs:
        ir_joint_name_to_mjidx[world_ir_joint.name] = 1
    for offset, ir_ji in enumerate(sorted_ir_indices):
        ir_joint_name_to_mjidx[ir.joints[ir_ji].name] = offset + 2

    mimic_mult = torch.ones(n_model_joints, dtype=dtype)
    mimic_off = torch.zeros(n_model_joints, dtype=dtype)
    mimic_src_list: list[int] = list(range(n_model_joints))
    mimic_targets: set[int] = set()

    mimic_ir_joints: list[tuple[IRJoint, int]] = []
    if world_ir_idxs:
        mimic_ir_joints.append((world_ir_joint, 1))
    mimic_ir_joints.extend((ir.joints[ir_ji], offset + 2) for offset, ir_ji in enumerate(sorted_ir_indices))

    for ir_j, mjidx in mimic_ir_joints:
        if ir_j.mimic_source is not None:
            src = ir_joint_name_to_mjidx.get(ir_j.mimic_source)
            if src is None:
                raise IRError(f"Mimic source {ir_j.mimic_source!r} not found (referenced by joint {ir_j.name!r})")
            mimic_targets.add(mjidx)
            mimic_src_list[mjidx] = src
            mimic_mult[mjidx] = ir_j.mimic_multiplier
            mimic_off[mjidx] = ir_j.mimic_offset

    mimic_source = tuple(mimic_src_list)

    mimic_reduction = _build_mimic_reduction(
        joint_models=joint_models,
        joint_names=joint_names,
        mimic_source=mimic_source,
        mimic_multiplier=mimic_mult,
        mimic_offset=mimic_off,
        mimic_targets=mimic_targets,
        nqs_full=nqs_full,
        nvs_full=nvs_full,
        idx_qs_full=idx_qs_full,
        idx_vs_full=idx_vs_full,
        lower_full=lower_pos_limit,
        upper_full=upper_pos_limit,
        velocity_full=velocity_limit,
        effort_full=effort_limit,
        rotor_full=rotor_inertia,
        armature_full=armature,
        friction_full=friction,
        damping_full=damping,
    )
    nqs = mimic_reduction.nqs
    nvs = mimic_reduction.nvs
    idx_qs = mimic_reduction.idx_qs
    idx_vs = mimic_reduction.idx_vs
    nq_total = mimic_reduction.nq
    nv_total = mimic_reduction.nv
    q_expansion = mimic_reduction.q_expansion
    q_offset = mimic_reduction.q_offset
    v_expansion = mimic_reduction.v_expansion
    lower_pos_limit = mimic_reduction.lower_pos_limit
    upper_pos_limit = mimic_reduction.upper_pos_limit
    velocity_limit = mimic_reduction.velocity_limit
    effort_limit = mimic_reduction.effort_limit
    rotor_inertia = mimic_reduction.rotor_inertia
    armature = mimic_reduction.armature
    friction = mimic_reduction.friction
    damping = mimic_reduction.damping

    # ── 15. q_neutral ─────────────────────────────────────────────────────────
    neutral_parts: list[torch.Tensor] = []
    for joint_id, jm in enumerate(joint_models_list):
        if nqs[joint_id] > 0:
            neutral_parts.append(jm.neutral())
    q_neutral = (
        torch.cat(neutral_parts, dim=-1).to(dtype=dtype) if neutral_parts else torch.zeros(nq_total, dtype=dtype)
    )

    # ── 16. Gravity ───────────────────────────────────────────────────────────
    gravity = ir.gravity.to(dtype=dtype)

    # ── 17. Move to device ────────────────────────────────────────────────────
    if device is not None:

        def _dev(t: torch.Tensor) -> torch.Tensor:
            return t.to(device=device)

        joint_placements = _dev(joint_placements)
        body_inertias = _dev(body_inertias)
        lower_pos_limit = _dev(lower_pos_limit)
        upper_pos_limit = _dev(upper_pos_limit)
        velocity_limit = _dev(velocity_limit)
        effort_limit = _dev(effort_limit)
        rotor_inertia = _dev(rotor_inertia)
        armature = _dev(armature)
        friction = _dev(friction)
        damping = _dev(damping)
        mimic_mult = _dev(mimic_mult)
        mimic_off = _dev(mimic_off)
        q_expansion = _dev(q_expansion)
        q_offset = _dev(q_offset)
        v_expansion = _dev(v_expansion)
        q_neutral = _dev(q_neutral)
        gravity = _dev(gravity)

    # ── 18. Enforce topology invariants (docs/conventions/contracts.md §1.5) ───────────
    _check_topology_invariants(
        parents=parents,
        nqs=nqs,
        nvs=nvs,
        idx_qs=idx_qs,
        idx_vs=idx_vs,
        nq_total=nq_total,
        nv_total=nv_total,
    )
    _check_topology_invariants(
        parents=parents,
        nqs=nqs_full,
        nvs=nvs_full,
        idx_qs=idx_qs_full,
        idx_vs=idx_vs_full,
        nq_total=nq_full,
        nv_total=nv_full,
    )

    # ── 19. Pack the one canonical structure/value pair ───────────────────────
    model_device = joint_placements.device
    codes = tuple(JOINT_KIND_CODES.get(joint.kind, -1) for joint in joint_models)
    axes = torch.tensor(
        [_joint_axis(joint) for joint in joint_models],
        device=model_device,
        dtype=dtype,
    )
    pitches = torch.tensor(
        [float(getattr(joint, "pitch", 0.0)) for joint in joint_models],
        device=model_device,
        dtype=dtype,
    )
    max_nv = max(nvs_full, default=0)
    motion_subspaces = torch.zeros(
        (n_model_joints, 6, max_nv),
        device=model_device,
        dtype=dtype,
    )
    for index, joint in enumerate(joint_models):
        if joint.nv == 0:
            continue
        neutral = joint.neutral().to(device=model_device, dtype=dtype)
        subspace = joint.joint_motion_subspace(neutral).to(device=model_device, dtype=dtype)
        motion_subspaces[index, :, : joint.nv] = subspace

    child_offsets, child_indices = _flatten_rows(children)
    subtree_offsets, subtree_indices = _flatten_rows(subtrees)
    support_offsets, support_indices = _flatten_rows(supports)

    manifold_groups: dict[str, list[int]] = {
        "euclidean": [],
        "spherical": [],
        "free_flyer": [],
        "unbounded": [],
        "planar": [],
        "fallback": [],
    }
    for joint_id, joint in enumerate(joint_models):
        public_nq = nqs[joint_id]
        public_nv = nvs[joint_id]
        if public_nq == 0 and public_nv == 0:
            continue
        if public_nq != joint.nq or public_nv != joint.nv:
            manifold_groups["fallback"].append(joint_id)
            continue

        joint_type = type(joint)
        if joint_type in _EUCLIDEAN_MANIFOLD_TYPES:
            manifold_groups["euclidean"].append(joint_id)
        elif joint_type is JointSpherical:
            manifold_groups["spherical"].append(joint_id)
        elif joint_type is JointFreeFlyer:
            manifold_groups["free_flyer"].append(joint_id)
        elif joint_type is JointRevoluteUnbounded:
            manifold_groups["unbounded"].append(joint_id)
        elif joint_type is JointPlanar:
            manifold_groups["planar"].append(joint_id)
        else:
            manifold_groups["fallback"].append(joint_id)

    euclidean_ids = tuple(manifold_groups["euclidean"])
    spherical_ids = tuple(manifold_groups["spherical"])
    free_flyer_ids = tuple(manifold_groups["free_flyer"])
    unbounded_ids = tuple(manifold_groups["unbounded"])
    planar_ids = tuple(manifold_groups["planar"])
    fallback_ids = tuple(manifold_groups["fallback"])

    _, euclidean_q_indices = _flat_slice_indices(euclidean_ids, idx_qs, nqs, device=model_device)
    _, euclidean_v_indices = _flat_slice_indices(euclidean_ids, idx_vs, nvs, device=model_device)
    spherical_q_indices = _matrix_slice_indices(spherical_ids, idx_qs, 4, device=model_device)
    spherical_v_indices = _matrix_slice_indices(spherical_ids, idx_vs, 3, device=model_device)
    free_flyer_q_indices = _matrix_slice_indices(free_flyer_ids, idx_qs, 7, device=model_device)
    free_flyer_v_indices = _matrix_slice_indices(free_flyer_ids, idx_vs, 6, device=model_device)
    unbounded_q_indices = _matrix_slice_indices(unbounded_ids, idx_qs, 2, device=model_device)
    unbounded_v_indices = _matrix_slice_indices(unbounded_ids, idx_vs, 1, device=model_device)
    planar_q_indices = _matrix_slice_indices(planar_ids, idx_qs, 4, device=model_device)
    planar_v_indices = _matrix_slice_indices(planar_ids, idx_vs, 3, device=model_device)
    fallback_q_offsets, fallback_q_indices = _flat_slice_indices(
        fallback_ids,
        idx_qs,
        nqs,
        device=model_device,
    )
    fallback_v_offsets, fallback_v_indices = _flat_slice_indices(
        fallback_ids,
        idx_vs,
        nvs,
        device=model_device,
    )

    def i32(values: list[int] | tuple[int, ...]) -> torch.Tensor:
        return torch.tensor(values, device=model_device, dtype=torch.int32)

    frame_parent_joint_ids = tuple(frame.parent_joint for frame in frames)
    structure = ModelStructure(
        njoints=n_model_joints,
        nbodies=n_model_joints,
        nframes=nframes,
        nq=nq_total,
        nv=nv_total,
        nq_full=nq_full,
        nv_full=nv_full,
        name=ir.name,
        joint_names=joint_names,
        body_names=body_names,
        frame_names=frame_names,
        joint_name_to_id=joint_name_to_id,
        body_name_to_id=body_name_to_id,
        frame_name_to_id=frame_name_to_id,
        frame_parent_joint_ids=frame_parent_joint_ids,
        frame_types=tuple(frame.frame_type for frame in frames),
        parents=parents,
        children=children,
        subtrees=subtrees,
        supports=supports,
        topo_order=topo_order,
        nqs=nqs,
        nvs=nvs,
        idx_qs=idx_qs,
        idx_vs=idx_vs,
        nqs_full=nqs_full,
        nvs_full=nvs_full,
        idx_qs_full=idx_qs_full,
        idx_vs_full=idx_vs_full,
        joint_models=joint_models,
        joint_kind_codes=codes,
        mimic_source=mimic_source,
        has_mimic=bool(mimic_targets),
        manifold_euclidean_joint_ids=euclidean_ids,
        manifold_spherical_joint_ids=spherical_ids,
        manifold_free_flyer_joint_ids=free_flyer_ids,
        manifold_unbounded_joint_ids=unbounded_ids,
        manifold_planar_joint_ids=planar_ids,
        manifold_fallback_joint_ids=fallback_ids,
        manifold_fallback_q_offsets=fallback_q_offsets,
        manifold_fallback_v_offsets=fallback_v_offsets,
        joint_kind_tensor=torch.tensor(codes, device=model_device, dtype=torch.int8),
        parents_tensor=i32(parents),
        topo_order_tensor=i32(topo_order),
        nqs_tensor=i32(nqs),
        nvs_tensor=i32(nvs),
        idx_qs_tensor=i32(idx_qs),
        idx_vs_tensor=i32(idx_vs),
        nqs_full_tensor=i32(nqs_full),
        nvs_full_tensor=i32(nvs_full),
        idx_qs_full_tensor=i32(idx_qs_full),
        idx_vs_full_tensor=i32(idx_vs_full),
        children_offsets=i32(child_offsets),
        children_indices=i32(child_indices),
        subtree_offsets=i32(subtree_offsets),
        subtree_indices=i32(subtree_indices),
        support_offsets=i32(support_offsets),
        support_indices=i32(support_indices),
        frame_parent_joints=i32(frame_parent_joint_ids),
        mimic_source_tensor=i32(mimic_source),
        q_expansion=q_expansion,
        q_offset=q_offset,
        v_expansion=v_expansion,
        manifold_euclidean_q_indices=euclidean_q_indices,
        manifold_euclidean_v_indices=euclidean_v_indices,
        manifold_spherical_q_indices=spherical_q_indices,
        manifold_spherical_v_indices=spherical_v_indices,
        manifold_free_flyer_q_indices=free_flyer_q_indices,
        manifold_free_flyer_v_indices=free_flyer_v_indices,
        manifold_unbounded_q_indices=unbounded_q_indices,
        manifold_unbounded_v_indices=unbounded_v_indices,
        manifold_planar_q_indices=planar_q_indices,
        manifold_planar_v_indices=planar_v_indices,
        manifold_fallback_q_indices=fallback_q_indices,
        manifold_fallback_v_indices=fallback_v_indices,
        joint_axes=axes,
        joint_pitches=pitches,
        joint_motion_subspaces=motion_subspaces,
    )
    frame_placements = (
        torch.stack(
            [frame.joint_placement.to(device=model_device, dtype=dtype) for frame in frames],
            dim=0,
        )
        if frames
        else joint_placements.new_empty((0, 7))
    )
    values = ModelValues(
        joint_placements=joint_placements,
        body_inertias=body_inertias,
        frame_placements=frame_placements,
        lower_pos_limit=lower_pos_limit,
        upper_pos_limit=upper_pos_limit,
        velocity_limit=velocity_limit,
        effort_limit=effort_limit,
        rotor_inertia=rotor_inertia,
        armature=armature,
        friction=friction,
        damping=damping,
        gravity=gravity,
        mimic_multiplier=mimic_mult,
        mimic_offset=mimic_off,
        q_neutral=q_neutral,
    )
    return Model(
        structure=structure,
        values=values,
        meta={"ir": ir, **dict(getattr(ir, "meta", {}) or {})},
    )
