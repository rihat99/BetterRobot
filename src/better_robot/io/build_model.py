"""``build_model(ir, root_joint=...)`` — IR → frozen ``Model`` factory."""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import torch

from ..exceptions import ModelError
from ..model.joints import (
    JointFixed,
    JointFreeFlyer,
    JointHelical,
    JointModel,
    JointPlanar,
    JointPrismatic,
    JointRevolute,
    JointRevoluteUnbounded,
    JointSpherical,
    JointTranslation,
)
from ..model.model import Model
from .ir import IRBody, IRJoint, IRModel

# ──────────────────────────────── constants ──────────────────────────────────

#: Parent body name that signals "connect to the universe (joint 0)".
_WORLD_SENTINEL = "world"

_IDENTITY_SE3_VALS = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
_SUPPORTED_MIMIC_JOINTS = (JointRevolute, JointPrismatic, JointHelical)


@dataclass(frozen=True, slots=True)
class _MimicReduction:
    """Named build-time result for reduced mimic coordinates.

    ``source_ids``, ``multipliers`` and ``offsets`` hold each joint's resolved chain: the public root joint, the
    total scale and the total offset (a joint that is not a mimic target maps to itself with 1 and 0).
    """

    nqs: tuple[int, ...]
    nvs: tuple[int, ...]
    idx_qs: tuple[int, ...]
    idx_vs: tuple[int, ...]
    nq: int
    nv: int
    source_ids: tuple[int, ...]
    multipliers: tuple[float, ...]
    offsets: tuple[float, ...]
    lower_position_limit: torch.Tensor
    upper_position_limit: torch.Tensor
    velocity_limit: torch.Tensor
    effort_limit: torch.Tensor
    armature: torch.Tensor
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

    Raises `ModelError` at
    build time so the caller never sees a ``Model`` in an inconsistent
    state.
    """
    # 1. Universe joint is rooted at -1.
    if parents[0] != -1:
        raise ModelError(f"parents[0] must be -1 (universe), got {parents[0]}")
    # 2. Topologically sorted: parents[i] < i for i > 0.
    for i, p in enumerate(parents[1:], start=1):
        if p >= i:
            raise ModelError(f"parents[{i}]={p} >= {i}; topological sort violated")
    # 3. sum(nqs) == nq, sum(nvs) == nv.
    if sum(nqs) != nq_total:
        raise ModelError(f"sum(nqs)={sum(nqs)} != nq={nq_total}")
    if sum(nvs) != nv_total:
        raise ModelError(f"sum(nvs)={sum(nvs)} != nv={nv_total}")
    # 4. Contiguous slicing: idx_qs[i] + nqs[i] == idx_qs[i+1] (and same for v).
    for i in range(len(nqs) - 1):
        if idx_qs[i] + nqs[i] != idx_qs[i + 1]:
            raise ModelError(
                f"idx_qs[{i}] + nqs[{i}] = {idx_qs[i] + nqs[i]} != idx_qs[{i + 1}]={idx_qs[i + 1]} (q-slicing gap)"
            )
        if idx_vs[i] + nvs[i] != idx_vs[i + 1]:
            raise ModelError(
                f"idx_vs[{i}] + nvs[{i}] = {idx_vs[i] + nvs[i]} != idx_vs[{i + 1}]={idx_vs[i + 1]} (v-slicing gap)"
            )


# ─────────────────────────────── helpers ─────────────────────────────────��───


def _cumulative_layout(widths: tuple[int, ...]) -> tuple[tuple[int, ...], int]:
    indices: list[int] = []
    offset = 0
    for width in widths:
        indices.append(offset)
        offset += width
    return tuple(indices), offset


def _build_mimic_reduction(  # noqa: PLR0912, PLR0913, PLR0915 - one build-time policy boundary
    *,
    joint_models: tuple[JointModel, ...],
    joint_names: tuple[str, ...],
    mimic_source: tuple[int, ...],
    mimic_multiplier: tuple[float, ...],
    mimic_offset: tuple[float, ...],
    mimic_targets: set[int],
    nqs_full: tuple[int, ...],
    nvs_full: tuple[int, ...],
    idx_qs_full: tuple[int, ...],
    idx_vs_full: tuple[int, ...],
    lower_full: torch.Tensor,
    upper_full: torch.Tensor,
    velocity_full: torch.Tensor,
    effort_full: torch.Tensor,
    armature_full: torch.Tensor,
    damping_full: torch.Tensor,
) -> _MimicReduction:
    """Build public layouts, resolved mimic chains, and reduced scalar limits."""

    for target in mimic_targets:
        source = mimic_source[target]
        if source == target:
            raise ModelError(f"Mimic cycle detected: joint {joint_names[target]!r} references itself")
        for role, joint_id in (("target", target), ("source", source)):
            joint = joint_models[joint_id]
            if not isinstance(joint, _SUPPORTED_MIMIC_JOINTS):
                raise ModelError(
                    f"Mimic {role} joint {joint_names[joint_id]!r} has unsupported "
                    f"kind {type(joint).__name__}; reduced mimic coordinates currently "
                    "require concrete scalar Euclidean revolute, prismatic, or "
                    "helical joints (nq=nv=1)"
                )

    public_nqs = tuple(0 if joint_id in mimic_targets else width for joint_id, width in enumerate(nqs_full))
    public_nvs = tuple(0 if joint_id in mimic_targets else width for joint_id, width in enumerate(nvs_full))
    public_idx_qs, nq = _cumulative_layout(public_nqs)
    public_idx_vs, nv = _cumulative_layout(public_nvs)

    state = [0] * len(joint_models)
    resolved: list[tuple[int, float, float] | None] = [None] * len(joint_models)

    def resolve(joint_id: int, path: tuple[int, ...] = ()) -> tuple[int, float, float]:
        if joint_id not in mimic_targets:
            return joint_id, 1.0, 0.0
        if state[joint_id] == 1:
            cycle = (*path, joint_id)
            names = " -> ".join(joint_names[index] for index in cycle)
            raise ModelError(f"Mimic cycle detected: {names}")
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

    chains = [resolve(joint_id) for joint_id in range(len(joint_models))]

    lower = lower_full.new_empty((nq,))
    upper = upper_full.new_empty((nq,))
    velocity = velocity_full.new_empty((nv,))
    effort = effort_full.new_empty((nv,))
    armature = armature_full.new_empty((nv,))
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
        armature[reduced_iv : reduced_iv + nv_joint] = armature_full[full_iv : full_iv + nv_joint]
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
                raise ModelError(
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
            raise ModelError(
                f"Mimic limits for source joint {joint_names[root]!r} are empty "
                f"after applying target {joint_names[target]!r}"
            )
        absolute_scale = abs(scale)
        velocity[reduced_iv] = torch.minimum(velocity[reduced_iv], velocity_full[full_iv] / absolute_scale)
        effort[reduced_iv] = effort[reduced_iv] + absolute_scale * effort_full[full_iv]
        armature[reduced_iv] = armature[reduced_iv] + scale**2 * armature_full[full_iv]
        damping[reduced_iv] = damping[reduced_iv] + scale**2 * damping_full[full_iv]

    return _MimicReduction(
        nqs=public_nqs,
        nvs=public_nvs,
        idx_qs=public_idx_qs,
        idx_vs=public_idx_vs,
        nq=nq,
        nv=nv,
        source_ids=tuple(root for root, _, _ in chains),
        multipliers=tuple(scale for _, scale, _ in chains),
        offsets=tuple(offset for _, _, offset in chains),
        lower_position_limit=lower,
        upper_position_limit=upper,
        velocity_limit=velocity,
        effort_limit=effort,
        armature=armature,
        damping=damping,
    )


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
                raise ModelError(f"Cycle detected: body {child!r} reachable via multiple paths")
            visited.add(child)
            stack.append((child, ji))

    if len(sorted_indices) != len(ir_indices):
        disconnected = set(ir_indices) - set(sorted_indices)
        names = [ir.joints[i].name for i in disconnected]
        raise ModelError(f"Disconnected joints (not reachable from root): {names}")

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
    `build_model`.
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
            raise ModelError(f"Cycle detected: body {child_body!r} reachable via multiple paths")
        visited_bodies.add(child_body)
        sorted_indices.append(joint_index)
        for child_index in parent_to_joints.get(child_body, ()):
            heapq.heappush(ready, child_index)

    if len(sorted_indices) != len(ir_indices):
        selected = set(sorted_indices)
        names = [ir.joints[index].name for index in ir_indices if index not in selected]
        raise ModelError(f"Disconnected joints (not reachable from root): {names}")

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
    2.  Resolve mimic chains to each joint's public root, total multiplier and offset.
    3.  Topologically sort joints so parents precede children.
    4.  Assign ``idx_q``/``idx_v`` by accumulating ``nq_i``/``nv_i``.
    5.  Take each joint's ``JointModel`` from its ``IRJoint.joint``.
    6.  Pack per-joint limits, armature and damping.
    7.  Pack per-body inertias into the 10-vector form.
    8.  Build frames (including ``body_<name>`` defaults).
    9.  Return the frozen ``Model``, moved to ``device`` when one is given.

    **World-sentinel convention**: an ``IRJoint`` with ``parent_body="world"``
    signals that the joint connects to the universe (joint 0) directly.  That
    joint becomes model joint 1.  A synthetic ``JointFixed`` is *not* inserted;
    the IR joint's ``joint`` is used instead (unless ``root_joint`` overrides
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
            raise ModelError(
                f"Multiple joints with parent_body='world' is not supported; "
                f"found: {[ir.joints[i].name for i in world_ir_idxs]}"
            )
        world_ir_ji = world_ir_idxs[0]
        world_ir_joint = ir.joints[world_ir_ji]
        root_body = world_ir_joint.child_body

        # Root joint model: IR kind OR user-supplied override
        _root_jm: JointModel = root_joint if root_joint is not None else world_ir_joint.joint
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
                raise ModelError(f"root_body {root_body!r} not found in IR bodies")
        else:
            candidates = [b.name for b in ir.bodies if b.name not in child_bodies]
            if len(candidates) != 1:
                raise ModelError(f"Expected exactly 1 root body, found {len(candidates)}: {candidates}")
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
    # joint 0 = universe (JointFixed, body = "universe" placeholder)
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

    # ── 7. Joint models ───────────────────────────────────────────────────────
    joint_models_list: list[JointModel] = [JointFixed(), _root_jm]
    for ir_ji in sorted_ir_indices:
        joint_models_list.append(ir.joints[ir_ji].joint)
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

        ir_j = _ir_joint_for(j_idx)

        if isinstance(jm, JointFreeFlyer):
            lower_pos.extend([-_INF] * 7)
            upper_pos.extend([_INF] * 7)
            vel_lim.extend([_INF] * 6)
            eff_lim.extend([_INF] * 6)

        elif isinstance(jm, JointRevoluteUnbounded):
            lower_pos.extend([-_INF, -_INF])
            upper_pos.extend([_INF, _INF])
            vel = ir_j.velocity_limit if ir_j is not None and ir_j.velocity_limit is not None else 0.0
            eff = ir_j.effort_limit if ir_j is not None and ir_j.effort_limit is not None else 0.0
            vel_lim.append(vel)
            eff_lim.append(eff)

        elif isinstance(jm, JointSpherical):
            lower_pos.extend([-1.0] * 4)
            upper_pos.extend([1.0] * 4)
            v = ir_j.velocity_limit if ir_j is not None and ir_j.velocity_limit is not None else 0.0
            e = ir_j.effort_limit if ir_j is not None and ir_j.effort_limit is not None else 0.0
            vel_lim.extend([v] * 3)
            eff_lim.extend([e] * 3)

        elif isinstance(jm, JointPlanar):
            lower_pos.extend([-_INF, -_INF, -1.0, -1.0])
            upper_pos.extend([_INF, _INF, 1.0, 1.0])
            vel_lim.extend([0.0] * 3)
            eff_lim.extend([0.0] * 3)

        elif isinstance(jm, JointTranslation):
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
    # Scalar IR joint dynamics apply to every velocity coordinate of the joint, like the limits above.
    joint_dynamics = [
        (ir_j.armature, ir_j.damping) if (ir_j := _ir_joint_for(j_idx)) is not None else (0.0, 0.0)
        for j_idx in range(len(joint_models_list))
    ]
    armature, damping = (
        torch.tensor(
            [row[k] for row, jm in zip(joint_dynamics, joint_models_list) for _ in range(jm.nv)], dtype=dtype
        ).reshape(nv_full)
        for k in range(2)
    )

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

    # ── 12. Frames: one ``body_<name>`` frame per body, then the IR frames ───
    frame_names_list = [f"body_{bname}" for bname in body_names_list]
    frame_parent_ids = list(range(len(body_names_list)))
    frame_placement_rows = [identity_se3.clone() for _ in body_names_list]
    for ir_frame in ir.frames:
        frame_names_list.append(ir_frame.name)
        frame_parent_ids.append(body_to_mjidx.get(ir_frame.parent_body, 1))
        frame_placement_rows.append(ir_frame.placement.to(dtype=dtype))
    frame_placements = torch.stack(frame_placement_rows, dim=0)

    # ── 13. Names ─────────────────────────────────────────────────────────────
    joint_names = tuple(joint_names_list)
    body_names = tuple(body_names_list)
    frame_names = tuple(frame_names_list)

    # ── 14. Mimic ─────────────────────────────────────────────────────────────
    ir_joint_name_to_mjidx: dict[str, int] = {}
    if world_ir_idxs:
        ir_joint_name_to_mjidx[world_ir_joint.name] = 1
    for offset, ir_ji in enumerate(sorted_ir_indices):
        ir_joint_name_to_mjidx[ir.joints[ir_ji].name] = offset + 2

    mimic_mult = [1.0] * n_model_joints
    mimic_off = [0.0] * n_model_joints
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
                raise ModelError(f"Mimic source {ir_j.mimic_source!r} not found (referenced by joint {ir_j.name!r})")
            mimic_targets.add(mjidx)
            mimic_src_list[mjidx] = src
            mimic_mult[mjidx] = float(ir_j.mimic_multiplier)
            mimic_off[mjidx] = float(ir_j.mimic_offset)

    mimic_source = tuple(mimic_src_list)

    reduction = _build_mimic_reduction(
        joint_models=joint_models,
        joint_names=joint_names,
        mimic_source=mimic_source,
        mimic_multiplier=tuple(mimic_mult),
        mimic_offset=tuple(mimic_off),
        mimic_targets=mimic_targets,
        nqs_full=nqs_full,
        nvs_full=nvs_full,
        idx_qs_full=idx_qs_full,
        idx_vs_full=idx_vs_full,
        lower_full=lower_pos_limit,
        upper_full=upper_pos_limit,
        velocity_full=velocity_limit,
        effort_full=effort_limit,
        armature_full=armature,
        damping_full=damping,
    )
    nqs = reduction.nqs
    nvs = reduction.nvs
    idx_qs = reduction.idx_qs
    idx_vs = reduction.idx_vs
    nq_total = reduction.nq
    nv_total = reduction.nv

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

    # ── 17. Enforce topology invariants (docs/conventions/contracts.md §1.5) ───────────
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

    # ── 18. The record ────────────────────────────────────────────────────────
    model = Model(
        name=ir.name,
        joint_names=joint_names,
        body_names=body_names,
        frame_names=frame_names,
        joints=joint_models,
        parents=parents,
        frame_parent_joint_ids=tuple(frame_parent_ids),
        nqs=nqs,
        nvs=nvs,
        idx_qs=idx_qs,
        idx_vs=idx_vs,
        nqs_full=nqs_full,
        nvs_full=nvs_full,
        idx_qs_full=idx_qs_full,
        idx_vs_full=idx_vs_full,
        mimic_source_ids=reduction.source_ids,
        mimic_multipliers=reduction.multipliers,
        mimic_offsets=reduction.offsets,
        joint_placements=joint_placements,
        body_inertias=body_inertias,
        frame_placements=frame_placements,
        lower_position_limit=reduction.lower_position_limit,
        upper_position_limit=reduction.upper_position_limit,
        velocity_limit=reduction.velocity_limit,
        effort_limit=reduction.effort_limit,
        armature=reduction.armature,
        damping=reduction.damping,
        gravity=gravity,
        q_neutral=q_neutral,
    )
    return model if device is None else model.to(device=device)
