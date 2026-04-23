"""``build_model(ir, root_joint=...)`` — IR → frozen ``Model`` factory.

See ``docs/04_PARSERS.md §3`` for the 10 responsibilities.
"""

from __future__ import annotations

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
from ..data_model.topology import build_children, build_subtrees, build_supports, topo_sort
from .ir import IRBody, IRJoint, IRModel, IRError

# ──────────────────────────────── constants ──────────────────────────────────

#: Parent body name that signals "connect to the universe (joint 0)".
_WORLD_SENTINEL = "world"

_EPS = 1e-6
_IDENTITY_SE3_VALS = [0., 0., 0., 0., 0., 0., 1.]


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
    """Validate the four topology invariants from ``docs/17_CONTRACTS.md §1.5``.

    Raises :class:`~better_robot.exceptions.ModelInconsistencyError` at
    build time so the caller never sees a ``Model`` in an inconsistent
    state.
    """
    # 1. Universe joint is rooted at -1.
    if parents[0] != -1:
        raise ModelInconsistencyError(
            f"parents[0] must be -1 (universe), got {parents[0]}"
        )
    # 2. Topologically sorted: parents[i] < i for i > 0.
    for i, p in enumerate(parents[1:], start=1):
        if p >= i:
            raise ModelInconsistencyError(
                f"parents[{i}]={p} >= {i}; topological sort violated"
            )
    # 3. sum(nqs) == nq, sum(nvs) == nv.
    if sum(nqs) != nq_total:
        raise ModelInconsistencyError(
            f"sum(nqs)={sum(nqs)} != nq={nq_total}"
        )
    if sum(nvs) != nv_total:
        raise ModelInconsistencyError(
            f"sum(nvs)={sum(nvs)} != nv={nv_total}"
        )
    # 4. Contiguous slicing: idx_qs[i] + nqs[i] == idx_qs[i+1] (and same for v).
    for i in range(len(nqs) - 1):
        if idx_qs[i] + nqs[i] != idx_qs[i + 1]:
            raise ModelInconsistencyError(
                f"idx_qs[{i}] + nqs[{i}] = {idx_qs[i] + nqs[i]} "
                f"!= idx_qs[{i+1}]={idx_qs[i+1]} (q-slicing gap)"
            )
        if idx_vs[i] + nvs[i] != idx_vs[i + 1]:
            raise ModelInconsistencyError(
                f"idx_vs[{i}] + nvs[{i}] = {idx_vs[i] + nvs[i]} "
                f"!= idx_vs[{i+1}]={idx_vs[i+1]} (v-slicing gap)"
            )

# ─────────────────────────────── helpers ─────────────────────────────────��───


def _axis_near(a: torch.Tensor | None, ref: tuple[float, float, float]) -> bool:
    """True if ``a`` is within ``_EPS`` of ``ref``."""
    if a is None:
        return False
    r = torch.tensor(ref, dtype=a.dtype, device=a.device)
    return bool((a - r).norm() < _EPS)


def _kind_to_joint_model(ir_joint: IRJoint) -> JointModel:
    """Select a concrete ``JointModel`` from an ``IRJoint``."""
    kind = ir_joint.kind
    axis = ir_joint.axis

    if kind in ("fixed", "world"):
        return JointFixed()

    if kind in ("revolute",):
        if axis is None or _axis_near(axis, (1., 0., 0.)):
            return JointRX()
        if _axis_near(axis, (0., 1., 0.)):
            return JointRY()
        if _axis_near(axis, (0., 0., 1.)):
            return JointRZ()
        return JointRevoluteUnaligned(axis=axis.float())

    if kind in ("continuous",):
        _ax = axis if axis is not None else torch.tensor([0., 0., 1.])
        return JointRevoluteUnbounded(axis=_ax.float())

    if kind in ("prismatic",):
        if axis is None or _axis_near(axis, (1., 0., 0.)):
            return JointPX()
        if _axis_near(axis, (0., 1., 0.)):
            return JointPY()
        if _axis_near(axis, (0., 0., 1.)):
            return JointPZ()
        return JointPrismaticUnaligned(axis=axis.float())

    if kind in ("spherical", "ball"):
        return JointSpherical()

    if kind in ("free_flyer", "free", "floating"):
        return JointFreeFlyer()

    if kind in ("planar",):
        return JointPlanar()

    if kind in ("translation",):
        return JointTranslation()

    raise IRError(f"Unknown joint kind {kind!r} for joint {ir_joint.name!r}")


def _pack_inertia(ir_body: IRBody, dtype: torch.dtype) -> torch.Tensor:
    """Pack ``IRBody`` inertia into [m, cx,cy,cz, Ixx,Iyy,Izz,Ixy,Ixz,Iyz]."""
    m = torch.tensor([float(ir_body.mass)], dtype=dtype)
    com = ir_body.com.to(dtype=dtype)
    I = ir_body.inertia.to(dtype=dtype)
    sym6 = torch.stack([I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2]])
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


# ──────────────────────────────── main factory ───────────────────────────────


def build_model(
    ir: IRModel,
    *,
    root_joint: JointModel | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Model:
    """Consume an ``IRModel`` and return a frozen ``Model``.

    Responsibilities (see docs/04_PARSERS.md §3):

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
    10. Return ``Model(frozen=True)``.

    **World-sentinel convention**: an ``IRJoint`` with ``parent_body="world"``
    signals that the joint connects to the universe (joint 0) directly.  That
    joint becomes model joint 1.  A synthetic ``JointFixed`` is *not* inserted;
    the IR joint's ``kind`` is used instead (unless ``root_joint`` overrides
    it).  This lets programmatic builders embed a ``JointFreeFlyer`` root
    without extra ``load(…, free_flyer=True)`` kwargs.
    """
    identity_se3 = torch.tensor(_IDENTITY_SE3_VALS, dtype=dtype)

    # ── 1. Identify root structure ────────────────────────────────────────────
    world_ir_idxs = [
        i for i, j in enumerate(ir.joints)
        if j.parent_body == _WORLD_SENTINEL
    ]

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
        _root_jm: JointModel = (
            root_joint if root_joint is not None
            else _kind_to_joint_model(world_ir_joint)
        )
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
                raise IRError(
                    f"Expected exactly 1 root body, found {len(candidates)}: {candidates}"
                )
            root_body = candidates[0]

        _root_jm = root_joint if root_joint is not None else JointFixed()
        _root_placement = identity_se3.clone()
        _root_name = "root_joint"
        regular_ir_indices = list(range(len(ir.joints)))

    # ── 2. Topological sort of regular IR joints ──────────────────────────────
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
    nqs_list = [jm.nq for jm in joint_models_list]
    nvs_list = [jm.nv for jm in joint_models_list]

    idx_qs_list: list[int] = []
    idx_vs_list: list[int] = []
    q_off = 0
    v_off = 0
    for nq_j, nv_j in zip(nqs_list, nvs_list):
        idx_qs_list.append(q_off)
        idx_vs_list.append(v_off)
        q_off += nq_j
        v_off += nv_j

    nq_total = q_off
    nv_total = v_off
    nqs = tuple(nqs_list)
    nvs = tuple(nvs_list)
    idx_qs = tuple(idx_qs_list)
    idx_vs = tuple(idx_vs_list)

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
            vel = ir_j.velocity_limit if ir_j is not None and ir_j.velocity_limit is not None else 0.
            eff = ir_j.effort_limit if ir_j is not None and ir_j.effort_limit is not None else 0.
            vel_lim.append(vel)
            eff_lim.append(eff)

        elif kind == "spherical":
            lower_pos.extend([-1.] * 4)
            upper_pos.extend([1.] * 4)
            v = ir_j.velocity_limit if ir_j is not None and ir_j.velocity_limit is not None else 0.
            e = ir_j.effort_limit if ir_j is not None and ir_j.effort_limit is not None else 0.
            vel_lim.extend([v] * 3)
            eff_lim.extend([e] * 3)

        elif kind == "planar":
            lower_pos.extend([-_INF, -_INF, -1., -1.])
            upper_pos.extend([_INF, _INF, 1., 1.])
            vel_lim.extend([0.] * 3)
            eff_lim.extend([0.] * 3)

        elif kind == "translation":
            lower_pos.extend([-_INF] * 3)
            upper_pos.extend([_INF] * 3)
            vel_lim.extend([0.] * 3)
            eff_lim.extend([0.] * 3)

        else:
            # 1-DOF: revolute_rx/ry/rz, revolute_unaligned, prismatic_*, helical, mimic
            if ir_j is not None:
                lo = ir_j.lower if ir_j.lower is not None else -_INF
                hi = ir_j.upper if ir_j.upper is not None else _INF
                vel = ir_j.velocity_limit if ir_j.velocity_limit is not None else 0.
                eff = ir_j.effort_limit if ir_j.effort_limit is not None else 0.
            else:
                lo, hi, vel, eff = -_INF, _INF, _INF, _INF
            lower_pos.append(lo)
            upper_pos.append(hi)
            vel_lim.append(vel)
            eff_lim.append(eff)

    lower_pos_limit = torch.tensor(lower_pos, dtype=dtype)
    upper_pos_limit = torch.tensor(upper_pos, dtype=dtype)
    velocity_limit = torch.tensor(vel_lim, dtype=dtype)
    effort_limit = torch.tensor(eff_lim, dtype=dtype)
    rotor_inertia = torch.zeros(nv_total, dtype=dtype)
    armature = torch.zeros(nv_total, dtype=dtype)
    friction = torch.zeros(nv_total, dtype=dtype)
    damping = torch.zeros(nv_total, dtype=dtype)

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
        frames_list.append(Frame(
            name=f"body_{bname}",
            parent_joint=midx,
            joint_placement=identity_se3.clone(),
            frame_type="body",
        ))
    for ir_frame in ir.frames:
        parent_midx = body_to_mjidx.get(ir_frame.parent_body, 1)
        frames_list.append(Frame(
            name=ir_frame.name,
            parent_joint=parent_midx,
            joint_placement=ir_frame.placement.to(dtype=dtype),
            frame_type=ir_frame.frame_type,  # type: ignore[arg-type]
        ))
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

    for offset, ir_ji in enumerate(sorted_ir_indices):
        ir_j = ir.joints[ir_ji]
        mjidx = offset + 2
        if ir_j.mimic_source is not None:
            src = ir_joint_name_to_mjidx.get(ir_j.mimic_source)
            if src is None:
                raise IRError(
                    f"Mimic source {ir_j.mimic_source!r} not found "
                    f"(referenced by joint {ir_j.name!r})"
                )
            mimic_src_list[mjidx] = src
            mimic_mult[mjidx] = ir_j.mimic_multiplier
            mimic_off[mjidx] = ir_j.mimic_offset

    mimic_source = tuple(mimic_src_list)

    # ── 15. q_neutral ─────────────────────────────────────────────────────────
    neutral_parts: list[torch.Tensor] = []
    for jm in joint_models_list:
        if jm.nq > 0:
            neutral_parts.append(jm.neutral())
    q_neutral = (
        torch.cat(neutral_parts, dim=-1).to(dtype=dtype)
        if neutral_parts
        else torch.zeros(nq_total, dtype=dtype)
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
        q_neutral = _dev(q_neutral)
        gravity = _dev(gravity)

    # ── 18. Enforce topology invariants (docs/17_CONTRACTS.md §1.5) ───────────
    _check_topology_invariants(
        parents=parents,
        nqs=nqs,
        nvs=nvs,
        idx_qs=idx_qs,
        idx_vs=idx_vs,
        nq_total=nq_total,
        nv_total=nv_total,
    )

    # ── 19. Return frozen Model ───────────────────────────────────────────────
    return Model(
        njoints=n_model_joints,
        nbodies=n_model_joints,
        nframes=nframes,
        nq=nq_total,
        nv=nv_total,
        name=ir.name,
        joint_names=joint_names,
        body_names=body_names,
        frame_names=frame_names,
        joint_name_to_id=joint_name_to_id,
        body_name_to_id=body_name_to_id,
        frame_name_to_id=frame_name_to_id,
        parents=parents,
        children=children,
        subtrees=subtrees,
        supports=supports,
        topo_order=topo_order,
        joint_models=joint_models,
        nqs=nqs,
        nvs=nvs,
        idx_qs=idx_qs,
        idx_vs=idx_vs,
        joint_placements=joint_placements,
        body_inertias=body_inertias,
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
        mimic_source=mimic_source,
        frames=frames,
        q_neutral=q_neutral,
        meta={"ir": ir},
    )
