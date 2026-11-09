"""``build_model(ir, root_joint=...)``: an ``IRModel`` becomes one frozen ``Model`` through named stages.

The stages run in order and each returns a named result: the root (`_root`), the joint order (`_joint_order`: a
Kahn sort, parents first, that keeps the file's order wherever it is already topological), the topology
(`_topology`), the coordinate layout (`_layout`), the limits (`_limits`), the inertias (`_inertias`), the frames
(`_frames`: one per body named after it, then the IR frames), the reduced mimic coordinates (`_mimic`) and the
neutral configuration (`_neutral`, clamped into the limits). One ``Model(...)`` call takes them all.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import final

import torch

from .. import _internal_checks
from ..exceptions import ModelError
from ..lie import se3
from ..model.joints import JointFixed, JointHelical, JointModel, JointPrismatic, JointRevolute
from ..model.manifolds import CIRCLE, PLANE, RIGID, ROTATION, JointManifold
from ..model.model import Model
from .ir import IRBody, IRJoint, IRModel

#: Parent body name that connects a joint to the universe (joint 0) directly.
_WORLD_SENTINEL = "world"
_SUPPORTED_MIMIC_JOINTS = (JointRevolute, JointPrismatic, JointHelical)
_INF = float("inf")
#: Position bounds a manifold puts on its coordinates (a quaternion lies in ``[-1, 1]``); a joint on any other
#: manifold takes its IR joint's ``lower`` and ``upper``.
_MANIFOLD_BOUNDS: dict[JointManifold, tuple[list[float], list[float]]] = {
    CIRCLE: ([-_INF] * 2, [_INF] * 2),
    ROTATION: ([-1.0] * 4, [1.0] * 4),
    PLANE: ([-_INF, -_INF, -1.0, -1.0], [_INF, _INF, 1.0, 1.0]),
    RIGID: ([-_INF] * 7, [_INF] * 7),
}


@final
@dataclass(frozen=True, slots=True)
class _Root:
    """Joint 1 of the model: the body under the universe and the joint that holds it there."""

    body: str
    joint: JointModel
    name: str
    ir_joint: IRJoint | None  #: the IR joint from ``"world"`` to the root body, when the description wrote one
    regular: tuple[int, ...]  #: IR indices of every other joint


@final
@dataclass(frozen=True, slots=True)
class _Topology:
    """The tree in model order: joint ``i`` moves body ``i`` and hangs under joint ``parent_joint_ids[i]``."""

    joint_names: tuple[str, ...]
    body_names: tuple[str, ...]
    parent_joint_ids: tuple[int, ...]
    joints: tuple[JointModel, ...]
    ir_joints: tuple[
        IRJoint | None, ...
    ]  #: the IR joint behind each model joint; the universe and a synthetic root have none
    joint_placements: torch.Tensor  #: ``(njoints, 7)``


@final
@dataclass(frozen=True, slots=True)
class _Layout:
    """Width and start of each joint's slice of ``q`` and ``v``, and the totals."""

    nqs: tuple[int, ...]
    nvs: tuple[int, ...]
    idx_qs: tuple[int, ...]
    idx_vs: tuple[int, ...]
    nq: int
    nv: int


@final
@dataclass(frozen=True, slots=True)
class _Limits:
    """Per-coordinate limits and joint dynamics over every joint's coordinates: ``(nq_full,)`` and ``(nv_full,)``."""

    lower: torch.Tensor
    upper: torch.Tensor
    velocity: torch.Tensor
    effort: torch.Tensor
    armature: torch.Tensor
    damping: torch.Tensor


@final
@dataclass(frozen=True, slots=True)
class _Frames:
    """Every frame: one per body first, named after it and at the joint's origin, then the IR frames."""

    names: tuple[str, ...]
    parent_joint_ids: tuple[int, ...]
    placements: torch.Tensor  #: ``(nframes, 7)``


@final
@dataclass(frozen=True, slots=True)
class _MimicReduction:
    """The public layout once mimic targets are folded into their sources, and the reduced limits.

    ``source_ids``, ``multipliers`` and ``offsets`` hold each joint's resolved chain: the public root joint, the
    total scale and the total offset (a joint that is not a mimic target maps to itself with 1 and 0).
    """

    layout: _Layout
    source_ids: tuple[int, ...]
    multipliers: tuple[float, ...]
    offsets: tuple[float, ...]
    limits: _Limits


def _root(ir: IRModel, root_joint: JointModel | None) -> _Root:
    """The root body and its joint: the IR joint from ``"world"`` when there is one, else a fixed ``root_joint``."""
    world = [index for index, joint in enumerate(ir.joints) if joint.parent_body == _WORLD_SENTINEL]
    if len(world) > 1:
        names = [ir.joints[index].name for index in world]
        raise ModelError(f"Multiple joints with parent_body='world' is not supported; found: {names}")
    if world:
        ir_joint = ir.joints[world[0]]
        return _Root(
            body=ir_joint.child_body,
            joint=root_joint if root_joint is not None else ir_joint.joint,
            name=ir_joint.name,
            ir_joint=ir_joint,
            regular=tuple(index for index in range(len(ir.joints)) if index != world[0]),
        )
    body_names = [body.name for body in ir.bodies]
    if ir.root_body:
        if ir.root_body not in body_names:
            raise ModelError(f"root_body {ir.root_body!r} not found in IR bodies")
        body = ir.root_body
    else:
        children = {joint.child_body for joint in ir.joints}
        candidates = [name for name in body_names if name not in children]
        if len(candidates) != 1:
            raise ModelError(f"Expected exactly 1 root body, found {len(candidates)}: {candidates}")
        body = candidates[0]
    return _Root(
        body=body,
        joint=root_joint if root_joint is not None else JointFixed(),
        name="root_joint",
        ir_joint=None,
        regular=tuple(range(len(ir.joints))),
    )


def _joint_order(ir: IRModel, root_body: str, indices: tuple[int, ...]) -> tuple[int, ...]:
    """The IR indices of ``indices`` parents first (Kahn); among the joints ready at once the lowest IR index goes
    first, so a description whose order is already topological keeps it exactly."""
    by_parent: dict[str, list[int]] = {}
    for index in indices:
        by_parent.setdefault(ir.joints[index].parent_body, []).append(index)
    ready = list(by_parent.get(root_body, ()))
    heapq.heapify(ready)
    order: list[int] = []
    reached = {root_body}
    while ready:
        index = heapq.heappop(ready)
        child = ir.joints[index].child_body
        if child in reached:
            raise ModelError(f"Cycle detected: body {child!r} reachable via multiple paths")
        reached.add(child)
        order.append(index)
        for next_index in by_parent.get(child, ()):
            heapq.heappush(ready, next_index)
    if len(order) != len(indices):
        placed = set(order)
        names = [ir.joints[index].name for index in indices if index not in placed]
        raise ModelError(f"Disconnected joints (not reachable from root): {names}")
    return tuple(order)


def _unique(kinds: str, names: list[str]) -> None:
    """Raise ``ModelError`` at the first of ``names`` used twice: names are how a model is addressed."""
    seen: set[str] = set()
    for name in names:
        if name in seen:
            given = " (build_model names joint 0 and body 0 'universe', and an added root joint 'root_joint')"
            hint = given if name in ("universe", "root_joint") else ""
            raise ModelError(f"two {kinds} are named {name!r}{hint}; rename one")
        seen.add(name)


def _topology(ir: IRModel, root: _Root, order: tuple[int, ...], dtype: torch.dtype) -> _Topology:
    """Joint 0 is the universe, joint 1 the root, then the IR joints in ``order``; a body is named by its joint.
    Joint names and body names are each unique, ``"universe"`` included."""
    _unique("bodies", ["universe", *(body.name for body in ir.bodies)])
    identity = se3.identity(dtype=dtype)
    joint_names = ["universe", root.name]
    body_names = ["universe", root.body]
    parent_joint_ids = [-1, 0]
    joints: list[JointModel] = [JointFixed(), root.joint]
    ir_joints: list[IRJoint | None] = [None, root.ir_joint]
    placements = [identity, root.ir_joint.origin.to(dtype=dtype) if root.ir_joint is not None else identity]
    joint_of_body = {root.body: 1}
    for index in order:
        joint = ir.joints[index]
        joint_of_body[joint.child_body] = len(joint_names)
        joint_names.append(joint.name)
        body_names.append(joint.child_body)
        parent_joint_ids.append(joint_of_body[joint.parent_body])
        joints.append(joint.joint)
        ir_joints.append(joint)
        placements.append(joint.origin.to(dtype=dtype))
    _unique("joints", joint_names)
    return _Topology(
        joint_names=tuple(joint_names),
        body_names=tuple(body_names),
        parent_joint_ids=tuple(parent_joint_ids),
        joints=tuple(joints),
        ir_joints=tuple(ir_joints),
        joint_placements=torch.stack(placements, dim=0),
    )


def _layout(nqs: tuple[int, ...], nvs: tuple[int, ...]) -> _Layout:
    """Start indices by accumulating the widths."""
    idx_qs, nq = _cumulative(nqs)
    idx_vs, nv = _cumulative(nvs)
    return _Layout(nqs=nqs, nvs=nvs, idx_qs=idx_qs, idx_vs=idx_vs, nq=nq, nv=nv)


def _cumulative(widths: tuple[int, ...]) -> tuple[tuple[int, ...], int]:
    starts: list[int] = []
    offset = 0
    for width in widths:
        starts.append(offset)
        offset += width
    return tuple(starts), offset


def _joint_limits(
    joint: JointModel, ir_joint: IRJoint | None
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Lower, upper, velocity and effort limits of one joint's coordinates. The joint's manifold bounds its
    coordinates where it has bounds, else the IR joint's ``lower`` and ``upper`` do. A joint no IR joint describes
    (a synthetic root) and a free flyer move unlimited; an IR joint's scalars fill the rest, 0.0 where it has none."""
    if ir_joint is None or joint.manifold is RIGID:
        velocity = effort = _INF
    else:
        velocity = ir_joint.velocity_limit if ir_joint.velocity_limit is not None else 0.0
        effort = ir_joint.effort_limit if ir_joint.effort_limit is not None else 0.0
    if joint.manifold in _MANIFOLD_BOUNDS:
        lower, upper = _MANIFOLD_BOUNDS[joint.manifold]
    else:
        if ir_joint is not None and ir_joint.lower is not None and ir_joint.upper is not None:
            if ir_joint.lower > ir_joint.upper:
                raise ModelError(
                    f"joint {ir_joint.name!r}: lower limit {ir_joint.lower} > upper limit {ir_joint.upper}; swap them"
                )
        lower = [ir_joint.lower if ir_joint is not None and ir_joint.lower is not None else -_INF] * joint.nq
        upper = [ir_joint.upper if ir_joint is not None and ir_joint.upper is not None else _INF] * joint.nq
    return lower, upper, [velocity] * joint.nv, [effort] * joint.nv


def _limits(topology: _Topology, dtype: torch.dtype) -> _Limits:
    """Limits, armature and damping over every joint's coordinates (mimic targets included)."""
    lower: list[float] = []
    upper: list[float] = []
    velocity: list[float] = []
    effort: list[float] = []
    armature: list[float] = []
    damping: list[float] = []
    for joint, behind in zip(topology.joints, topology.ir_joints):
        # A root_joint= that replaced the description's world joint keeps only that joint's placement.
        ir_joint = behind if behind is not None and behind.joint is joint else None
        joint_lower, joint_upper, joint_velocity, joint_effort = _joint_limits(joint, ir_joint)
        lower.extend(joint_lower)
        upper.extend(joint_upper)
        velocity.extend(joint_velocity)
        effort.extend(joint_effort)
        # One scalar per IR joint applies to each of its velocity coordinates, like the limits above.
        armature.extend([ir_joint.armature if ir_joint is not None else 0.0] * joint.nv)
        damping.extend([ir_joint.damping if ir_joint is not None else 0.0] * joint.nv)
    return _Limits(
        lower=torch.tensor(lower, dtype=dtype),
        upper=torch.tensor(upper, dtype=dtype),
        velocity=torch.tensor(velocity, dtype=dtype),
        effort=torch.tensor(effort, dtype=dtype),
        armature=torch.tensor(armature, dtype=dtype),
        damping=torch.tensor(damping, dtype=dtype),
    )


def _pack_inertia(body: IRBody, dtype: torch.dtype) -> torch.Tensor:
    """``[m, cx, cy, cz, Ixx, Iyy, Izz, Ixy, Ixz, Iyz]`` of one body."""
    mass = torch.tensor([float(body.mass)], dtype=dtype)
    inertia = body.inertia.to(dtype=dtype)
    upper = torch.stack([inertia[0, 0], inertia[1, 1], inertia[2, 2], inertia[0, 1], inertia[0, 2], inertia[1, 2]])
    return torch.cat([mass, body.com.to(dtype=dtype), upper])


def _inertias(ir: IRModel, body_names: tuple[str, ...], dtype: torch.dtype) -> torch.Tensor:
    """Packed inertia of every body, ``(nbodies, 10)``; the universe weighs nothing."""
    bodies = {body.name: body for body in ir.bodies}
    rows = [torch.zeros(10, dtype=dtype)]
    for name in body_names[1:]:
        if name not in bodies:
            raise ModelError(
                f"body {name!r} is moved by a joint but is not among the IR bodies; add an IRBody of that name"
            )
        rows.append(_pack_inertia(bodies[name], dtype))
    return torch.stack(rows, dim=0)


def _frames(ir: IRModel, topology: _Topology, dtype: torch.dtype) -> _Frames:
    """A frame per body, named after it and at its joint's origin, then the IR frames with their own names."""
    joint_of_body = {name: index for index, name in enumerate(topology.body_names)}
    names = list(topology.body_names)
    parent_joint_ids = list(range(len(names)))
    placements = [se3.identity(dtype=dtype)] * len(names)
    taken = set(names)
    for frame in ir.frames:
        if frame.name in taken:
            raise ModelError(f"frame {frame.name!r} is already the name of a body or of another frame; rename one")
        if frame.parent_body not in joint_of_body:
            raise ModelError(f"frame {frame.name!r} is attached to unknown body {frame.parent_body!r}")
        taken.add(frame.name)
        names.append(frame.name)
        parent_joint_ids.append(joint_of_body[frame.parent_body])
        placements.append(frame.placement.to(dtype=dtype))
    return _Frames(
        names=tuple(names), parent_joint_ids=tuple(parent_joint_ids), placements=torch.stack(placements, dim=0)
    )


def _mimic(topology: _Topology, layout: _Layout, limits: _Limits) -> _MimicReduction:  # noqa: PLR0912, PLR0915 - one build-time policy boundary
    """Resolve every mimic chain to its public root, drop the targets from the public layout, reduce the limits."""
    joint_names = topology.joint_names
    joint_of_name = {joint.name: index for index, joint in enumerate(topology.ir_joints) if joint is not None}
    njoints = len(topology.joints)
    source_of = list(range(njoints))
    multiplier_of = [1.0] * njoints
    offset_of = [0.0] * njoints
    targets: set[int] = set()
    for index, ir_joint in enumerate(topology.ir_joints):
        if ir_joint is None or ir_joint.mimic_source is None:
            continue
        source = joint_of_name.get(ir_joint.mimic_source)
        if source is None:
            raise ModelError(
                f"Mimic source {ir_joint.mimic_source!r} not found (referenced by joint {ir_joint.name!r})"
            )
        targets.add(index)
        source_of[index] = source
        multiplier_of[index] = float(ir_joint.mimic_multiplier)
        offset_of[index] = float(ir_joint.mimic_offset)

    for target in targets:
        if source_of[target] == target:
            raise ModelError(f"Mimic cycle detected: joint {joint_names[target]!r} references itself")
        for role, index in (("target", target), ("source", source_of[target])):
            if not isinstance(topology.joints[index], _SUPPORTED_MIMIC_JOINTS):
                raise ModelError(
                    f"Mimic {role} joint {joint_names[index]!r} has unsupported kind "
                    f"{type(topology.joints[index]).__name__}; reduced mimic coordinates currently require concrete "
                    "scalar Euclidean revolute, prismatic, or helical joints (nq=nv=1)"
                )

    public = _layout(
        tuple(0 if index in targets else width for index, width in enumerate(layout.nqs)),
        tuple(0 if index in targets else width for index, width in enumerate(layout.nvs)),
    )

    state = [0] * njoints
    resolved: list[tuple[int, float, float] | None] = [None] * njoints

    def resolve(index: int, path: tuple[int, ...] = ()) -> tuple[int, float, float]:
        if index not in targets:
            return index, 1.0, 0.0
        if state[index] == 1:
            names = " -> ".join(joint_names[step] for step in (*path, index))
            raise ModelError(f"Mimic cycle detected: {names}")
        if state[index] == 2:
            result = resolved[index]
            assert result is not None
            return result
        state[index] = 1
        root, source_scale, source_offset = resolve(source_of[index], (*path, index))
        result = (root, multiplier_of[index] * source_scale, multiplier_of[index] * source_offset + offset_of[index])
        resolved[index] = result
        state[index] = 2
        return result

    chains = [resolve(index) for index in range(njoints)]

    lower = limits.lower.new_empty((public.nq,))
    upper = limits.upper.new_empty((public.nq,))
    velocity = limits.velocity.new_empty((public.nv,))
    effort = limits.effort.new_empty((public.nv,))
    armature = limits.armature.new_empty((public.nv,))
    damping = limits.damping.new_empty((public.nv,))
    for index, (nq_joint, nv_joint) in enumerate(zip(layout.nqs, layout.nvs)):
        if index in targets:
            continue
        full_q, full_v = layout.idx_qs[index], layout.idx_vs[index]
        public_q, public_v = public.idx_qs[index], public.idx_vs[index]
        lower[public_q : public_q + nq_joint] = limits.lower[full_q : full_q + nq_joint]
        upper[public_q : public_q + nq_joint] = limits.upper[full_q : full_q + nq_joint]
        velocity[public_v : public_v + nv_joint] = limits.velocity[full_v : full_v + nv_joint]
        effort[public_v : public_v + nv_joint] = limits.effort[full_v : full_v + nv_joint]
        armature[public_v : public_v + nv_joint] = limits.armature[full_v : full_v + nv_joint]
        damping[public_v : public_v + nv_joint] = limits.damping[full_v : full_v + nv_joint]

    for target in targets:
        root, scale, offset = resolve(target)
        full_q, full_v = layout.idx_qs[target], layout.idx_vs[target]
        public_q, public_v = public.idx_qs[root], public.idx_vs[root]
        target_lower = float(limits.lower[full_q])
        target_upper = float(limits.upper[full_q])
        if scale == 0.0:
            if offset < target_lower or offset > target_upper:
                raise ModelError(
                    f"Constant mimic joint {joint_names[target]!r} has offset {offset}, outside its position limits "
                    f"[{target_lower}, {target_upper}]"
                )
            continue
        mapped_a = (target_lower - offset) / scale
        mapped_b = (target_upper - offset) / scale
        lower[public_q] = torch.maximum(lower[public_q], lower.new_tensor(min(mapped_a, mapped_b)))
        upper[public_q] = torch.minimum(upper[public_q], upper.new_tensor(max(mapped_a, mapped_b)))
        if float(lower[public_q]) > float(upper[public_q]):
            raise ModelError(
                f"Mimic limits for source joint {joint_names[root]!r} are empty after applying target "
                f"{joint_names[target]!r}"
            )
        absolute_scale = abs(scale)
        velocity[public_v] = torch.minimum(velocity[public_v], limits.velocity[full_v] / absolute_scale)
        effort[public_v] = effort[public_v] + absolute_scale * limits.effort[full_v]
        armature[public_v] = armature[public_v] + scale**2 * limits.armature[full_v]
        damping[public_v] = damping[public_v] + scale**2 * limits.damping[full_v]

    return _MimicReduction(
        layout=public,
        source_ids=tuple(root for root, _, _ in chains),
        multipliers=tuple(scale for _, scale, _ in chains),
        offsets=tuple(offset for _, _, offset in chains),
        limits=_Limits(lower=lower, upper=upper, velocity=velocity, effort=effort, armature=armature, damping=damping),
    )


def _neutral(topology: _Topology, public: _Layout, limits: _Limits, dtype: torch.dtype) -> torch.Tensor:
    """Each public joint's neutral coordinates (its IR joint's ``neutral``, else the kind's), clamped into the limits
    so the neutral pose is a valid one, ``(nq,)``."""
    parts: list[torch.Tensor] = []
    for joint, behind, width in zip(topology.joints, topology.ir_joints, public.nqs):
        if width == 0:
            continue
        # A root_joint= that replaced the description's world joint starts from its own neutral.
        given = behind.neutral if behind is not None and behind.joint is joint else None
        if given is not None and tuple(given.shape) != (joint.nq,):
            raise ModelError(f"joint {behind.name!r}: neutral has shape {tuple(given.shape)}, expected ({joint.nq},)")
        parts.append((joint.neutral() if given is None else given).to(dtype=dtype))
    neutral = torch.cat(parts, dim=-1) if parts else torch.zeros(0, dtype=dtype)
    return torch.clamp(neutral, limits.lower, limits.upper)


def _check_topology_invariants(
    *,
    parent_joint_ids: tuple[int, ...],
    nqs: tuple[int, ...],
    nvs: tuple[int, ...],
    idx_qs: tuple[int, ...],
    idx_vs: tuple[int, ...],
    nq_total: int,
    nv_total: int,
) -> None:
    """The four topology invariants of a ``Model``, checked at build time so no caller sees a broken record."""
    if parent_joint_ids[0] != -1:
        raise ModelError(f"parent_joint_ids[0] must be -1 (universe), got {parent_joint_ids[0]}")
    for joint, parent in enumerate(parent_joint_ids[1:], start=1):
        if parent >= joint:
            raise ModelError(f"parent_joint_ids[{joint}]={parent} >= {joint}; topological sort violated")
    if sum(nqs) != nq_total:
        raise ModelError(f"sum(nqs)={sum(nqs)} != nq={nq_total}")
    if sum(nvs) != nv_total:
        raise ModelError(f"sum(nvs)={sum(nvs)} != nv={nv_total}")
    for i in range(len(nqs) - 1):
        if idx_qs[i] + nqs[i] != idx_qs[i + 1]:
            raise ModelError(
                f"idx_qs[{i}] + nqs[{i}] = {idx_qs[i] + nqs[i]} != idx_qs[{i + 1}]={idx_qs[i + 1]} (q-slicing gap)"
            )
        if idx_vs[i] + nvs[i] != idx_vs[i + 1]:
            raise ModelError(
                f"idx_vs[{i}] + nvs[{i}] = {idx_vs[i] + nvs[i]} != idx_vs[{i + 1}]={idx_vs[i + 1]} (v-slicing gap)"
            )


def build_model(
    ir: IRModel,
    *,
    root_joint: JointModel | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Model:
    """The frozen ``Model`` of ``ir``, built on ``device`` in ``dtype`` (record 0015).

    The root body is the child of the one IR joint whose parent is ``"world"``, else the one body no joint moves
    (or ``ir.root_body``); ``root_joint`` replaces the joint that holds it under the universe (a ``JointFixed``
    when the description wrote none). Joints follow the description's order wherever it is topological; a body is
    named by the joint that moves it and gets a frame of its own name. The neutral configuration is clamped into
    the limits. A description that cannot be a robot raises ``ModelError``.
    """
    root = _root(ir, root_joint)
    topology = _topology(ir, root, _joint_order(ir, root.body, root.regular), dtype)
    full = _layout(tuple(joint.nq for joint in topology.joints), tuple(joint.nv for joint in topology.joints))
    reduction = _mimic(topology, full, _limits(topology, dtype))
    frames = _frames(ir, topology, dtype)
    public = reduction.layout
    for layout in (public, full):
        _check_topology_invariants(
            parent_joint_ids=topology.parent_joint_ids,
            nqs=layout.nqs,
            nvs=layout.nvs,
            idx_qs=layout.idx_qs,
            idx_vs=layout.idx_vs,
            nq_total=layout.nq,
            nv_total=layout.nv,
        )
    model = Model(
        name=ir.name,
        joint_names=topology.joint_names,
        body_names=topology.body_names,
        frame_names=frames.names,
        joints=topology.joints,
        parent_joint_ids=topology.parent_joint_ids,
        frame_parent_joint_ids=frames.parent_joint_ids,
        nqs=public.nqs,
        nvs=public.nvs,
        idx_qs=public.idx_qs,
        idx_vs=public.idx_vs,
        nqs_full=full.nqs,
        nvs_full=full.nvs,
        idx_qs_full=full.idx_qs,
        idx_vs_full=full.idx_vs,
        mimic_source_ids=reduction.source_ids,
        mimic_multipliers=reduction.multipliers,
        mimic_offsets=reduction.offsets,
        joint_placements=topology.joint_placements,
        body_inertias=_inertias(ir, topology.body_names, dtype),
        frame_placements=frames.placements,
        lower_position_limit=reduction.limits.lower,
        upper_position_limit=reduction.limits.upper,
        velocity_limit=reduction.limits.velocity,
        effort_limit=reduction.limits.effort,
        armature=reduction.limits.armature,
        damping=reduction.limits.damping,
        gravity=ir.gravity.to(dtype=dtype),
        q_neutral=_neutral(topology, public, reduction.limits, dtype),
    )
    if _internal_checks.ENABLED:
        _internal_checks.check_unit_quaternions(model.joint_placements, name="joint_placements")
        _internal_checks.check_unit_quaternions(model.frame_placements, name="frame_placements")
    return model if device is None else model.to(device=device)
