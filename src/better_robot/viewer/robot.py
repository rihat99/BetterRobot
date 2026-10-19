"""Robot adapter: a ``Model`` and a configuration clip ``q (T, nq)`` become viewer layers.

The only viewer file that knows ``Model``. Forward kinematics runs once for the whole clip; the
layers receive numpy arrays and never touch the model again.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from ._math import colormap, rotate
from .layer import Layer
from .layers.frames import FramesLayer
from .layers.mesh import MeshSpec, RigidMeshLayer, SpheresLayer
from .layers.skeleton import SkeletonLayer, marker_extent, marker_of
from .theme import LIGHT, Theme

if TYPE_CHECKING:
    from ..collision.robot_spheres import RobotSpheres
    from ..data_model.model import Model
    from ..io.ir import IRGeom


class RobotView:
    """Layers for one robot and a clip ``q (T, nq)``; ``update(q)`` rewrites them in place.

    ``visual``/``collision`` draw the URDF/MJCF geometry as ``RigidMeshLayer``s; ``skeleton``
    draws joints and links, coloured and thickened by ``|tau| / effort_limit`` when ``tau (T, nv)`` is given;
    ``frames`` lists frame names or ids for a ``FramesLayer``; ``spheres`` draws a
    ``RobotSpheres`` collision model. Forward kinematics runs once per ``update``.
    """

    def __init__(
        self,
        model: Model,
        q: torch.Tensor,
        *,
        name: str = "robot",
        visual: bool = True,
        collision: bool = False,
        skeleton: bool = False,
        frames: Sequence[str | int] = (),
        tau: torch.Tensor | None = None,
        spheres: RobotSpheres | None = None,
        theme: Theme = LIGHT,
        opacity: float = 1.0,
    ) -> None:
        self.model = model
        self.frame_ids = [model.frame_id(frame) if isinstance(frame, str) else int(frame) for frame in frames]
        self.sphere_model = spheres
        # Typed layer slots (None when not drawn); ``update`` writes each one directly.
        self.mesh: RigidMeshLayer | None = None
        self.collision: RigidMeshLayer | None = None
        self.skeleton: SkeletonLayer | None = None
        self.frames: FramesLayer | None = None
        self.spheres: SpheresLayer | None = None
        joint_poses, frame_poses, centers = self._kinematics(q)
        if visual:
            meshes = robot_meshes(model, collision=False, default_color=theme.robot)
            if meshes:
                self.mesh = RigidMeshLayer(name, meshes, joint_poses, opacity=opacity)
        if collision:
            meshes = robot_meshes(model, collision=True, default_color=theme.force)
            if meshes:
                self.collision = RigidMeshLayer(f"{name} collision", meshes, joint_poses, opacity=0.4)
        if skeleton:
            self.skeleton = robot_skeleton(model, joint_poses, name=f"{name} skeleton", tau=tau, theme=theme)
        if self.frame_ids:
            self.frames = FramesLayer(f"{name} frames", frame_poses)
        if spheres is not None:
            radii = spheres.radii.detach().cpu().numpy()
            self.spheres = SpheresLayer(f"{name} spheres", centers, radii, colors=theme.torque)
        slots = (self.mesh, self.collision, self.skeleton, self.frames, self.spheres)
        self.layers: list[Layer] = [layer for layer in slots if layer is not None]

    def __iter__(self) -> Iterator[Layer]:
        return iter(self.layers)

    def update(self, q: torch.Tensor) -> None:
        """Recompute kinematics for ``q`` (same number of frames) into the existing layer arrays."""
        joint_poses, frame_poses, centers = self._kinematics(q)
        for mesh in (self.mesh, self.collision):
            if mesh is not None:
                mesh.body_poses[:] = joint_poses
        if self.skeleton is not None:
            self.skeleton.joints[:] = joint_poses[:, 1:, :3]
            self.skeleton.rotations[:] = joint_poses[:, 1:, 3:]
        if self.frames is not None:
            self.frames.poses[:] = frame_poses
        if self.spheres is not None:
            self.spheres.centers[:] = centers

    def _kinematics(self, q: torch.Tensor) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        from ..kinematics.state import forward_kinematics_state  # noqa: PLC0415 - keep the core import-light

        if q.ndim == 1:
            q = q[None]
        if q.ndim != 2:
            raise ValueError("q must be (T, nq) or (nq,)")
        with torch.no_grad():
            state = forward_kinematics_state(self.model, q)
            joint_poses = state.joint_pose.cpu().numpy().astype(np.float32)
            frame_poses = state.frame_pose[:, self.frame_ids].cpu().numpy() if self.frame_ids else None
            centers = self.sphere_model.centers(state).cpu().numpy() if self.sphere_model is not None else None
        return joint_poses, frame_poses, centers


def robot_layers(model: Model, q: torch.Tensor, **options: Any) -> list[Layer]:
    """Layers for a clip ``q (T, nq)`` (or one pose ``(nq,)``); see ``RobotView`` for the options."""
    return RobotView(model, q, **options).layers


_TORQUE_THICKNESS = (0.8, 1.5)  # a joint marker's cross-section, times its default, at zero load and at the limit


def robot_skeleton(
    model: Model,
    joint_poses: np.ndarray,
    *,
    name: str = "skeleton",
    tau: torch.Tensor | np.ndarray | None = None,
    theme: Theme = LIGHT,
    radius: float = 0.025,
    bone_radius: float | np.ndarray = 0.008,
    end_points: Mapping[str | int, Sequence[float]] | None = None,
) -> SkeletonLayer:
    """Skeleton over joints 1.. of ``joint_poses (T, njoints, 7)``, one marker per joint kind (``skeleton_spec``;
    a revolute joint is a cylinder of ``radius`` and length 2.4 ``radius``), laid out by ``spread_offsets``: markers
    that would overlap slide apart along their own axes, and bones join the drawn marker centres as axis-aligned legs
    (``SkeletonLayer``).

    ``tau`` ``(T, nv)`` or ``(nv,)`` shows each joint's load ``|tau| / effort_limit`` on its marker, in colour
    (``theme.torque_ramp`` up to the limit, ``theme.over_limit`` past it) and thickness (its cross-section from 0.8
    times the default at zero load to 1.5 times at the limit, its length unchanged); joints without a finite effort
    limit keep the plain look.

    ``bone_radius`` is one float or ``(njoints - 1,)``, entry ``j`` for the bone into joint ``j + 1``; 0 hides a
    bone, such as one to a massless sensor mount (``model.body_inertias[1:, 0] == 0``). ``end_points`` maps frames
    (names or ids) to a point ``(3,)`` in each frame: a bone, as thick as the one into the frame's parent joint, runs
    from that joint to the point, such as a hand out to where its contacts are, past the last wrist joint.
    """
    parents = np.asarray(model.parents[1:], dtype=np.int64) - 1  # the universe becomes -1
    kinds, axes = skeleton_spec(model)
    poses, offsets = joint_poses[:, 1:], spread_offsets(model, radius=radius)
    if end_points:  # each end point joins the skeleton as a joint without a marker
        ends, end_joints = _end_poses(model, joint_poses, end_points)
        count = len(end_joints)
        poses = np.concatenate([poses, ends], axis=1)
        parents = np.concatenate([parents, end_joints - 1])
        kinds += ["fixed"] * count
        axes = np.concatenate([axes, np.tile(np.float32([0.0, 0.0, 1.0]), (count, 1))])
        offsets = np.concatenate([offsets, np.zeros((count, 3), dtype=np.float32)])
        if np.ndim(bone_radius):
            bone_radius = np.concatenate([bone_radius, np.asarray(bone_radius)[end_joints - 1]])
    colors: Any = theme.joint
    thickness = None
    if tau is not None:
        ratio, limited = _torque_ratio(model, torch.as_tensor(tau).detach().cpu().numpy(), joint_poses.shape[0])
        extra = len(kinds) - len(limited)  # end points carry no load
        ratio, limited = np.pad(ratio, ((0, 0), (0, extra))), np.pad(limited, (0, extra))
        colors = _torque_colors(ratio, limited, theme)
        low, high = _TORQUE_THICKNESS
        thickness = np.where(limited, low + (high - low) * np.minimum(ratio, 1.0), 1.0)
    return SkeletonLayer(
        name,
        poses,
        parents,
        kinds=kinds,
        axes=axes,
        offsets=offsets,
        radius=radius,
        thickness=thickness,
        bone_radius=bone_radius,
        colors=colors,
        bone_color=theme.link,
    )


def _end_poses(
    model: Model, joint_poses: np.ndarray, end_points: Mapping[str | int, Sequence[float]]
) -> tuple[np.ndarray, np.ndarray]:
    """Poses ``(T, E, 7)`` of ``end_points`` (frame -> point in it) over ``joint_poses (T, njoints, 7)``, oriented as
    the frames' parent joints, and those joints ``(E,)``."""
    frames = [model.frames[model.frame_id(frame) if isinstance(frame, str) else int(frame)] for frame in end_points]
    joints = np.array([frame.parent_joint for frame in frames], dtype=np.int64)
    placement = np.stack([frame.joint_placement.detach().cpu().numpy() for frame in frames]).astype(np.float32)
    local = placement[:, :3] + rotate(placement[:, 3:], np.asarray(list(end_points.values()), dtype=np.float32))
    rotations = joint_poses[:, joints, 3:]
    return np.concatenate([joint_poses[:, joints, :3] + rotate(rotations, local), rotations], axis=-1), joints


def skeleton_spec(model: Model) -> tuple[list[str], np.ndarray]:
    """Joint kinds and axes ``(njoints - 1, 3)`` (joint frame; ``+z`` for joints without one) of joints 1..,
    for ``SkeletonLayer(kinds=, axes=)``."""
    # The packed table, not ``joint.axis``: JointRX/RY/RZ and PX/PY/PZ keep their axis in a ClassVar (axis None).
    axes = model.structure.joint_axes[1:].detach().cpu().numpy().astype(np.float32)
    axes[~np.any(axes, axis=-1)] = (0.0, 0.0, 1.0)  # zero rows: spherical, free-flyer, planar, fixed, ...
    return [joint.kind for joint in model.joint_models[1:]], axes


def spread_offsets(
    model: Model, *, radius: float = 0.025, gap: float = 0.5, reach: float = 6.0, step: float = 0.05
) -> np.ndarray:
    """Marker offsets ``(njoints - 1, 3)`` in each joint's frame that draw co-located joints apart, textbook style.

    Computed once at ``model.q_neutral`` for markers of the plain thickness (``robot_skeleton`` without ``tau``), so
    showing joint loads moves no marker (markers thickened by a large load may touch). Markers without an axis
    (spheres, the free-flyer triad, cubes) stay on their joint; then each cylinder or box slides along its own axis
    (so it stays on the true axis line) by the shortest multiple of ``step`` that leaves ``gap`` between it and every
    marker placed before it, at most ``reach`` (all in units of the joint ``radius``; a marker that cannot clear stays
    put). The order is a priority: joints with a marker further down their chain go first, leaves to root, then the
    chain ends (an ankle, a wrist), which so give way to the joints above them; an inner joint still slides to clear
    the markers placed before it.

    Of the two equally short slides a joint takes the one away from the root joint along its axis or, when its
    origin is within ``radius / 4`` of the plane through the root normal to the axis, the one along the axis's
    largest component in the root joint's frame. So on a robot mirror-symmetric through a coordinate plane of that
    frame (a humanoid's sagittal plane), mirror-image joints slide as mirror images whatever the signs of their
    axes, unless a marker must clear its own mirror image or the second rule meets an axis whose largest component
    is normal to that plane.
    """
    from ..kinematics.state import forward_kinematics_state  # noqa: PLC0415 - keep the core import-light

    kinds, axes = skeleton_spec(model)
    with torch.no_grad():
        poses = forward_kinematics_state(model, model.q_neutral).joint_pose[1:].cpu().numpy().astype(np.float64)
    origins = poses[:, :3]
    directions = rotate(poses[:, 3:], axes.astype(np.float64))
    # Tie sign: the axis's largest component in the root joint's frame positive (rounded, so mirror images agree).
    in_root = np.round(rotate(poses[0, 3:] * (-1.0, -1.0, -1.0, 1.0), directions), 6)
    tie_sign = np.sign(in_root[np.arange(len(in_root)), np.abs(in_root).argmax(axis=1)])
    markers = [marker_of(kind) for kind in kinds]
    extent = np.array([marker_extent(m) if m else (0.0, 0.0) for m in markers]) * radius  # (radius, half-length)
    # A marker counts as the capsule around its axis segment shortened by one radius at each end (a sphere when the
    # marker is no longer than it is wide): exact side by side, a cylinder's rim at most 0.41 radius outside it.
    extent[:, 1] = np.maximum(extent[:, 1] - extent[:, 0], 0.0)
    slides = np.zeros(len(kinds))
    axial = [marker in ("cylinder", "box") for marker in markers]
    inner = np.zeros(len(markers), dtype=bool)  # a marker further down the chain (not a chain end)
    for joint in reversed(range(len(markers))):
        parent = model.parents[joint + 1] - 1
        if parent >= 0:
            inner[parent] |= inner[joint] or markers[joint] is not None
    fixed = [joint for joint, marker in enumerate(markers) if marker and not axial[joint]]
    sliding = [joint for joint in reversed(range(len(markers))) if axial[joint] and inner[joint]]
    sliding += [joint for joint in reversed(range(len(markers))) if axial[joint] and not inner[joint]]
    placed: list[int] = []
    steps = np.arange(1, int(round(reach / step)) + 1) * step * radius
    for joint in fixed + sliding:
        if axial[joint] and placed:
            away = float((origins[joint] - origins[0]) @ directions[joint])
            sign = tie_sign[joint] if abs(away) <= radius / 4 else np.sign(away)
            candidates = np.concatenate([[0.0], np.stack([sign * steps, -sign * steps], axis=1).reshape(-1)])
            half = extent[joint, 1] * directions[joint]
            centres = origins[joint] + candidates[:, None] * directions[joint]  # (S, 3)
            others = np.asarray(placed)
            other_centres = origins[others] + slides[others, None] * directions[others]
            other_half = extent[others, 1, None] * directions[others]
            distance = _segment_distance(
                (centres - half)[:, None], (centres + half)[:, None],
                (other_centres - other_half)[None], (other_centres + other_half)[None],
            )  # fmt: skip
            clear = (distance - extent[joint, 0] - extent[others, 0] >= gap * radius).all(axis=1)
            slides[joint] = candidates[np.argmax(clear)] if clear.any() else 0.0
        placed.append(joint)
    return (slides[:, None] * axes).astype(np.float32)


def _segment_distance(p1: np.ndarray, q1: np.ndarray, p2: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Shortest distance between segments ``p1 q1`` and ``p2 q2`` (``(..., 3)``, broadcast; points allowed)."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e = (d1 * d1).sum(-1), (d2 * d2).sum(-1)
    b, c, f = (d1 * d2).sum(-1), (d1 * r).sum(-1), (d2 * r).sum(-1)
    tiny = 1e-12
    a_safe, e_safe = np.where(a > tiny, a, 1.0), np.where(e > tiny, e, 1.0)
    denom = a * e - b * b
    s = np.where(denom > tiny, np.clip((b * f - c * e) / np.where(denom > tiny, denom, 1.0), 0.0, 1.0), 0.0)
    s = np.where(e > tiny, s, np.where(a > tiny, np.clip(-c / a_safe, 0.0, 1.0), 0.0))
    t = np.where(e > tiny, (b * s + f) / e_safe, 0.0)
    s = np.where(t < 0.0, np.clip(-c / a_safe, 0.0, 1.0), np.where(t > 1.0, np.clip((b - c) / a_safe, 0.0, 1.0), s))
    s = np.where(a > tiny, s, 0.0)
    t = np.clip(t, 0.0, 1.0)
    return np.linalg.norm(p1 + d1 * s[..., None] - p2 - d2 * t[..., None], axis=-1)


def _torque_colors(ratio: np.ndarray, limited: np.ndarray, theme: Theme) -> np.ndarray:
    """``(T, J, 3)`` colours of the loads ``ratio (T, J)`` (``|tau| / effort_limit``): ``theme.torque_ramp`` up to 1,
    ``theme.over_limit`` past it, ``theme.joint`` on joints that are not ``limited (J,)``."""
    colors = np.where((ratio > 1.0)[..., None], np.asarray(theme.over_limit), colormap(ratio, *theme.torque_ramp))
    return np.where(limited[:, None], colors, np.asarray(theme.joint)).astype(np.float32)


def _torque_ratio(model: Model, tau: np.ndarray, nsteps: int) -> tuple[np.ndarray, np.ndarray]:
    """``|tau| / effort_limit`` ``(nsteps, njoints - 1)`` of joints 1.. for ``tau (nsteps or 1, nv)``, the largest
    over each joint's dofs, and ``(njoints - 1,)`` whether a joint has a dof with a finite positive limit."""
    tau = tau.reshape(-1, model.nv)
    if tau.shape[0] not in (1, nsteps):
        raise ValueError(f"tau must have {nsteps} frames or one, got {tau.shape[0]}")
    limits = model.values.effort_limit.detach().cpu().numpy().reshape(-1)
    # Dofs without a finite effort limit (free-flyer, unbounded URDF joints) contribute a ratio of 0.
    finite = np.isfinite(limits) & (limits > 0)
    limits = np.where(finite, limits, np.inf)
    ratio = np.zeros((tau.shape[0], model.njoints - 1), dtype=np.float32)
    limited = np.zeros(model.njoints - 1, dtype=bool)
    for joint, (start, count) in enumerate(zip(model.idx_vs[1:], model.nvs[1:])):
        if count:
            block = np.abs(tau[:, start : start + count]) / limits[start : start + count]
            ratio[:, joint] = block.max(axis=1)
            limited[joint] = finite[start : start + count].any()
    return np.broadcast_to(ratio, (nsteps, ratio.shape[1])).copy(), limited


def robot_meshes(model: Model, *, collision: bool = False, default_color: Any = (0.8, 0.8, 0.8)) -> list[MeshSpec]:
    """Mesh specs for the visual (or collision) geometry stored in ``model.meta["ir"]``.

    Mesh files resolve through ``model.meta["asset_resolver"]`` when present; unreadable files
    are skipped. Primitives are tessellated with trimesh.
    """
    ir = model.meta.get("ir")
    if ir is None:
        return []
    resolver = model.meta.get("asset_resolver")
    specs: list[MeshSpec] = []
    for ir_body in ir.bodies:
        body = model.body_name_to_id.get(ir_body.name)
        if body is None:
            continue
        for geom in ir_body.collision_geoms if collision else ir_body.visual_geoms:
            spec = _load_geom(geom, body, resolver, default_color)
            if spec is not None:
                specs.append(spec)
    return specs


def _load_geom(geom: IRGeom, body: int, resolver: Any, default_color: Any) -> MeshSpec | None:
    import trimesh  # noqa: PLC0415 - optional dependency, imported at the boundary

    origin = geom.origin.detach().cpu().numpy().astype(np.float32)
    color = tuple(geom.rgba[:3]) if geom.rgba is not None else tuple(default_color)
    kind, params = geom.kind, geom.params
    if kind == "mesh":
        path = params.get("path", "")
        if not path:
            return None
        if resolver is not None:
            try:
                path = str(resolver.resolve(path))
            except (FileNotFoundError, OSError):
                pass
        try:
            mesh = trimesh.load(path, force="mesh")
        except Exception:  # noqa: BLE001 - trimesh raises many loader-specific errors
            return None
        if not isinstance(mesh, trimesh.Trimesh) or mesh.faces.shape[0] == 0:
            return None
        mesh.apply_scale(params.get("scale", [1.0, 1.0, 1.0]))
        source = mesh if _has_own_colors(mesh) else None
    elif kind == "box":
        mesh, source = trimesh.creation.box(extents=params["size"]), None
    elif kind == "cylinder":
        mesh, source = trimesh.creation.cylinder(radius=params["radius"], height=params["length"]), None
    elif kind == "sphere":
        mesh, source = trimesh.creation.icosphere(subdivisions=2, radius=params["radius"]), None
    elif kind == "capsule":
        mesh, source = trimesh.creation.capsule(radius=params["radius"], height=params["length"]), None
    else:
        return None
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.uint32)
    return MeshSpec(vertices, faces, body, origin, color=color, trimesh=source)


def _has_own_colors(mesh: Any) -> bool:
    """True for textured meshes or meshes with more than one vertex/face colour (DAE, OBJ)."""
    kind = mesh.visual.kind
    if kind == "texture":
        return True
    if kind == "vertex":
        return len(np.unique(np.asarray(mesh.visual.vertex_colors), axis=0)) > 1
    if kind == "face":
        return len(np.unique(np.asarray(mesh.visual.face_colors), axis=0)) > 1
    return False
