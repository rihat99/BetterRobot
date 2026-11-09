"""Robot adapter: joint poses from one forward-kinematics pass and a ``Geometry`` become viewer layers.

The only viewer file that knows ``Model``. ``robot_meshes(geometry, root=)`` gives a ``RigidMeshLayer`` its meshes
and ``robot_skeleton(model, joint_poses)`` draws joints and bones; the layers receive numpy arrays and never touch
the model again.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from ..exceptions import ModelError, ShapeError
from ..model.joints import (
    JointFixed,
    JointFreeFlyer,
    JointHelical,
    JointPlanar,
    JointPrismatic,
    JointRevolute,
    JointRevoluteUnbounded,
    JointSpherical,
    JointTranslation,
)
from ..model.geometry import AttachedShape, Box, Cylinder, Geometry, Mesh, Sphere
from ._extras import needs_extra
from ._lie import act, inverse
from .layer import as_frames
from .layers.mesh import MeshSpec
from .layers.skeleton import SkeletonLayer, marker_extent, marker_of
from .theme import LIGHT, RGB, Theme, colormap

if TYPE_CHECKING:
    from ..model.model import Model


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
    """Skeleton over joints 1.. of ``joint_poses (T, njoints, 7)`` (one pose ``(njoints, 7)`` counts as ``T = 1``),
    one marker per joint kind (``skeleton_spec``; a revolute joint is a cylinder of ``radius`` and length 2.4
    ``radius``), laid out by ``spread_offsets``: markers that would overlap slide apart along their own axes, and
    bones join the drawn marker centres as axis-aligned legs (``SkeletonLayer``).

    ``tau`` ``(T, nv)`` or ``(nv,)`` shows each joint's load ``|tau| / effort_limit`` on its marker, in colour
    (``theme.torque_ramp`` up to the limit, ``theme.over_limit`` past it) and thickness (its cross-section from 0.8
    times the default at zero load to 1.5 times at the limit, its length unchanged); joints without a finite effort
    limit keep the plain look.

    ``bone_radius`` is one float or ``(njoints - 1,)``, entry ``j`` for the bone into joint ``j + 1``; 0 hides a
    bone, such as one to a massless sensor mount (``model.body_inertias[1:, 0] == 0``). ``end_points`` maps frames
    (names or ids) to a point ``(3,)`` in each frame: a bone, as thick as the one into the frame's parent joint, runs
    from that joint to the point, such as a hand out to where its contacts are, past the last wrist joint.
    """
    joint_poses = as_frames(joint_poses, 2, name="joint_poses", width=7)
    if joint_poses.shape[1] != model.njoints:
        raise ShapeError(
            f"joint_poses must be (T, njoints, 7) = (T, {model.njoints}, 7), one pose per joint of the model "
            f"(KinematicState.joint_poses), got {joint_poses.shape}"
        )
    parent_ids = np.asarray(model.parent_joint_ids[1:], dtype=np.int64) - 1  # the universe becomes -1
    kinds, axes = skeleton_spec(model)
    poses, offsets = joint_poses[:, 1:], spread_offsets(model, radius=radius)
    if end_points:  # each end point joins the skeleton as a joint without a marker
        ends, end_joints = _end_poses(model, joint_poses, end_points)
        count = len(end_joints)
        poses = np.concatenate([poses, ends], axis=1)
        parent_ids = np.concatenate([parent_ids, end_joints - 1])
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
        parent_ids=parent_ids,
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
    frame_ids = [model.frame_id(frame) for frame in end_points]
    joints = np.array([model.frame_parent_joint_ids[frame_id] for frame_id in frame_ids], dtype=np.int64)
    placement = model.frame_placements[..., frame_ids, :].detach().cpu().numpy().astype(np.float32)
    local = placement[:, :3] + act(placement[:, 3:], np.asarray(list(end_points.values()), dtype=np.float32))
    rotations = joint_poses[:, joints, 3:]
    return np.concatenate([joint_poses[:, joints, :3] + act(rotations, local), rotations], axis=-1), joints


_KIND_WORD = {
    JointRevolute: "revolute",
    JointRevoluteUnbounded: "continuous",
    JointPrismatic: "prismatic",
    JointHelical: "helical",
    JointSpherical: "spherical",
    JointFreeFlyer: "free_flyer",
    JointPlanar: "planar",
    JointTranslation: "translation",
    JointFixed: "fixed",
}


def skeleton_spec(model: Model) -> tuple[list[str], np.ndarray]:
    """Joint kind words and axes ``(njoints - 1, 3)`` (joint frame; ``+z`` for joints without one) of joints 1..,
    for ``SkeletonLayer(kinds=, axes=)``."""
    axes = np.array([joint.axis or (0.0, 0.0, 1.0) for joint in model.joints[1:]], dtype=np.float32).reshape(-1, 3)
    return [_KIND_WORD[type(joint)] for joint in model.joints[1:]], axes


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
    from ..kinematics import forward_kinematics  # noqa: PLC0415 - keep the core import-light

    kinds, axes = skeleton_spec(model)
    with torch.no_grad():
        poses = forward_kinematics(model, model.q_neutral).joint_poses[1:].cpu().numpy().astype(np.float64)
    origins = poses[:, :3]
    directions = act(poses[:, 3:], axes.astype(np.float64))
    # Tie sign: the axis's largest component in the root joint's frame positive (rounded, so mirror images agree).
    in_root = np.round(act(inverse(poses[0, 3:]), directions), 6)
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
        parent = model.parent_joint_ids[joint + 1] - 1
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
    if tau.ndim not in (1, 2) or tau.shape[-1] != model.nv:
        raise ShapeError(f"tau must be (T, nv) or (nv,) with nv = {model.nv}, got {tau.shape}")
    tau = tau.reshape(-1, model.nv)
    if tau.shape[0] not in (1, nsteps):
        raise ShapeError(f"tau must have {nsteps} frames or one, got {tau.shape[0]}")
    limits = model.effort_limit.detach().cpu().numpy().reshape(-1)
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


def robot_meshes(
    geometry: Geometry, *, root: Path | None = None, collision: bool = False, default_color: RGB = (0.8, 0.8, 0.8)
) -> list[MeshSpec]:
    """Mesh specs for the collision shapes of ``geometry`` or, by default, its display shapes (record 0016).

    A ``Mesh`` URI is read as written when absolute; ``package://<package>/<rel>`` and a relative path are read
    under ``root`` (the directory the description's URIs are relative to; the working directory when ``None``). A
    file that is missing or does not load raises ``ModelError`` naming the body and the URI. Primitives are
    tessellated with trimesh.
    """
    return [
        _mesh_spec(attached, root, default_color) for attached in geometry.shapes if attached.collision == collision
    ]


def _mesh_path(uri: str, root: Path | None) -> Path:
    """The file ``uri`` names: as written when absolute, else under ``root`` with a ``package://<package>/`` prefix
    stripped."""
    path = Path(uri.removeprefix("package://").partition("/")[2] if uri.startswith("package://") else uri)
    return path if path.is_absolute() or root is None else root / path


def _mesh_spec(attached: AttachedShape, root: Path | None, default_color: RGB) -> MeshSpec:
    with needs_extra("meshes"):
        import trimesh  # noqa: PLC0415 - optional dependency, imported at the boundary

    shape = attached.shape
    origin = np.asarray(shape.placement, dtype=np.float32)
    color = shape.color[:3] if shape.color is not None else tuple(default_color)
    if isinstance(shape, Mesh):
        path = _mesh_path(shape.uri, root)
        if not path.is_file():
            raise ModelError(f"body {attached.body}: mesh {shape.uri!r} is not a file at {path}")
        try:
            mesh = trimesh.load(path, force="mesh")
        except Exception as exc:  # noqa: BLE001 - trimesh raises many loader-specific errors
            raise ModelError(f"body {attached.body}: mesh {shape.uri!r} does not load: {exc}") from exc
        if not isinstance(mesh, trimesh.Trimesh) or mesh.faces.shape[0] == 0:
            raise ModelError(f"body {attached.body}: mesh {shape.uri!r} has no faces")
        mesh.apply_scale(shape.scale)
        source = mesh if _has_own_colors(mesh) else None
    elif isinstance(shape, Box):
        mesh, source = trimesh.creation.box(extents=shape.size), None
    elif isinstance(shape, Cylinder):
        mesh, source = trimesh.creation.cylinder(radius=shape.radius, height=shape.length), None
    elif isinstance(shape, Sphere):
        mesh, source = trimesh.creation.icosphere(subdivisions=2, radius=shape.radius), None
    else:
        mesh, source = trimesh.creation.capsule(radius=shape.radius, height=shape.length), None
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.uint32)
    return MeshSpec(vertices, faces, body=attached.body, origin=origin, color=color, trimesh=source)


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
