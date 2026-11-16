"""``RobotVisual``: one robot in the viewer, as its meshes or as a skeleton, with a collision overlay and frames.

```python
import better_robot as br
from better_robot.viewer import Viewer
from better_robot.viewer.robot import RobotVisual

model = br.load("panda.urdf")
viewer = Viewer()
robot = viewer.add(RobotVisual(model, style="maximal"))
robot.update(model.q_neutral)  # a new configuration, or a clip (T, nq) of as many steps
viewer.serve()
```

The older adapter functions are still here until their callers move to ``RobotVisual``. ``robot_meshes(geometry, root=)`` gives a ``RigidMeshLayer`` its meshes
and ``robot_skeleton(model, joint_poses)`` draws joints and bones; the layers receive numpy arrays and never touch
the model again.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, final, override

import numpy as np
import torch

from ..collision import RobotSpheres
from ..exceptions import ArgumentError, ModelError, ShapeError, UnknownNameError
from ..kinematics import forward_kinematics
from ..lie import se3
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
from ._lie import act, inverse, to_wxyz
from .layer import Bounds, Layer, as_frames, box
from .layers.frames import FramesLayer
from .layers.mesh import MeshSpec, RigidMeshLayer, SpheresLayer
from .layers.skinned import SkinnedMeshLayer
from .layers.skeleton import SkeletonLayer, marker_extent, marker_of
from .sidebar import Sidebar
from .skeleton import MAXIMAL, MINIMAL, Skeleton, SkeletonRecipe, build_skeleton, unit_ball, unit_box, unit_cylinder
from .theme import LIGHT, RGB, Theme, colormap, rgb_uint8

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
        axes = np.concatenate([axes, np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (count, 1))])
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
    for ``SkeletonLayer(kinds=, axes=)``; a joint kind defined outside the library is ``"spherical"`` (a sphere)."""
    axes = np.array([joint.axis or (0.0, 0.0, 1.0) for joint in model.joints[1:]], dtype=np.float32).reshape(-1, 3)
    return [_KIND_WORD.get(type(joint), "spherical") for joint in model.joints[1:]], axes


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
                (centres - half)[:, None],
                (centres + half)[:, None],
                (other_centres - other_half)[None],
                (other_centres + other_half)[None],
            )
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
    color = shape.color[:3] if shape.color is not None else default_color
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


# ------------------------------------------------------------------------------------------------ the robot visual

Style = Literal["meshes", "minimal", "maximal"]

_STYLES = ("meshes", "minimal", "maximal")
_OVERLAY = 0.35  # opacity of the collision overlay
#: How far a part's colour moves from the robot's hue, towards black (negative) or white (positive). ``Theme.shade``
#: takes these values over; until it has them the robot visual shades its own blocks.
_SHADES = {"joint": -0.22, "face": -0.58, "body": 0.0, "stick": 0.25}


def _shade(hue: RGB, role: str) -> RGB:
    mix, target = abs(_SHADES[role]), float(_SHADES[role] > 0.0)
    red, green, blue = (channel + mix * (target - channel) for channel in hue)
    return red, green, blue


@final
@dataclass(frozen=True)
class Skin:
    """A skinned surface over a model's joints: ``vertices (V, 3)`` in the world at ``bind``, ``faces (F, 3)`` and
    ``weights (V, njoints)``; ``bind (nq,)`` is the configuration of the rest surface, ``None`` for ``q_neutral``."""

    vertices: np.ndarray
    faces: np.ndarray
    weights: np.ndarray
    bind: np.ndarray | None = None


@final
class _Blocks(Layer):
    """The nodes of one skeleton: a batched node per unit mesh in use and a line node for limit arcs."""

    def __init__(self, name: str, skeleton: Skeleton, poses: np.ndarray, hue: RGB) -> None:
        super().__init__(name, poses.shape[0])
        self.skeleton, self.poses, self.hue = skeleton, poses, hue
        self._nodes: dict[str, Any] = {}
        self._lines: Any = None

    @override
    def build(self, scene: Any, root: str) -> None:
        meshes = {"cylinders": unit_cylinder(), "balls": unit_ball(), "boxes": unit_box()}
        colors = self.skeleton.colors(self.hue, _shade)
        for mesh, (world, scale) in self.skeleton.place(self.poses[0]).items():
            if len(world):  # viser's batched nodes do not take zero instances
                vertices, faces = meshes[mesh]
                self._nodes[mesh] = self._track(
                    scene.add_batched_meshes_simple(
                        f"{root}/{mesh}", vertices, faces, batched_wxyzs=to_wxyz(world[:, 3:]),
                        batched_positions=world[:, :3], batched_scales=scale.astype(np.float32),
                        batched_colors=rgb_uint8(colors[mesh]), lod="off", flat_shading=mesh == "boxes",
                    )
                )  # fmt: skip
        if len(self.skeleton.line_body):
            points = self.skeleton.lines(self.poses[0])
            color = rgb_uint8(_shade(self.hue, "face"))
            self._lines = self._track(scene.add_line_segments(f"{root}/limits", points, color, line_width=2.5))

    @override
    def set_step(self, step: int) -> None:
        for mesh, (world, scale) in self.skeleton.place(self.poses[step]).items():
            if mesh in self._nodes:
                self._nodes[mesh].batched_positions = world[:, :3]
                self._nodes[mesh].batched_wxyzs = to_wxyz(world[:, 3:])
                if (getattr(self.skeleton, mesh).anchors[:, 0] >= 0).any():  # sticks change length with the pose
                    self._nodes[mesh].batched_scales = scale.astype(np.float32)
        if self._lines is not None:
            self._lines.points = self.skeleton.lines(self.poses[step])

    def recolor(self, hue: RGB) -> None:
        """Draw the blocks in the shades of ``hue``."""
        self.hue = hue
        colors = self.skeleton.colors(hue, _shade)
        for mesh, node in self._nodes.items():
            node.batched_colors = rgb_uint8(colors[mesh])
        if self._lines is not None:
            self._lines.colors = rgb_uint8(_shade(hue, "face"))


@final
class RobotVisual(Layer):
    """One robot: its ``model`` at the configurations ``q``, shown in one of three styles.

    ``"meshes"`` draws the visual shapes of ``geometry`` (mesh files are found under ``root``, the description file
    or its folder, and under each folder above it) or the ``skin``; ``"minimal"`` and ``"maximal"`` draw the
    skeleton of the recipes ``minimal`` and ``maximal``. ``style=None`` is ``"meshes"`` when there is something to
    show as meshes, else ``"maximal"``. ``collision`` overlays the ``spheres``, or without them the collision
    shapes of ``geometry``; ``frames`` draws a triad per moving joint (``True``) or per named frame. Every part is
    built the first time it is shown; ``set_style``, ``set_collision`` and ``set_frames`` switch them while the
    viewer runs, and the sidebar gets the same three controls, shared by every robot in the viewer.

    ``q`` is ``(nq,)`` or a clip ``(T, nq)``; ``None`` is ``model.q_neutral``. ``hue=None`` takes the theme's next
    hue, one per robot. ``name`` defaults to the model's name.
    """

    def __init__(
        self,
        model: Model,
        q: torch.Tensor | np.ndarray | None = None,
        *,
        name: str | None = None,
        geometry: Geometry | None = None,
        root: str | Path | None = None,
        spheres: RobotSpheres | None = None,
        skin: Skin | None = None,
        style: Style | None = None,
        minimal: SkeletonRecipe = MINIMAL,
        maximal: SkeletonRecipe = MAXIMAL,
        hue: RGB | None = None,
        collision: bool = False,
        frames: bool | Sequence[str] = False,
        tags: Sequence[str] = (),
    ) -> None:
        self.model, self.geometry, self.spheres, self.skin = model, geometry, spheres, skin
        self.q, self._poses = self._solve(q)
        super().__init__(name or model.name, self._poses.shape[0], tags=tags)
        self._root_folder = None if root is None else Path(root)
        self._recipes = {"minimal": minimal, "maximal": maximal}
        self._skeletons: dict[str, Skeleton] = {}
        visual = geometry is not None and any(not attached.collision for attached in geometry.shapes)
        self.styles: tuple[Style, ...] = _STYLES if visual or skin is not None else _STYLES[1:]
        self.style: Style = style or ("meshes" if "meshes" in self.styles else "maximal")
        if self.style not in self.styles:
            raise ArgumentError(
                f"style must be one of {list(self.styles)} for robot {self.name!r}, got {self.style!r}; 'meshes' "
                "needs a geometry with visual shapes or a skin"
            )
        self._has_collision = spheres is not None or (
            geometry is not None and any(attached.collision for attached in geometry.shapes)
        )
        self._collision = False
        self._frames: bool | tuple[str, ...] = False
        self._hue, self._index, self._theme = hue, None, None
        self._scene: Any = None
        self._path = ""
        self._step = 0
        self._parts: dict[str, tuple[Any, list[Layer]]] = {}  # part -> (its frame node, its layers)
        if collision:
            self.set_collision(visible=True)
        self.set_frames(frames)

    # ------------------------------------------------------------------ the configurations

    def _solve(self, q: torch.Tensor | np.ndarray | None) -> tuple[torch.Tensor, np.ndarray]:
        """``q`` as ``(T, nq)`` and the joint poses ``(T, njoints, 7)`` of one forward-kinematics pass."""
        model = self.model
        q = (
            model.q_neutral
            if q is None
            else torch.as_tensor(q, dtype=model.q_neutral.dtype, device=model.q_neutral.device)
        )
        if q.ndim not in (1, 2) or q.shape[-1] != model.nq:
            raise ShapeError(f"q must be (nq,) or (T, nq) with nq = {model.nq}, got {tuple(q.shape)}")
        q = q.reshape(-1, model.nq)
        with torch.no_grad():
            poses = forward_kinematics(model, q).joint_poses
        return q, np.ascontiguousarray(poses.cpu().numpy(), dtype=np.float32)

    def update(self, q: torch.Tensor | np.ndarray) -> None:
        """Show the configurations ``q``, ``(nq,)`` or ``(T, nq)`` with the ``T`` the visual was made with: one
        forward-kinematics pass, then the shown parts are redrawn. A clip of another length is a new visual
        (``Viewer.replace``)."""
        q, poses = self._solve(q)
        if poses.shape[0] != self.nsteps:
            raise ShapeError(f"q must keep the {self.nsteps} steps of robot {self.name!r}, got {poses.shape[0]}")
        with self._lock():
            self.q = q
            self._poses[...] = poses  # the parts read this array
            for part in ("collision", "frames"):
                if part in self._parts:
                    self._derive(part, self._parts[part][1][0])
            if self.viewer is not None:
                self.redraw()
            elif self._scene is not None:
                self.set_step(self._step)

    def _lock(self) -> Any:
        return self.viewer.lock if self.viewer is not None else nullcontext()

    # ------------------------------------------------------------------ the parts

    def _shown(self) -> tuple[str, ...]:
        return (self.style, *(("collision",) if self._collision else ()), *(("frames",) if self._frames else ()))

    def _skeleton(self, part: str) -> Skeleton:
        if part not in self._skeletons:
            self._skeletons[part] = build_skeleton(
                self.model, self._recipes[part], spheres=self.spheres, geometry=self.geometry
            )
        return self._skeletons[part]

    def _color(self) -> RGB:
        if self._hue is not None:
            return self._hue
        return self._theme.hue(self._index or 0) if self._theme is not None else (0.32, 0.63, 0.41)

    def _make(self, part: str) -> list[Layer]:
        """The layers of ``part``, not yet built."""
        if part in ("minimal", "maximal"):
            return [_Blocks(part, self._skeleton(part), self._poses, self._color())]
        if part == "meshes" and self.skin is not None:
            bind = self._solve(self.skin.bind)[1][0]
            return [
                SkinnedMeshLayer(
                    part, self.skin.vertices, faces=self.skin.faces, weights=self.skin.weights, bind_poses=bind,
                    bone_poses=self._poses, color=_shade(self._color(), "body"),
                )
            ]  # fmt: skip
        if part == "frames":
            layer = FramesLayer(part, np.zeros((self.nsteps, self._frame_count(), 7), np.float32))
            layer.axes_length, layer.axes_radius = (
                1.2 * self._skeleton("maximal").radius,
                0.06 * self._skeleton("maximal").radius,
            )
            return [self._derive(part, layer)]
        if part == "collision" and self.spheres is not None:
            radii = self.spheres.radii.detach().cpu().numpy()
            centers = np.zeros((self.nsteps, len(radii), 3), np.float32)
            color = self._theme.collision if self._theme is not None else (0.94, 0.55, 0.24)
            return [self._derive(part, SpheresLayer(part, centers, radii=radii, colors=color, opacity=_OVERLAY))]
        shapes = self.geometry.shapes if self.geometry is not None else ()
        collision = part == "collision"
        color = (
            (self._theme.collision if self._theme is not None else (0.94, 0.55, 0.24))
            if collision
            else _shade(self._color(), "body")
        )
        specs = [
            _shape_spec(attached.body, attached.shape, self._root_folder, color, own_colors=not collision)
            for attached in shapes
            if attached.collision == collision
        ]
        return [RigidMeshLayer(part, specs, body_poses=self._poses, opacity=_OVERLAY if collision else 1.0)]

    def _frame_count(self) -> int:
        return len(self._moving()) if self._frames is True else len(self._frames or ())

    def _moving(self) -> list[int]:
        return [j for j in range(1, self.model.njoints) if not isinstance(self.model.joints[j], JointFixed)]

    def _derive(self, part: str, layer: Any) -> Any:
        """Fill the arrays of the ``frames`` or sphere ``collision`` layer from the joint poses."""
        model, poses = self.model, torch.from_numpy(self._poses)
        if part == "frames" and self._frames is True:
            layer.poses[...] = self._poses[:, self._moving()]
        elif part == "frames":
            ids = [model.frame_names.index(name) for name in self._frames or ()]
            parents = [model.frame_parent_joint_ids[f] for f in ids]
            layer.poses[...] = se3.compose(poses[:, parents], model.frame_placements[ids].to(poses)).numpy()
        elif self.spheres is not None:
            ids = self.spheres.frame_ids.tolist()
            parents = [model.frame_parent_joint_ids[f] for f in ids]
            frames = se3.compose(poses[:, parents], model.frame_placements[ids].to(poses))
            layer.centers[...] = se3.act(frames, self.spheres.offsets.to(poses)).numpy()
        return layer

    def _apply(self) -> None:
        """Build what is shown for the first time, show and hide the parts' frames, push the step to the new ones."""
        if self._scene is None:
            return
        with self._lock():
            for part in self._shown():
                if part not in self._parts:
                    frame = self._scene.add_frame(f"{self._path}/{part}", show_axes=False)
                    layers = self._make(part)
                    for layer in layers:
                        layer.build(self._scene, f"{self._path}/{part}")
                    self._parts[part] = (frame, layers)
                    frame.visible = False  # so that the loop below pushes its step
            for part, (frame, layers) in self._parts.items():
                show = self._visible and part in self._shown()
                if show and not frame.visible:
                    for layer in layers:
                        layer.set_step(self._step)
                if frame.visible != show:
                    frame.visible = show

    def set_style(self, style: Style) -> None:
        """Show the robot as ``style``; the first use of a style builds its nodes, later ones only show them."""
        if style not in self.styles:
            raise ArgumentError(f"style must be one of {list(self.styles)} for robot {self.name!r}, got {style!r}")
        self.style = style
        self._apply()

    def set_collision(self, visible: bool) -> None:  # noqa: FBT001 - the switch is the main operand
        """Show or hide the collision overlay: the spheres, or the collision shapes of the geometry."""
        if visible and not self._has_collision:
            raise ArgumentError(f"robot {self.name!r} has no spheres and no collision shapes to overlay")
        self._collision = visible
        self._apply()

    def set_frames(self, frames: bool | Sequence[str]) -> None:  # noqa: FBT001 - the switch is the main operand
        """Draw a triad per moving joint (``True``), per frame named in ``frames``, or none (``False``)."""
        chosen = frames if isinstance(frames, bool) else tuple(frames)
        for name in () if isinstance(chosen, bool) else chosen:
            if name not in self.model.frame_names:
                raise UnknownNameError("frame", name, valid=self.model.frame_names)
        if chosen != self._frames and "frames" in self._parts:  # another set of frames is another node
            frame, layers = self._parts.pop("frames")
            for layer in layers:
                layer.dispose()
            frame.remove()
        self._frames = chosen
        self._apply()

    # ------------------------------------------------------------------ the Layer contract

    @override
    def build(self, scene: Any, root: str) -> None:
        self._scene, self._path, self._parts = scene, root, {}
        self._apply()

    @override
    def set_step(self, step: int) -> None:
        self._step = step
        for part in self._shown():
            for layer in self._parts[part][1] if part in self._parts else ():
                layer.set_step(step)

    @override
    def build_gui(self, sidebar: Sidebar) -> None:
        def style(value: str) -> None:  # the dropdown is shared: another robot may have a style this one lacks
            if value in self.styles:
                self.set_style(value)  # type: ignore[arg-type]

        sidebar.dropdown("Robot", "Style", options=self.styles, value=self.style, on_change=style)
        if self._has_collision:
            sidebar.checkbox("Robot", "Collision", value=self._collision, on_change=self.set_collision)
        sidebar.checkbox(
            "Robot",
            "Frames",
            value=bool(self._frames),
            on_change=lambda on: self.set_frames(on and (self._frames or True)),
        )

    @override
    def apply_theme(self, theme: Theme) -> None:
        if self._index is None:  # the robot's place among the robots of its viewer picks its hue
            others = self.viewer.layers if self.viewer is not None else ()
            self._index = sum(isinstance(layer, RobotVisual) and layer is not self for layer in others)
        self._theme = theme
        for part in ("minimal", "maximal"):
            if part in self._parts:
                self._parts[part][1][0].recolor(self._color())  # type: ignore[attr-defined]

    @override
    def bounds(self, step: int) -> Bounds | None:
        pad = 2.0 * self._skeleton("maximal").radius
        low, high = box(self._poses[step, 1:, :3]) or (np.zeros(3), np.zeros(3))
        return low - pad, high + pad

    @override
    def set_visible(self, visible: bool) -> None:  # noqa: FBT001 - the switch is the main operand
        super().set_visible(visible)
        self._apply()

    @override
    def dispose(self) -> None:
        for frame, layers in self._parts.values():
            for layer in layers:
                layer.dispose()
            frame.remove()
        self._parts.clear()
        self._scene = None
        super().dispose()


# ------------------------------------------------------------------------------------------------ mesh files


def _mesh_file(uri: str, root: Path | None) -> Path:
    """The file a mesh ``uri`` names: as written when absolute, else under ``root`` (a description file or its
    folder) or a folder above it; ``package://<package>/<path>`` is tried with and without its package folder."""
    path = Path(uri.removeprefix("package://"))
    if path.is_absolute():
        return path
    relative = [path, *([Path(*path.parts[1:])] if uri.startswith("package://") and len(path.parts) > 1 else [])]
    start = Path.cwd() if root is None else (root if root.is_dir() else root.parent).resolve()
    folders = [start, *start.parents]
    for folder in folders:
        for candidate in relative:
            if (folder / candidate).is_file():
                return folder / candidate
    raise ModelError(f"mesh {uri!r} is not a file under {start} or a folder above it; pass root=, the description file")


def _shape_spec(body: int, shape: Any, root: Path | None, color: RGB, *, own_colors: bool) -> MeshSpec:
    """One shape as a ``MeshSpec``: a mesh file loaded with trimesh, a primitive tessellated by it."""
    with needs_extra("meshes"):
        import trimesh  # noqa: PLC0415 - optional dependency, imported at the boundary

    source = None
    if isinstance(shape, Mesh):
        path = _mesh_file(shape.uri, root)
        try:
            mesh = trimesh.load(path, force="mesh")
        except Exception as exc:  # noqa: BLE001 - trimesh raises many loader-specific errors
            raise ModelError(f"body {body}: mesh {shape.uri!r} does not load: {exc}") from exc
        if not isinstance(mesh, trimesh.Trimesh) or mesh.faces.shape[0] == 0:
            raise ModelError(f"body {body}: mesh {shape.uri!r} has no faces")
        mesh.apply_scale(shape.scale)
        kind = mesh.visual.kind  # a texture or more than one vertex or face colour: the mesh keeps its own look
        varied = kind == "texture" or (
            kind in ("vertex", "face")
            and len(np.unique(np.asarray(getattr(mesh.visual, f"{kind}_colors")), axis=0)) > 1
        )
        source = mesh if own_colors and varied else None
    elif isinstance(shape, Box):
        mesh = trimesh.creation.box(extents=shape.size)
    elif isinstance(shape, Cylinder):
        mesh = trimesh.creation.cylinder(radius=shape.radius, height=shape.length)
    elif isinstance(shape, Sphere):
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=shape.radius)
    else:
        mesh = trimesh.creation.capsule(radius=shape.radius, height=shape.length)
    own = shape.color[:3] if own_colors and shape.color is not None else color
    return MeshSpec(
        np.asarray(mesh.vertices, dtype=np.float32), np.asarray(mesh.faces, dtype=np.uint32), body=body,
        origin=np.asarray(shape.placement, dtype=np.float32), color=own, trimesh=source,
    )  # fmt: skip
