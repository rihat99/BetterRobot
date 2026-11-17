"""The abstract skeleton of a robot: ``SkeletonRecipe`` and the layout that turns a ``Model`` into blocks.

A skeleton is drawn from three unit meshes (a cylinder, a ball and a chamfered box), each instanced many times.
``build_skeleton`` solves the layout once, at the neutral configuration: sizes from the robot's extent and effort
limits, one glyph per joint kind (a drum for a revolute joint, a sleeve and rail for a prismatic one, a ball for a
spherical one, a cube for a free-flyer), bones between glyphs, hulls for the bodies that branch and end blocks for
the bodies that end a chain. Close joints never overlap: a drum slides along its own axis until it clears its
neighbours and shrinks only when no slide does, so every axis line stays exact. Every block is rigid in one joint
frame, so ``Skeleton.place`` poses a step with one pose composition per block. With ``glyphs="ball"`` the layout is
a stick figure instead: one ball per cluster of close joints, joined by sticks whose ends follow the balls.

Nothing here imports viser: the robot visual sends the arrays. Poses are ``[tx, ty, tz, qx, qy, qz, qw]``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Literal, final

import numpy as np
import torch

from ..collision import RobotSpheres
from ..exceptions import ArgumentError, UnknownNameError
from ..kinematics import forward_kinematics
from ..lie import se3, so3
from ..model import (
    Box,
    Capsule,
    Cylinder,
    Geometry,
    JointFixed,
    JointFreeFlyer,
    JointHelical,
    JointPrismatic,
    JointRevolute,
    JointRevoluteUnbounded,
    JointSpherical,
    Mesh,
    Model,
    Sphere,
)

#: A colour as the viewer takes it everywhere: red, green and blue in ``[0, 1]``.
RGB = tuple[float, float, float]

_TWIST = float(np.cos(np.radians(35.0)))  # a joint axis within 35 degrees of its limb twists the limb
_HALF = 0.8  # half length of a drum over its radius
_SLEEVE = 0.6  # half length of a sleeve along its rail over the joint radius
_MARKER = 0.6  # radius of the plain marker of an unlisted joint kind over the joint radius
_TRUNK_PAD = 0.7  # a trunk hull pads its sphere centres by this share of their median radius
_TRUNK = 0.8  # a trunk keeps this share of the half extents of its geometry: the skeleton is slimmer than the robot
_TAPER = 0.8  # a maximal block's side at z = 0.5 over its side at z = -0.5: every body narrows towards +z
_HEAD_CAP = 0.8  # a second frame of a body is a ball of at most this share of the body hull's smallest half extent
_END_PAD = 0.4  # an end block pads its points by this share of min(joint radius, median sphere radius)
_FLAT = 0.3  # a leaf's geometry is a sole when its thinnest extent is under this share of its middle one
_TIP = 0.55  # the ball that ends a leaf limb, over the leaf joint's radius
_PLATE = 0.45  # a sole is this many joint radii thick
_LEAF_CAP = 1.25  # a leaf glyph is at most this share of the half thickness of the body it moves
_FIXED_CAP = 0.4  # a glyph that cannot slide is at most this share of its shortest bone
_ARC = 24  # segments of a limit arc
_KINDS = {
    JointFixed: "fixed", JointRevolute: "revolute", JointRevoluteUnbounded: "revolute", JointHelical: "revolute",
    JointPrismatic: "prismatic", JointSpherical: "spherical", JointFreeFlyer: "free",
}  # fmt: skip
#: Per joint kind, its blocks: (unit mesh, part, rigid in the parent group or in the child, scale over the radius).
_GLYPHS = {
    "revolute": (("cylinders", "drum", True, (1.0, 1.0, 2 * _HALF)), ("cylinders", "face", True, (0.8, 0.8, 2.2 * _HALF))),
    "prismatic": (("boxes", "sleeve", False, (1.6, 1.6, 2 * _SLEEVE)), ("cylinders", "rail", True, (0.5, 0.5, 4.4 * _HALF))),
    "spherical": (("balls", "ball", False, (1.0, 1.0, 1.0)),),
    "free": (("boxes", "base", False, (1.5, 1.5, 1.5)),),
    "other": (("balls", "marker", False, (1.0, 1.0, 1.0)),),
}  # fmt: skip
_Shade = Literal["joint", "face", "body", "bone", "stick"]
#: The shade each part of the skeleton takes from the robot's hue: light bodies, mid drums, dark drum faces.
_ROLE: dict[str, _Shade] = {
    "drum": "joint", "sleeve": "joint", "ball": "joint", "tip": "joint",
    "face": "face", "rail": "face", "marker": "face", "base": "face",
    "bone": "bone", "fill": "bone", "hull": "bone", "end": "bone",
    "stick": "stick",
}  # fmt: skip
_MESHES = ("cylinders", "balls", "boxes")


@final
@dataclass(frozen=True, kw_only=True)
class SkeletonRecipe:
    """How a robot's skeleton is drawn; ``MAXIMAL`` and ``MINIMAL`` are the two presets.

    Parameters
    ----------
    glyphs
        ``"kind"`` draws one glyph per joint kind, ``"ball"`` one ball per cluster of close joints, ``"none"`` bones
        only.
    bones
        Bone radius over glyph radius; 0 draws no bones.
    hulls
        ``"auto"`` gives a hull to the root body and to every body with two or more child joints and an end block
        to every body that ends a chain; ``"all"`` also gives a hull to the bodies with one child joint; ``"none"``
        draws neither.
    limits
        Draw the limit arc of each bounded revolute joint, with a tick at the current angle (``glyphs="kind"``).
    cluster
        Joints closer than ``cluster`` times the robot's size share one ball (``glyphs="ball"``).
    twists
        ``False`` draws no glyph for a lone joint whose axis runs along its limb; the bone runs through.
    sides
        Tint the left side of the robot lighter and the right side darker.
    size
        Multiplies every glyph radius.
    joint_size
        Joint name to a multiplier of its glyph radius; 0 hides the glyph.
    body_hull
        Frame name to ``True`` (draw the hull of its geometry) or ``False`` (leave it out).
    color
        Joint or frame name to the colour of its glyph or hull, in place of the robot's hue.
    """

    glyphs: Literal["kind", "ball", "none"] = "kind"
    bones: float = 0.6
    hulls: Literal["auto", "all", "none"] = "auto"
    limits: bool = False
    cluster: float = 0.0
    twists: bool = True
    sides: bool = False
    size: float = 1.0
    joint_size: Mapping[str, float] = field(default_factory=dict)
    body_hull: Mapping[str, bool] = field(default_factory=dict)
    color: Mapping[str, RGB] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Raise ``ArgumentError`` for an unknown ``glyphs`` or ``hulls`` word or a size out of range."""
        if self.glyphs not in ("kind", "ball", "none"):
            raise ArgumentError(f"glyphs must be 'kind', 'ball' or 'none', got {self.glyphs!r}")
        if self.hulls not in ("auto", "all", "none"):
            raise ArgumentError(f"hulls must be 'auto', 'all' or 'none', got {self.hulls!r}")
        if (
            self.bones < 0.0
            or self.cluster < 0.0
            or self.size <= 0.0
            or min(self.joint_size.values(), default=0.0) < 0.0
        ):
            raise ArgumentError(
                f"bones, cluster and joint_size must be at least 0 and size above 0, got bones={self.bones}, "
                f"cluster={self.cluster}, size={self.size}, joint_size={dict(self.joint_size)}"
            )


MAXIMAL = SkeletonRecipe()
MINIMAL = SkeletonRecipe(glyphs="ball", bones=0.42, hulls="none", cluster=0.06, twists=False, size=0.62)


@final
@dataclass(frozen=True)
class Blocks:
    """``N`` instances of one unit mesh.

    Instance ``i`` is rigid in the frame of joint ``body[i]`` at ``local[i]``, with per-axis ``scale[i]``. A moving
    instance (``anchors[i, 0] >= 0``) instead sits midway between two anchors of its ``Skeleton``; a moving
    cylinder also points from the first to the second and takes their distance as its length.
    """

    body: np.ndarray  # (N,) joint index
    local: np.ndarray  # (N, 7) pose in that joint's frame
    scale: np.ndarray  # (N, 3)
    anchors: np.ndarray  # (N, 2) anchor indices, -1 for a rigid instance
    owner: np.ndarray  # (N,) the joint a block draws or belongs to, -1 for a hull
    part: tuple[str, ...]  # a key of _ROLE per instance
    side: np.ndarray  # (N,) +1 left, -1 right, 0 neither or sides off
    color: tuple[RGB | None, ...]  # the recipe's colour for the instance, if any

    def __len__(self) -> int:
        """The number of instances."""
        return len(self.part)


@final
@dataclass(frozen=True)
class Skeleton:
    """The blocks of one robot under one recipe: ``build_skeleton`` makes it, ``place`` poses it.

    ``scale`` is the robot's size (diagonal of the box of its joint origins at neutral) and ``radius`` the reference
    glyph radius. ``slides`` (metres along the joint's own axis) and ``shrinks`` (factor in ``[0.5, 1]``) list, by
    joint name, the glyphs the layout moved or shrank; ``clusters`` the joint names behind each ball of a
    ``glyphs="ball"`` recipe. ``taper`` is the ``unit_box`` the boxes are drawn with: 0.8 for the maximal layout,
    whose bodies narrow towards their +z, 1 (a cube) for the stick figure.
    """

    scale: float
    radius: float
    taper: float
    cylinders: Blocks
    balls: Blocks
    boxes: Blocks
    points_body: np.ndarray  # (P,) the joint each anchor point is rigid in
    points_local: np.ndarray  # (P, 3)
    weights: np.ndarray  # (K, P): anchor k is the weighted mean of the points
    line_body: np.ndarray  # (L,) limit arcs and ticks as segments rigid in a joint frame
    line_local: np.ndarray  # (L, 2, 3)
    slides: Mapping[str, float]
    shrinks: Mapping[str, float]
    clusters: tuple[tuple[str, ...], ...]

    def place(self, joint_poses: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """World poses and scales of every block at ``joint_poses (..., njoints, 7)``.

        Returns ``{"cylinders" | "balls" | "boxes": (poses (..., N, 7), scales)}``; ``scales`` is ``(N, 3)``, or
        ``(..., N, 3)`` for the cylinders of a skeleton with moving sticks, whose lengths follow the pose.
        """
        poses = torch.from_numpy(np.ascontiguousarray(joint_poses))
        anchors = None
        if len(self.weights):
            local = torch.from_numpy(self.points_local).to(poses.dtype)
            points = se3.act(poses[..., self.points_body, :], local).numpy()
            anchors = self.weights.astype(points.dtype) @ points  # (..., K, 3)
        placed = {}
        for name in _MESHES:
            blocks: Blocks = getattr(self, name)
            local = torch.from_numpy(blocks.local).to(poses.dtype)
            world = se3.compose(poses[..., blocks.body, :], local).numpy()
            scale = blocks.scale
            moving = blocks.anchors[:, 0] >= 0
            if anchors is not None and moving.any():
                start, end = anchors[..., blocks.anchors[moving, 0], :], anchors[..., blocks.anchors[moving, 1], :]
                world[..., moving, :3] = (start + end) / 2
                if name == "cylinders":
                    world[..., moving, 3:] = _from_z(end - start)
                    scale = np.broadcast_to(scale, (*world.shape[:-1], 3)).astype(world.dtype)
                    scale[..., moving, 2] = np.linalg.norm(end - start, axis=-1)
            placed[name] = (world, scale)
        return placed

    def lines(self, joint_poses: np.ndarray) -> np.ndarray:
        """World end points ``(..., L, 2, 3)`` of the limit arcs and ticks at ``joint_poses (..., njoints, 7)``."""
        poses = torch.from_numpy(np.ascontiguousarray(joint_poses))
        local = torch.from_numpy(self.line_local).to(poses.dtype)
        return se3.act(poses[..., self.line_body, None, :], local).numpy()

    def colors(self, hue: RGB, shade: Callable[[RGB, _Shade], RGB]) -> dict[str, np.ndarray]:
        """Colours ``(N, 3)`` in ``[0, 1]`` of every block: ``shade(hue, role)`` per part, tinted by side, or its
        override.

        ``shade`` is ``Theme.shade``. A left block takes the hue 35 % towards white, a right one 25 % towards black.
        """
        left, right = _mix(hue, (1.0, 1.0, 1.0), 0.35), _mix(hue, (0.0, 0.0, 0.0), 0.25)
        colors = {}
        for name in _MESHES:
            blocks: Blocks = getattr(self, name)
            rows = np.zeros((len(blocks), 3), dtype=np.float32)
            for i, (part, side, own) in enumerate(zip(blocks.part, blocks.side, blocks.color, strict=True)):
                role = _ROLE[part]
                base = own if own is not None else left if side > 0 else right if side < 0 else hue
                rows[i] = base if own is not None and role != "face" else shade(base, role)
            colors[name] = rows
        return colors


def build_skeleton(
    model: Model,
    recipe: SkeletonRecipe,
    *,
    spheres: RobotSpheres | None = None,
    geometry: Geometry | None = None,
    mesh_vertices: Callable[[Mesh], np.ndarray | None] | None = None,
) -> Skeleton:
    """Lay out the skeleton of ``model`` under ``recipe``, from one forward-kinematics pass at ``q_neutral``.

    Hulls and end blocks are sized from the first source a body has: the ``spheres`` on its frames, the collision
    shapes of ``geometry`` (a mesh only through ``mesh_vertices``, which returns the vertices ``(V, 3)`` of a mesh
    file or ``None``), its inertia. A name of ``recipe`` that the model does not have raises ``UnknownNameError``.
    """
    for name in recipe.joint_size:
        if name not in model.joint_names:
            raise UnknownNameError("joint", name, valid=model.joint_names)
    for name in recipe.body_hull:
        if name not in model.frame_names:
            raise UnknownNameError("frame", name, valid=model.frame_names)
    for name in recipe.color:
        if name not in model.joint_names and name not in model.frame_names:
            raise UnknownNameError("joint or frame", name, valid=(*model.joint_names, *model.frame_names))
    tree = _tree(model, recipe)
    clouds = _clouds(tree, spheres, geometry, mesh_vertices)
    tree = _leaf_caps(tree, clouds, maximal=recipe.glyphs == "kind")
    rows = _Rows(tree, recipe)
    slides: dict[str, float] = {}
    shrinks: dict[str, float] = {}
    clusters: tuple[tuple[str, ...], ...] = ()
    if recipe.glyphs == "kind":
        hidden = {j for j in tree.glyphs if tree.radius[j] <= 0.0}
        if not recipe.twists:
            hidden |= {j for j, after in tree.children.items() if j and len(after) == 1 and _twists(tree, j, after)}
        slide, shrink = _solve(tree, hidden, clouds)
        slides = {model.joint_names[j]: s for j, s in slide.items() if abs(s) > 1e-9}
        shrinks = {model.joint_names[j]: k for j, k in shrink.items() if k < 1.0}
        reach = {j: 0.0 if j in hidden else shrink[j] * tree.radius[j] for j in tree.glyphs}
        _maximal(tree, recipe, rows, clouds, reach=reach, slide=slide)
    else:
        clusters = _minimal(tree, recipe, rows, clouds)
    return rows.pack(slides=slides, shrinks=shrinks, clusters=clusters)


# ------------------------------------------------------------------------------------------------ numpy helpers


def _numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy().astype(np.float64)


def _matrix(quaternion: np.ndarray) -> np.ndarray:
    """Rotation matrices ``(..., 3, 3)`` of ``xyzw`` quaternions ``(..., 4)`` (``so3.to_matrix``)."""
    return so3.to_matrix(torch.from_numpy(np.ascontiguousarray(quaternion, dtype=np.float64))).numpy()


def _from_z(direction: np.ndarray) -> np.ndarray:
    """Shortest-arc ``xyzw`` rotations ``(..., 4)`` taking ``+z`` onto each ``direction (..., 3)``.

    A zero direction gives the identity, one opposite ``+z`` a half turn about ``x``.
    """
    norm = np.linalg.norm(direction, axis=-1, keepdims=True)
    unit = np.where(norm > 1e-12, direction / np.where(norm > 1e-12, norm, 1.0), [0.0, 0.0, 1.0])
    # the axis z x d with the half-angle form (sin(t) n, 1 + cos(t)), then normalised
    turn = np.stack([-unit[..., 1], unit[..., 0], np.zeros_like(unit[..., 0]), 1.0 + unit[..., 2]], axis=-1)
    length = np.linalg.norm(turn, axis=-1, keepdims=True)
    return np.where(length > 1e-9, turn / np.where(length > 1e-9, length, 1.0), [1.0, 0.0, 0.0, 0.0])


def _limb_frame(direction: np.ndarray, across: np.ndarray | None) -> np.ndarray:
    """The rotation ``(3, 3)`` of a limb along the unit ``direction``: its z is ``direction`` and its x the part of
    ``across`` square to it (a drum's axis, so that the limb's sides face the way the drum's faces do)."""
    side = np.zeros(3) if across is None else across - (across @ direction) * direction
    if np.linalg.norm(side) < 1e-6:
        return _matrix(_from_z(direction))
    side = side / np.linalg.norm(side)
    return np.stack([side, np.cross(direction, side), direction], axis=1)


def _mix(color: RGB, other: RGB, share: float) -> RGB:
    red, green, blue = (a + (b - a) * share for a, b in zip(color, other, strict=True))
    return red, green, blue


_OrientedBox = tuple[np.ndarray, np.ndarray, np.ndarray]  # centre (3,), rotation (3, 3), half extents (3,)


def _gaps(centers: np.ndarray, rotation: np.ndarray, half: np.ndarray, others: Sequence[_OrientedBox]) -> np.ndarray:
    """Signed gap ``(C,)`` from each of ``C`` placements of one oriented box to the nearest of ``others``.

    The 15-axis separating-axis test: positive is clear space, negative the penetration depth. ``centers (C, 3)``
    are the candidate centres of the first box; with no other box the gap is infinite.
    """
    if not others:
        return np.full(len(centers), np.inf)
    other_centers, other_rotations, other_halves = (np.stack(column) for column in zip(*others, strict=True))
    own = np.broadcast_to(rotation.T, (len(others), 3, 3))
    theirs = np.swapaxes(other_rotations, 1, 2)
    cross = np.cross(own[:, :, None, :], theirs[:, None, :, :]).reshape(len(others), 9, 3)
    axes = np.concatenate([own, theirs, cross], axis=1)  # (M, 15, 3)
    norm = np.linalg.norm(axes, axis=2, keepdims=True)
    axes = axes / np.where(norm > 1e-9, norm, 1.0)
    reach = np.abs(axes @ rotation) @ half + np.einsum("mki,mi->mk", np.abs(axes @ other_rotations), other_halves)
    offset = other_centers[None] - centers[:, None]  # (C, M, 3)
    gap = np.abs(np.einsum("cmi,mki->cmk", offset, axes)) - reach[None]
    gap = np.where(norm[None, :, :, 0] > 1e-9, gap, -np.inf)  # a cross product of parallel edges separates nothing
    return gap.max(axis=2).min(axis=1)


# ------------------------------------------------------------------------------------------------ the tree and its sizes


@final
@dataclass(frozen=True)
class _Tree:
    """What every stage of the layout reads: the joint tree, the neutral poses and the sizes.

    A rigid group is a moving joint plus the bodies welded to it by fixed joints; group 0 is the world. ``anchor[j]``
    is the group of joint ``j``, ``glyphs`` the moving joints, ``group[j]`` the parent group of glyph ``j`` and
    ``children[g]`` the glyphs whose parent group is ``g``.
    """

    model: Model
    kind: tuple[str, ...]  # a value of _KINDS, or "other", per joint
    anchor: tuple[int, ...]
    root: tuple[int, ...]  # per joint, the joint under the universe on its path: the root body it hangs from
    glyphs: tuple[int, ...]
    group: Mapping[int, int]
    children: Mapping[int, tuple[int, ...]]
    origin: np.ndarray  # (njoints, 3) at neutral
    rotation: np.ndarray  # (njoints, 3, 3) at neutral
    neutral: np.ndarray  # (njoints, 7)
    axis: Mapping[int, np.ndarray]  # world axis at neutral of each revolute or prismatic glyph
    scale: float  # the robot's size
    reference: float  # the reference glyph radius
    radius: Mapping[int, float]  # per glyph

    def body(self, group: int, joint: int) -> int:
        """The joint whose frame carries what is rigid in ``group``: the group itself, or for the world the root
        body that ``joint`` hangs from."""
        return group if group else self.root[joint]

    def bones(self, joint: int) -> list[float]:
        """Distances from glyph ``joint`` to its parent and child glyphs, those under 1 % of the robot ignored."""
        ends = (*((self.group[joint],) if self.group[joint] else ()), *self.children[joint])
        lengths = [float(np.linalg.norm(self.origin[joint] - self.origin[i])) for i in ends]
        return [length for length in lengths if length > 0.01 * self.scale]


def _tree(model: Model, recipe: SkeletonRecipe) -> _Tree:
    """The tree of ``model`` at ``q_neutral`` with the sizes of ``recipe``.

    The robot's size is the diagonal of the box of its joint origins; the reference radius a third of the mean bone,
    kept between 1 % and 6 % of the size; a joint's radius the reference times the sixth root of its effort limit
    over the median one, kept between 0.7 and 1.3. A glyph that cannot slide is no thicker than 0.4 of its
    shortest bone. The maximal layout (``glyphs="kind"``) is calmer: the reference between 2 % and 4 % of the size,
    the factor between 0.8 and 1.2, against the median of the joints of the same kind.
    """
    njoints, parents = model.njoints, model.parent_joint_ids
    kind = ("fixed", *(_KINDS.get(type(joint), "other") for joint in model.joints[1:]))
    anchor, root = [0] * njoints, [0] * njoints
    for j in range(1, njoints):
        anchor[j] = j if kind[j] != "fixed" else anchor[parents[j]]
        root[j] = j if parents[j] == 0 else root[parents[j]]
    glyphs = tuple(j for j in range(1, njoints) if kind[j] != "fixed")
    group = {j: anchor[parents[j]] for j in glyphs}
    children = {g: tuple(j for j in glyphs if group[j] == g) for g in (0, *glyphs)}
    with torch.no_grad():
        neutral = _numpy(forward_kinematics(model, model.q_neutral).joint_poses)
    origin, rotation = neutral[:, :3], _matrix(neutral[:, 3:])
    axis = {j: rotation[j] @ np.array(model.joints[j].axis) for j in glyphs if kind[j] in ("revolute", "prismatic")}
    extent = origin[1:] if njoints > 1 else origin
    scale = float(np.linalg.norm(extent.max(axis=0) - extent.min(axis=0)))
    scale = scale if scale > 1e-9 else 1.0  # a single body: a unit robot
    spans = [float(np.linalg.norm(origin[j] - origin[group[j]])) for j in glyphs if group[j]]
    spans = [span for span in spans if span > 0.01 * scale]
    maximal = recipe.glyphs == "kind"  # the stick figure keeps its own sizes
    low, high, spread = (0.02, 0.04, 0.2) if maximal else (0.01, 0.06, 0.3)
    reference = float(np.clip(np.mean(spans) / 3.0, low * scale, high * scale)) if spans else 0.03 * scale
    reference *= recipe.size
    tree = _Tree(
        model, kind, tuple(anchor), tuple(root), glyphs, group, children, origin, rotation, neutral, axis, scale,
        reference, {},
    )  # fmt: skip
    effort = _numpy(model.effort_limit)
    torque = {}
    for j in glyphs:  # a mimic joint is as strong as the joint that drives it
        source = model.mimic_source_ids[j]
        value = float(effort[model.idx_vs[source]]) if model.nvs[source] == 1 else np.nan
        torque[j] = value if np.isfinite(value) and value > 0.0 else np.nan
    radius = {}
    for j in glyphs:  # the maximal layout compares each kind with its own median: a force is not a torque
        known = [value for i, value in torque.items() if np.isfinite(value) and (kind[i] == kind[j] or not maximal)]
        ratio = torque[j] / np.median(known) if torque[j] > 0.0 else 1.0
        factor = float(np.clip(ratio ** (1.0 / 6.0), 1.0 - spread, 1.0 + spread))
        radius[j] = reference * factor * recipe.joint_size.get(model.joint_names[j], 1.0)
        radius[j] *= _MARKER if kind[j] == "other" else 1.0
        if kind[j] in ("spherical", "other") and tree.bones(j):
            radius[j] = min(radius[j], _FIXED_CAP * min(tree.bones(j)))
    return replace(tree, radius=radius)


def _leaf_caps(tree: _Tree, clouds: Mapping[int, list[_Cloud]], *, maximal: bool) -> _Tree:
    """``tree`` with each leaf glyph no thicker than the body it moves: at most 1.25 times the median radius of the
    spheres it carries or, in the ``maximal`` layout, of the middle half extent of its collision points in its own
    frame.

    The ``maximal`` layout also narrows towards the tips: a glyph is no wider than the glyph before it (a
    free-flyer's cube does not count)."""
    radius = dict(tree.radius)
    for j in tree.glyphs:
        if tree.children[j] or j not in clouds:
            continue
        radii = [cloud.radii for cloud in clouds[j] if cloud.radii is not None]
        if radii:
            size = float(np.median(np.concatenate(radii)))
        elif not maximal:  # the stick figure is capped by spheres only
            continue
        else:
            points = np.concatenate([cloud.origin + cloud.points @ cloud.rotation.T for cloud in clouds[j]])
            size = float(np.sort(np.ptp((points - tree.origin[j]) @ tree.rotation[j], axis=0))[1]) / 2
        radius[j] = min(radius[j], _LEAF_CAP * size)
    for j in tree.glyphs if maximal else ():  # in index order, a parent before its children
        parent = tree.group[j]
        if parent and tree.kind[parent] != "free":
            radius[j] = min(radius[j], radius[parent])
    return replace(tree, radius=radius)


def _twists(tree: _Tree, joint: int, after: Sequence[int]) -> bool:
    """Whether revolute ``joint`` twists its limb: a joint before it, and its axis within 35 degrees of the line from
    that joint to the mean of the joints ``after`` it (none: no limb, no twist)."""
    before = tree.group[joint]
    if tree.kind[joint] != "revolute" or not before or not after:
        return False
    limb = tree.origin[list(after)].mean(axis=0) - tree.origin[before]
    return abs(float(tree.axis[joint] @ limb)) > _TWIST * max(float(np.linalg.norm(limb)), 1e-12)


# ------------------------------------------------------------------------------------------------ slide, then shrink


def _solve(
    tree: _Tree, hidden: set[int], clouds: Mapping[int, list[_Cloud]]
) -> tuple[dict[int, float], dict[int, float]]:
    """Slide ``s`` along its own axis and shrink ``k`` of every glyph, so that neighbours do not overlap.

    A sleeve of a prismatic leaf starts over the middle of the leaf's geometry along its axis, and on the same side
    of the joint's zero place when the geometry is: a finger rides on its own carriage, and two fingers that close
    on each other have their carriages meet there. One deterministic pass. Joints with a child joint are placed in reverse index order, then the leaf joints, so a
    distal glyph keeps its place and a proximal one gives way; a leaf gives way last because its drum sits against
    its own end block. A glyph is tested against the placed glyphs of its rigid groups (parent, siblings, children:
    clearance that holds for every configuration) and, where it can, against those two hops away (at neutral only).
    """
    origin = tree.origin
    solid = [j for j in tree.glyphs if j not in hidden and tree.kind[j] != "free"]  # the base cube sits in the hull
    frame = {j: _matrix(_from_z(tree.axis[j])) if j in tree.axis else np.eye(3) for j in solid}
    slide = {j: 0.0 for j in tree.glyphs}
    shrink = {j: 1.0 for j in tree.glyphs}
    for j in solid:  # a drum on a twist joint starts with its far face on the joint: a collar on the limb
        limb = origin[j] - (origin[tree.group[j]] if tree.group[j] else origin[tree.root[j]])
        length = float(np.linalg.norm(limb))
        along = float(tree.axis[j] @ limb) / length if tree.kind[j] == "revolute" and length > tree.radius[j] else 0.0
        if abs(along) > _TWIST:
            slide[j] = -np.sign(along) * _HALF * tree.radius[j]
        if tree.kind[j] == "prismatic" and not tree.children[j] and clouds.get(j):
            reach = np.concatenate([(c.origin + c.points @ c.rotation.T - origin[j]) @ tree.axis[j] for c in clouds[j]])
            slide[j] = float(reach.min() + reach.max()) / 2
            if reach.min() * reach.max() >= 0.0:  # a finger on one side of its zero place: its sleeve stays there too
                slide[j] = float(np.sign(slide[j])) * max(abs(slide[j]), (_SLEEVE + 0.05) * tree.radius[j])

    def box(j: int) -> _OrientedBox:
        center = origin[j] + slide[j] * tree.axis[j] if j in tree.axis else origin[j]
        return center, frame[j], _half(tree, j, shrink[j] * tree.radius[j])

    def near(j: int) -> set[int]:
        return {*tree.children[tree.group[j]], *tree.children[j], tree.group[j]}.intersection(solid) - {j}

    placed: set[int] = set()
    for j in sorted(solid, key=lambda j: (not tree.children[j], -j)):
        hard = [box(i) for i in sorted(near(j) & placed)]
        soft = [box(i) for i in sorted({i for n in near(j) for i in near(n)} - near(j) - {j}) if i in placed]
        if hard:
            slide[j], shrink[j] = _fit(tree, j, slide[j], frame[j], hard, soft)
        placed.add(j)
    return slide, shrink


def _half(tree: _Tree, joint: int, reach: float) -> np.ndarray:
    """Half extents of the box around the glyph of ``joint`` drawn with radius ``reach``: a drum or a sleeve along z,
    else a ball."""
    along = _SLEEVE if tree.kind[joint] == "prismatic" else _HALF
    return reach * (np.array([1.0, 1.0, along]) if joint in tree.axis else np.ones(3))


def _fit(
    tree: _Tree, joint: int, base: float, frame: np.ndarray, hard: Sequence[_OrientedBox], soft: Sequence[_OrientedBox]
) -> tuple[float, float]:
    """The slide and shrink of one glyph against its placed neighbours ``hard`` and, if possible, ``soft``.

    Candidates step a quarter of the half length away from ``base``, alternating sides, first the side away from
    the root body (so left and right come out as mirror images). The first one clear of every box by 0.1 radius
    wins, else the first clear of ``hard``; with none the glyph shrinks by 0.85 and tries again, and at 0.5 keeps
    the least penetration. A ball or a marker has the one candidate and only shrinks. A leaf first shrinks at
    ``base`` until it clears ``hard``, and only when even half its size does not, slides as any other glyph: the end
    of a chain gets smaller instead of leaving its place.
    """
    radius, origin, axis = tree.radius[joint], tree.origin[joint], tree.axis.get(joint)
    sign = 1.0
    if axis is not None:
        away = float((origin - tree.origin[tree.root[joint]]) @ axis)
        if abs(away) < 1e-6 * tree.scale:  # level with the root: the axis's largest component in the root frame
            in_root = np.round(tree.rotation[tree.root[joint]].T @ axis, 6)
            away = float(in_root[np.abs(in_root).argmax()])
        sign = 1.0 if away >= 0.0 else -1.0
    shrink = 1.0
    while not tree.children[joint]:  # a leaf ends its chain smaller rather than away from its place
        center = origin + base * (axis if axis is not None else np.zeros(3))
        if _gaps(center[None], frame, _half(tree, joint, shrink * radius), hard)[0] >= 0.1 * shrink * radius:
            return base, shrink
        if shrink <= 0.5:
            shrink = 1.0
            break
        shrink = max(0.5, 0.85 * shrink)
    while True:
        half, offsets = _half(tree, joint, shrink * radius), [0.0]
        if axis is not None:
            step = 0.25 * half[2]
            steps = int(1.25 * (half[2] + max(other[2][0] for other in hard)) / step)
            offsets += [side * m * step for m in range(1, steps + 1) for side in (sign, -sign)]
        candidates = base * shrink + np.array(offsets)
        centers = origin + candidates[:, None] * (axis if axis is not None else np.zeros(3))
        clear = _gaps(centers, frame, half, hard)
        for gaps in (np.minimum(clear, _gaps(centers, frame, half, soft)), clear):
            if (gaps >= 0.1 * shrink * radius).any():
                return float(candidates[np.argmax(gaps >= 0.1 * shrink * radius)]), shrink
        if shrink <= 0.5:
            return float(candidates[np.argmax(clear)]), shrink
        shrink = max(0.5, 0.85 * shrink)


# ------------------------------------------------------------------------------------------------ hulls and end blocks


@final
@dataclass(frozen=True)
class _Cloud:
    """Geometry of one frame: ``points (n, 3)`` in the frame at ``rotation``, ``origin`` (world, at neutral)."""

    name: str
    joint: int
    rotation: np.ndarray
    origin: np.ndarray
    points: np.ndarray
    radii: np.ndarray | None  # per point for a sphere set; None when the points lie on the surface


@final
@dataclass(frozen=True)
class _Hull:
    """One hull or end block, rigid in joint ``body``; world, at neutral. ``half`` is the half extents along the
    columns of ``rotation``: of a box or a trunk (wide at ``-z``), of a limb from ``-z`` to ``+z``, or a ball's
    radius three times."""

    name: str
    part: str  # "hull" or "end"
    group: int
    body: int
    center: np.ndarray
    rotation: np.ndarray
    half: np.ndarray
    mesh: str = "boxes"  # "boxes" or "balls"


def _clouds(
    tree: _Tree,
    spheres: RobotSpheres | None,
    geometry: Geometry | None,
    mesh_vertices: Callable[[Mesh], np.ndarray | None] | None,
) -> dict[int, list[_Cloud]]:
    """Per rigid group, the geometry its hulls are sized from: its sphere set, else its collision shapes."""
    model = tree.model
    found: dict[int, list[_Cloud]] = {}
    if spheres is not None:
        frame_ids, offsets, radii = spheres.frame_ids.cpu().numpy(), _numpy(spheres.offsets), _numpy(spheres.radii)
        placements = _numpy(model.frame_placements)
        for f in np.unique(frame_ids):
            joint, placement, mask = model.frame_parent_joint_ids[f], placements[f], frame_ids == f
            rotation = tree.rotation[joint] @ _matrix(placement[3:])
            origin = tree.origin[joint] + tree.rotation[joint] @ placement[:3]
            cloud = _Cloud(model.frame_names[f], joint, rotation, origin, offsets[mask], radii[mask])
            found.setdefault(tree.anchor[joint], []).append(cloud)
    shapes: dict[int, list[np.ndarray]] = {}
    for attached in geometry.shapes if geometry is not None else ():
        points = _shape_points(attached.shape, mesh_vertices) if attached.collision else None
        if points is not None and tree.anchor[attached.body] not in found:
            placement = np.array(attached.shape.placement)
            shapes.setdefault(attached.body, []).append(points @ _matrix(placement[3:]).T + placement[:3])
    for joint, parts in shapes.items():
        cloud = _Cloud(
            model.body_names[joint], joint, tree.rotation[joint], tree.origin[joint], np.concatenate(parts), None
        )
        found.setdefault(tree.anchor[joint], []).append(cloud)
    return found


def _shape_points(shape: object, mesh_vertices: Callable[[Mesh], np.ndarray | None] | None) -> np.ndarray | None:
    """Points ``(n, 3)`` on the surface of a shape, in the shape's own frame; ``None`` for a mesh that does not load."""
    if isinstance(shape, Box):
        return np.array([[x, y, z] for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)]) * shape.size
    if isinstance(shape, Sphere):
        return np.concatenate([np.eye(3), -np.eye(3)]) * shape.radius
    if isinstance(shape, (Cylinder, Capsule)):
        angles = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
        ring = np.stack([np.cos(angles), np.sin(angles), np.zeros(8)], axis=1) * shape.radius
        tip = np.array([[0.0, 0.0, shape.length / 2 + (shape.radius if isinstance(shape, Capsule) else 0.0)]])
        return np.concatenate([ring + [0.0, 0.0, shape.length / 2], ring - [0.0, 0.0, shape.length / 2], tip, -tip])
    if isinstance(shape, Mesh) and mesh_vertices is not None:
        vertices = mesh_vertices(shape)
        return None if vertices is None else np.asarray(vertices, dtype=np.float64) * shape.scale
    return None


def _inertia(tree: _Tree, group: int) -> tuple[float, np.ndarray, np.ndarray] | None:
    """Mass, centre of mass (world, at neutral) and inertia about it of rigid ``group``; ``None`` without mass."""
    inertias = _numpy(tree.model.body_inertias)
    bodies = [b for b in range(1, tree.model.nbodies) if tree.anchor[b] == group and inertias[b, 0] > 0.0]
    if not bodies:
        return None
    mass = float(inertias[bodies, 0].sum())
    centers = {b: tree.origin[b] + tree.rotation[b] @ inertias[b, 1:4] for b in bodies}
    center = np.sum([inertias[b, 0] * centers[b] for b in bodies], axis=0) / mass
    total = np.zeros((3, 3))
    for b in bodies:
        xx, yy, zz, xy, xz, yz = inertias[b, 4:]
        own = tree.rotation[b] @ np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]]) @ tree.rotation[b].T
        arm = centers[b] - center
        total += own + inertias[b, 0] * (float(arm @ arm) * np.eye(3) - np.outer(arm, arm))
    return mass, center, total


def _hulls(
    tree: _Tree,
    recipe: SkeletonRecipe,
    clouds: Mapping[int, list[_Cloud]],
    reach: Mapping[int, float],
    glyphs: Mapping[int, _OrientedBox],
) -> list[_Hull]:
    """The hulls and end blocks of ``recipe``; ``reach[g]`` is the drawn radius of the glyph of group ``g`` and
    ``glyphs`` the boxes of the glyphs an end block gives way to.

    A trunk group (the root, or two or more child joints; with ``hulls="all"`` also one child joint) gets its trunk
    hulls, a leaf group (no child joint) its end block; ``body_hull`` adds or removes single frames.
    """
    hulls: list[_Hull] = []
    for g in (0, *tree.glyphs):
        kids = len(tree.children[g])
        root = g == 0 or (tree.group[g] == 0 and tree.kind[g] == "free")
        wanted = [cloud for cloud in clouds.get(g, []) if recipe.body_hull.get(cloud.name, True)]
        forced = [cloud for cloud in wanted if recipe.body_hull.get(cloud.name, False)]
        if recipe.hulls != "none" and (root or kids >= 2 or (kids == 1 and recipe.hulls == "all")):
            hulls += _trunk(tree, recipe, g, wanted, inertia=not clouds.get(g))
        elif recipe.hulls != "none" and kids == 0:
            near = (*(() if tree.kind[g] == "prismatic" else (g,)), tree.group[g])  # a sleeve stays behind its block
            hulls += _end(tree, recipe, g, wanted, reach[g], [glyphs[j] for j in near if j in glyphs])
        else:
            hulls += _trunk(tree, recipe, g, forced, inertia=False)
    return hulls


def _trunk(tree: _Tree, recipe: SkeletonRecipe, group: int, clouds: Sequence[_Cloud], *, inertia: bool) -> list[_Hull]:
    """Trunk hulls of a group, its body first.

    The bulkiest frame is the body: a box, axis-aligned in that frame around its points and padded by 0.7 of their
    median radius, then cut to 0.8 of its half extents about its centre. Any other frame is a ball on the body, no
    wider than 0.8 of the body's smallest extent (a head never wider than its trunk), and nothing at all when it
    sits inside the body; ``body_hull`` makes it a box. With ``inertia`` the body is the solid box that has the
    group's inertia, aligned with the group's frame, each half extent between 0.6 reference radii and 12 % of the
    robot, cut the same way. ``_tapered`` gives the body its shape.
    """

    def bulk(cloud: _Cloud) -> float:
        extent = cloud.points.max(axis=0) - cloud.points.min(axis=0)
        return float(np.sum(cloud.radii**3)) if cloud.radii is not None else float(np.prod(extent))

    hulls: list[_Hull] = []
    uncut: _OrientedBox | None = None  # the body's box before the cut
    for cloud in sorted(clouds, key=bulk, reverse=True):
        pad = _TRUNK_PAD * float(np.median(cloud.radii)) if cloud.radii is not None else 0.0
        low, high = cloud.points.min(axis=0) - pad, cloud.points.max(axis=0) + pad
        center, half = cloud.origin + cloud.rotation @ ((low + high) / 2), (high - low) / 2
        if uncut is not None and not recipe.body_hull.get(cloud.name, False):
            radius = min(float(np.median(half)), _HEAD_CAP * _TRUNK * float(uncut[2].min()))
            inside = np.all(np.abs(uncut[1].T @ (center - uncut[0])) <= uncut[2] + 0.5 * radius)
            if inside or any(  # a part round a joint of the body, or a part touching a head already drawn
                hull.mesh == "balls" and np.all(np.abs(hull.center - center) < half + hull.half) for hull in hulls
            ):
                continue
            hulls.append(
                _Hull(cloud.name, "hull", group, cloud.joint, center, cloud.rotation, np.full(3, radius), "balls")
            )
        else:
            uncut = uncut or (center, cloud.rotation, half)
            hulls.append(
                _tapered(tree, _Hull(cloud.name, "hull", group, cloud.joint, center, cloud.rotation, _TRUNK * half))
            )
    body = group if group else next((b for b in range(1, tree.model.njoints) if tree.anchor[b] == 0), 0)
    solid = _inertia(tree, group) if inertia and recipe.body_hull.get(tree.model.body_names[body], True) else None
    if solid is not None:
        mass, center, total = solid
        moments = np.diag(tree.rotation[body].T @ total @ tree.rotation[body])
        sides = np.sqrt(np.maximum(6.0 * (moments.sum() - 2.0 * moments) / mass, 0.0))
        half = _TRUNK * np.clip(sides / 2, 0.6 * tree.reference, 0.12 * tree.scale)
        hulls.append(
            _tapered(tree, _Hull(tree.model.body_names[body], "hull", group, body, center, tree.rotation[body], half))
        )
    return hulls


def _tapered(tree: _Tree, hull: _Hull) -> _Hull:
    """A box hull as a trunk when its group hangs from a joint and has child joints and the box is longest along
    the line between them: wide at the child joints and narrow at the joint it hangs from. A box of such a group
    that is longest across that line (a palm) narrows across its thinnest side; any other box stays as it is."""
    group = hull.group
    if not group or tree.kind[group] == "free" or not tree.children[group]:
        return hull
    toward = hull.rotation.T @ (tree.origin[list(tree.children[group])].mean(axis=0) - tree.origin[group])
    if np.linalg.norm(toward) < 1e-6 * tree.scale:
        return hull
    axis = int(np.abs(toward).argmax())
    if hull.half[axis] < hull.half.max():  # not a trunk: a block narrowing across its thinnest side
        thin = int(hull.half.argmin())
        columns = [(thin + 1) % 3, (thin + 2) % 3, thin]
        return replace(hull, rotation=hull.rotation[:, columns], half=hull.half[columns])
    first, second = (axis + 1) % 3, (axis + 2) % 3
    if toward[axis] > 0.0:  # +z must run from the wide side at the children to the narrow side
        first, second = second, first
    columns = [first, second, axis]
    rotation = hull.rotation[:, columns] * np.array([1.0, 1.0, -np.sign(toward[axis])])
    return replace(hull, rotation=rotation, half=hull.half[columns])


def _end(
    tree: _Tree,
    recipe: SkeletonRecipe,
    group: int,
    clouds: Sequence[_Cloud],
    radius: float,
    glyphs: Sequence[_OrientedBox],
) -> list[_Hull]:
    """The end of a leaf group, from the points of all its frames in the leaf joint's frame: a sole when they are
    flat, else a limb with a ball at its tip.

    Sphere centres are padded by 0.4 of the smaller of the joint radius and their median radius; the radii are left
    out on purpose: the skeleton is slimmer than the robot, and a block of the collision volume would be out of
    scale. Without geometry the points are the joint and twice the centre of mass, padded by 0.4 joint radii, and
    there is no end when that is shorter than a quarter joint radius: the glyph is the end.

    A limb runs from the joint through the middle of the points' box to its side, as thick as a bone, and ends in a
    ball as wide as the box's middle half extent, between 0.55 and 1 joint radius (a hand, a paw), whose far side is
    at the box's side unless that makes it touch ``glyphs``. A sole is the points' box, 0.45 joint radii
    thick; it gives way to ``glyphs``, the leaf's drum and the glyph before it: along the limb it starts where a
    glyph that it would swallow stops. A foot becomes a plate under its ankle drums instead of a block around them.
    """
    origin, rotation, name = tree.origin[group], tree.rotation[group], tree.model.body_names[group]
    pad = _END_PAD * radius
    if clouds:
        points = np.concatenate([cloud.origin + cloud.points @ cloud.rotation.T for cloud in clouds])
        radii = [cloud.radii for cloud in clouds if cloud.radii is not None]
        pad = min(pad, _END_PAD * float(np.median(np.concatenate(radii)))) if radii else 0.0
    else:
        solid = _inertia(tree, group) if recipe.body_hull.get(name, True) else None
        if solid is None or np.linalg.norm(solid[1] - origin) < 0.25 * radius:
            return []
        points = np.stack([origin, 2.0 * solid[1] - origin])
    local = (points - origin) @ rotation
    thin, mid, _ = np.sort(np.ptp(local, axis=0))
    low, high = local.min(axis=0) - pad, local.max(axis=0) + pad
    middle, half = (low + high) / 2, (high - low) / 2
    if thin >= _FLAT * mid and tree.kind[group] != "prismatic":  # a limb; a slider is a block
        tip = float(np.clip(np.sort(half)[1], _TIP * radius, radius))
        if np.linalg.norm(middle) < 0.25 * radius:  # the body sits on its joint: a ball there
            return [_Hull(name, "end", group, group, origin + rotation @ middle, rotation, np.full(3, tip), "balls")]
        unit = middle / np.linalg.norm(middle)
        side = min(float((high if u > 0.0 else low)[k] / u) for k, u in enumerate(unit) if abs(u) > 1e-9)
        direction = rotation @ unit
        clear = max((float((c - origin) @ direction + np.abs(direction @ f) @ e) for c, f, e in glyphs), default=0.0)
        length = max(side - tip, clear + tip)
        end = origin + length * direction
        frame = _limb_frame(direction, tree.axis.get(group))
        limb = np.array([recipe.bones * radius, recipe.bones * radius, length / 2])
        return [
            _Hull(name, "end", group, group, (origin + end) / 2, frame, limb),
            _Hull(name, "end", group, group, end, frame, np.full(3, tip), "balls"),
        ]
    if thin < _FLAT * mid:  # a sole: a plate on the far side of the points, its underside where theirs is
        axis = int(np.argmin(np.ptp(local, axis=0)))
        side = high[axis] if abs(high[axis]) >= abs(low[axis]) else low[axis]
        half[axis] = _PLATE * radius / 2
        middle[axis] = side - np.sign(side) * half[axis]
        long = int(half.argmax())  # it narrows along its length towards its joint: a heel, then a wider toe
        first, second = (long + 1) % 3, (long + 2) % 3
        toward = -middle[long]  # from the plate's middle to the joint
        if toward < 0.0:  # +z runs to the joint; the columns stay right-handed
            first, second = second, first
        columns = [first, second, long]
        rotation = rotation[:, columns] * np.array([1.0, 1.0, np.sign(toward) or 1.0])
        half = half[columns]
        return [_Hull(name, "end", group, group, origin + tree.rotation[group] @ middle, rotation, half)]
    return [_Hull(name, "end", group, group, origin + rotation @ middle, rotation, half)]


# ------------------------------------------------------------------------------------------------ the blocks being built


@final
class _Rows:
    """The blocks of a skeleton while it is built: world placements at neutral, packed into joint frames at the end."""

    def __init__(self, tree: _Tree, recipe: SkeletonRecipe) -> None:
        self.tree, self.recipe = tree, recipe
        self.rows: dict[str, list[tuple]] = {mesh: [] for mesh in _MESHES}
        self.points: list[tuple[int, np.ndarray]] = [(j, np.zeros(3)) for j in range(tree.model.njoints)]
        self.weights: list[Mapping[int, float]] = []  # per anchor, point index to weight
        self.where: list[np.ndarray] = []  # per anchor, its place at neutral
        self.lines: list[tuple[int, np.ndarray, np.ndarray]] = []

    def add(
        self, mesh: str, part: str, body: int, center: np.ndarray, scale: Sequence[float] | np.ndarray, *,
        rotation: np.ndarray | None = None, owner: int = -1, name: str = "", anchors: tuple[int, int] = (-1, -1),
    ) -> None:  # fmt: skip
        """One block of ``part``, rigid in joint ``body`` at ``center`` and ``rotation`` (world, at neutral) or
        moving between ``anchors``; ``owner`` is the joint it draws, ``name`` the frame of a hull."""
        tree, side = self.tree, 0
        if self.recipe.sides:  # left is positive y of the root body, by more than 2 % of the robot
            root = tree.root[owner if owner >= 0 else body]
            offset = float(
                tree.rotation[root][:, 1] @ ((tree.origin[owner] if owner >= 0 else center) - tree.origin[root])
            )
            side = int(np.sign(offset)) if abs(offset) > 0.02 * tree.scale else 0
        named = part not in ("bone", "stick", "fill")  # a recipe colours glyphs and hulls, never what joins them
        color = self.recipe.color.get(name or tree.model.joint_names[owner]) if named and (name or owner >= 0) else None
        rotation = np.eye(3) if rotation is None else rotation
        self.rows[mesh].append(
            (body, center, rotation, np.asarray(scale, dtype=np.float64), anchors, owner, part, side, color)
        )

    def bone(
        self, body: int, start: np.ndarray, end: np.ndarray, radius: float, owner: int, *, across: np.ndarray | None
    ) -> None:
        """A limb, a block ``2 radius`` wide from ``start`` (its wide end) to ``end`` (world, at neutral) with its
        sides along ``across``, rigid in joint ``body``; nothing when they coincide."""
        length = float(np.linalg.norm(end - start))
        if length > 1e-6 and radius > 0.0:
            rotation = _limb_frame((end - start) / length, across)
            self.add(
                "boxes",
                "bone",
                body,
                (start + end) / 2,
                (2 * radius, 2 * radius, length),
                rotation=rotation,
                owner=owner,
            )

    def hull(self, hull: _Hull) -> None:
        """The block of ``hull``; an end block belongs to its leaf joint, and the ball at its tip is a tip."""
        scale = hull.half if hull.mesh == "balls" else 2 * hull.half
        owner = hull.group if hull.part == "end" else -1
        part = "tip" if hull.part == "end" and hull.mesh == "balls" else hull.part
        self.add(hull.mesh, part, hull.body, hull.center, scale, rotation=hull.rotation, owner=owner, name=hull.name)

    def anchor(self, members: Mapping[int, float], where: np.ndarray) -> int:
        """A new anchor, the weighted mean of the points ``members`` (index to weight), at ``where`` at neutral."""
        self.weights.append(members)
        self.where.append(where)
        return len(self.weights) - 1

    def point(self, body: int, where: np.ndarray) -> int:
        """A new anchor rigid in joint ``body`` at ``where`` (world, at neutral)."""
        self.points.append((body, self.tree.rotation[body].T @ (where - self.tree.origin[body])))
        return self.anchor({len(self.points) - 1: 1.0}, where)

    def stick(self, start: int, end: int, radius: float, owner: int) -> None:
        """A cylinder that follows anchors ``start`` and ``end``; nothing when they coincide at neutral."""
        if np.linalg.norm(self.where[end] - self.where[start]) > 1e-6 and radius > 0.0:
            center = (self.where[start] + self.where[end]) / 2
            self.add(
                "cylinders",
                "stick",
                self.tree.root[owner],
                center,
                (radius, radius, 1.0),
                owner=owner,
                anchors=(start, end),
            )

    def pack(
        self, *, slides: Mapping[str, float], shrinks: Mapping[str, float], clusters: tuple[tuple[str, ...], ...]
    ) -> Skeleton:
        """The finished ``Skeleton``: every world placement expressed in its joint's frame."""
        tree = self.tree
        packed = {}
        for mesh, rows in self.rows.items():
            body = np.array([row[0] for row in rows], dtype=np.int64)
            world = np.zeros((len(rows), 7))
            if rows:
                world[:, :3] = np.stack([row[1] for row in rows])
                world[:, 3:] = so3.from_matrix(torch.from_numpy(np.stack([row[2] for row in rows]))).numpy()
            inverse = se3.inverse(torch.from_numpy(tree.neutral[body]))
            packed[mesh] = Blocks(
                body=body,
                local=se3.compose(inverse, torch.from_numpy(world)).numpy(),
                scale=np.array([row[3] for row in rows]).reshape(len(rows), 3),
                anchors=np.array([row[4] for row in rows], dtype=np.int64).reshape(len(rows), 2),
                owner=np.array([row[5] for row in rows], dtype=np.int64),
                part=tuple(row[6] for row in rows),
                side=np.array([row[7] for row in rows], dtype=np.int64),
                color=tuple(row[8] for row in rows),
            )
        weights = np.zeros((len(self.weights), len(self.points)))
        for k, members in enumerate(self.weights):
            weights[k, list(members)] = list(members.values())
        line_local = np.zeros((len(self.lines), 2, 3))
        for i, (body, start, end) in enumerate(self.lines):
            line_local[i] = (np.stack([start, end]) - tree.origin[body]) @ tree.rotation[body]
        return Skeleton(
            scale=tree.scale,
            radius=tree.reference,
            taper=_TAPER if self.recipe.glyphs == "kind" else 1.0,
            **packed,
            points_body=np.array([body for body, _ in self.points], dtype=np.int64),
            points_local=np.stack([local for _, local in self.points]),
            weights=weights,
            line_body=np.array([body for body, _, _ in self.lines], dtype=np.int64),
            line_local=line_local,
            slides=slides,
            shrinks=shrinks,
            clusters=clusters,
        )


# ------------------------------------------------------------------------------------------------ a glyph per joint kind


def _maximal(
    tree: _Tree, recipe: SkeletonRecipe, rows: _Rows, clouds: Mapping[int, list[_Cloud]], *,
    reach: Mapping[int, float], slide: Mapping[int, float],
) -> None:  # fmt: skip
    """Glyphs per joint kind, bones from glyph centre to glyph centre, hulls and end blocks tied to their glyph.

    A drum is symmetric about its axis, so it is rigid in the parent group as well as in the child: a bone, the hull
    of a group and the glyphs of its child joints are all rigid in that group, and the layout holds for every
    configuration. Every glyph contains the ball of 0.8 of its radius about its centre and a bone is thinner, so a
    bone that runs centre to centre has both caps inside the glyphs. ``reach[j]`` is the drawn radius of glyph
    ``j``, 0 for a hidden one.
    """
    origin = tree.origin
    center = {j: origin[j] + slide[j] * tree.axis[j] if j in tree.axis else origin[j] for j in tree.glyphs}
    frame = {j: _matrix(_from_z(tree.axis[j])) if j in tree.axis else tree.rotation[j] for j in tree.glyphs}

    def thickness(*ends: int) -> float:
        shown = [reach[j] for j in ends if j and reach[j] > 0.0]
        return recipe.bones * (min(shown) if shown else tree.reference)

    solid = {j: (center[j], frame[j], _half(tree, j, reach[j])) for j in tree.glyphs if tree.kind[j] != "free"}
    hulls = _hulls(tree, recipe, clouds, reach, {j: box for j, box in solid.items() if reach[j] > 0.0})
    trunk: dict[int, _Hull] = {}
    for hull in hulls:
        if hull.part == "hull" and hull.mesh != "balls":
            trunk.setdefault(hull.group, hull)
    for j in tree.glyphs:
        for mesh, part, in_parent, scale in _GLYPHS[tree.kind[j]] if reach[j] > 0.0 else ():
            body = tree.body(tree.group[j], j) if in_parent else j
            place, size = center[j], reach[j] * np.array(scale)
            travel = _travel(tree, j) if part == "rail" else None
            if travel is not None:  # the rail spans where the sleeve can be, so it never leaves its parent
                place, size[2] = center[j] + sum(travel) / 2 * tree.axis[j], travel[1] - travel[0]
            rows.add(mesh, part, body, place, size, rotation=frame[j], owner=j)
        if reach[j] > 0.0 and recipe.limits and isinstance(tree.model.joints[j], JointRevolute):
            _limit_arc(tree, rows, j, center[j], reach[j])
        if reach[j] <= 0.0 and tree.kind[j] != "free":  # a joint without its glyph: a ball rounds the bend of its bones
            fill = thickness(tree.group[j], *tree.children[j])
            rows.add("balls", "fill", j, center[j], (fill, fill, fill), owner=j)
        if recipe.bones > 0.0 and tree.kind[j] != "free":
            group = tree.group[j]
            if group in trunk:  # a limb leaves its trunk from the trunk's centre
                body, start = trunk[group].body, trunk[group].center
            else:
                body, start = tree.body(group, j), center[group] if group else origin[tree.root[j]]
            rows.bone(body, start, center[j], thickness(group, j), j, across=tree.axis.get(j, tree.axis.get(group)))
    for hull in hulls:
        rows.hull(hull)
        if recipe.bones > 0.0 and hull.part == "hull":  # the spine from the group's glyph, the neck from the trunk
            main = trunk.get(hull.group)
            start = main.center if hull.mesh == "balls" and main is not None else origin[hull.body]
            start = center[hull.group] if hull.group and hull.mesh != "balls" else start
            across = tree.axis.get(hull.group)
            rows.bone(hull.body, start, hull.center, thickness(hull.group), hull.group or hull.body, across=across)


def _travel(tree: _Tree, joint: int) -> tuple[float, float] | None:
    """How far a prismatic ``joint`` moves from its neutral place, ``(lowest, highest)`` along its axis; ``None``
    without two finite limits. A mimic joint moves as its source does, times its multiplier."""
    model, source = tree.model, tree.model.mimic_source_ids[joint]
    index = model.idx_qs[source]
    lower, upper = float(model.lower_position_limit[index]), float(model.upper_position_limit[index])
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        return None
    ends = model.mimic_multipliers[joint] * (np.array([lower, upper]) - float(model.q_neutral[index]))
    return float(ends.min()), float(ends.max())


def _limit_arc(tree: _Tree, rows: _Rows, joint: int, center: np.ndarray, radius: float) -> None:
    """The arc from the lower to the upper limit of a bounded revolute joint, rigid in the parent, and a tick at the
    current angle, rigid in the child."""
    model = tree.model
    if model.nqs[joint] != 1:  # a mimic joint has no limits of its own
        return
    index = model.idx_qs[joint]
    lower, upper = float(model.lower_position_limit[index]), float(model.upper_position_limit[index])
    if not 0.0 < upper - lower < 2.0 * np.pi:
        return
    frame = _matrix(_from_z(tree.axis[joint]))
    zero = float(model.q_neutral[index])  # the tick's angle at neutral

    def spoke(angle: float, reach: float) -> np.ndarray:
        return center + reach * radius * (np.cos(angle - zero) * frame[:, 0] + np.sin(angle - zero) * frame[:, 1])

    angles = np.linspace(lower, upper, _ARC + 1)
    parent = tree.body(tree.group[joint], joint)
    rows.lines += [(parent, spoke(a, 1.15), spoke(b, 1.15)) for a, b in zip(angles[:-1], angles[1:], strict=True)]
    rows.lines.append((joint, spoke(zero, 1.0), spoke(zero, 1.3)))


# ------------------------------------------------------------------------------------------------ a ball per cluster


def _clusters(tree: _Tree, recipe: SkeletonRecipe) -> tuple[dict[int, list[int]], dict[int, int], set[int]]:
    """Clusters of close joints: ``(members per cluster, cluster of each glyph, clusters not drawn)``.

    Two joints share a cluster when they are parent and child, or siblings, and closer than ``cluster`` times the
    robot's size. With ``twists=False`` a lone twist joint is not drawn.
    """
    label = {j: j for j in tree.glyphs}

    def find(j: int) -> int:
        while label[j] != j:
            j = label[j]
        return j

    for j in tree.glyphs:
        siblings = [i for i in tree.children[tree.group[j]] if i < j]
        for i in (*siblings, *((tree.group[j],) if tree.group[j] else ())):
            if np.linalg.norm(tree.origin[j] - tree.origin[i]) < recipe.cluster * tree.scale:
                label[find(j)] = find(i)
    cluster = {j: find(j) for j in tree.glyphs}
    members: dict[int, list[int]] = {}
    for j in tree.glyphs:
        members.setdefault(cluster[j], []).append(j)
    lone = [] if recipe.twists else [c for c, joints in members.items() if len(joints) == 1]
    one = {c: tree.children[c] for c in lone if len({cluster[j] for j in tree.children[c]}) == 1}  # one ball after it
    return members, cluster, {c for c, after in one.items() if _twists(tree, c, after)}


def _minimal(
    tree: _Tree, recipe: SkeletonRecipe, rows: _Rows, clouds: Mapping[int, list[_Cloud]]
) -> tuple[tuple[str, ...], ...]:
    """A ball per cluster of close joints and sticks between the balls; returns the joint names behind each ball.

    A ball sits at the mean of its member origins at every step; the cluster of a free-flyer root is rigid in the
    root body and drawn as a cube. A ball is at most 0.4 of its shortest stick, and a stick never thicker than the
    balls it joins.
    """
    members, cluster, skipped = _clusters(tree, recipe)
    radius, anchor, base = {}, {}, {}
    for c, joints in members.items():
        if c in skipped:
            continue
        where = tree.origin[joints].mean(axis=0)
        radius[c] = float(np.mean([tree.radius[j] for j in joints]))
        floating = [j for j in joints if tree.kind[j] == "free" and tree.group[j] == 0]
        if floating:
            anchor[c], base[c] = rows.point(floating[0], where), floating[0]
        else:
            anchor[c] = rows.anchor({j: 1.0 / len(joints) for j in joints}, where)
    hulls = _hulls(tree, recipe, clouds, {j: radius.get(cluster[j], 0.0) for j in tree.glyphs}, {})
    sticks = _sticks(tree, rows, hulls, members=members, cluster=cluster, anchor=anchor)
    ends: dict[int, list[float]] = {}
    for start, end, _ in sticks:
        length = float(np.linalg.norm(rows.where[end] - rows.where[start]))
        for a in (start, end) if length > 0.01 * tree.scale else ():
            ends.setdefault(a, []).append(length)
    drawn = {anchor[c]: min(radius[c], _FIXED_CAP * min(ends.get(anchor[c], [np.inf]))) for c in radius}
    for c, a in anchor.items():
        r, joint = drawn[a], members[c][0]
        if c in base:
            rows.add(
                "boxes",
                "base",
                base[c],
                rows.where[a],
                (1.7 * r, 1.7 * r, 1.7 * r),
                rotation=tree.rotation[base[c]],
                owner=joint,
            )
        elif recipe.glyphs == "ball":
            rows.add("balls", "ball", tree.root[joint], rows.where[a], (r, r, r), owner=joint, anchors=(a, a))
    for hull in hulls:
        rows.hull(hull)
    for start, end, owner in sticks:
        balls = [drawn[a] for a in (start, end) if a in drawn and recipe.glyphs == "ball"]
        rows.stick(start, end, min([recipe.bones * tree.reference, *balls]), owner)
    return tuple(tuple(tree.model.joint_names[j] for j in members[c]) for c in anchor)


def _sticks(
    tree: _Tree, rows: _Rows, hulls: Sequence[_Hull], *,
    members: Mapping[int, list[int]], cluster: Mapping[int, int], anchor: Mapping[int, int],
) -> list[tuple[int, int, int]]:  # fmt: skip
    """The sticks of the stick figure as ``(anchor, anchor, joint whose side tints it)``.

    Each ball hangs from the ball of the nearest drawn group above it. A group with a hull, or with two or more
    balls below it, joins them through a hub that is rigid in the group: the centre of its body hull, or the mean
    of its child joints (spine, hip bar, shoulder bar). A fixed base gets a cube at the root body's origin and a
    stick up to each first ball; an end block hangs from its ball.
    """
    origin = tree.origin

    def above(joint: int) -> int:
        while joint and cluster[joint] not in anchor:  # a twist that is not drawn: the stick runs through
            joint = tree.group[joint]
        return joint

    below: dict[int, list[int]] = {}
    for c in anchor:
        below.setdefault(above(tree.group[members[c][0]]), []).append(c)
    sticks, bases = [], {}
    for group, kids in below.items():
        own = cluster[group] if group else -1
        first = {c: members[c][0] for c in kids if c != own}
        boxes = [hull for hull in hulls if hull.group == group and hull.part == "hull" and hull.mesh != "balls"]
        for c, joint in first.items() if group == 0 else ():
            root = tree.root[joint]
            if root != joint and root not in bases:
                bases[root] = rows.point(root, origin[root])
                side = 1.7 * tree.reference
                rows.add(
                    "boxes", "base", root, origin[root], (side, side, side), rotation=tree.rotation[root], owner=root
                )
            if root != joint:  # else the first joint moves the root body itself: nothing is below it
                sticks.append((bases[root], anchor[c], joint))
        if group and first and (len(first) >= 2 or boxes):
            where = boxes[0].center if boxes else np.mean([origin[joint] for joint in first.values()], axis=0)
            hub = rows.point(group, where)
            sticks += [(anchor[own], hub, group), *((hub, anchor[c], joint) for c, joint in first.items())]
        elif group:
            sticks += [(anchor[own], anchor[c], joint) for c, joint in first.items()]
    for hull in hulls:
        if hull.group and hull.part == "end":
            sticks.append((anchor[cluster[above(hull.group)]], rows.point(hull.body, hull.center), hull.group))
    return sticks


# ------------------------------------------------------------------------------------------------ the three unit meshes


def unit_cylinder(sides: int = 48) -> tuple[np.ndarray, np.ndarray]:
    """Closed cylinder of radius 1 along z from ``-0.5`` to ``0.5``: ``(vertices (V, 3), faces (F, 3))``.

    The caps have their own ring of vertices, so the rim stays sharp under smooth shading.
    """
    angles = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)
    ring = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    low = np.concatenate([ring, np.full((sides, 1), -0.5)], axis=1)
    high = np.concatenate([ring, np.full((sides, 1), 0.5)], axis=1)
    vertices = np.concatenate([low, high, low, high, [[0.0, 0.0, -0.5], [0.0, 0.0, 0.5]]])
    i = np.arange(sides)
    j = (i + 1) % sides
    faces = np.concatenate(
        [
            np.stack([i, j, sides + j], axis=1),
            np.stack([i, sides + j, sides + i], axis=1),
            np.stack([np.full(sides, 4 * sides), 2 * sides + j, 2 * sides + i], axis=1),
            np.stack([np.full(sides, 4 * sides + 1), 3 * sides + i, 3 * sides + j], axis=1),
        ]
    )
    return vertices.astype(np.float32), faces.astype(np.uint32)


def unit_ball(rings: int = 16, sides: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Sphere of radius 1 as ``rings`` bands of ``sides`` quads: ``(vertices (V, 3), faces (F, 3))``."""
    polar = np.linspace(0.0, np.pi, rings + 1)[1:-1, None]
    turn = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)[None, :]
    grid = np.stack(
        [np.sin(polar) * np.cos(turn), np.sin(polar) * np.sin(turn), np.cos(polar) * np.ones_like(turn)], axis=-1
    )
    vertices = np.concatenate([grid.reshape(-1, 3), [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]])
    top, bottom = len(vertices) - 2, len(vertices) - 1
    i = np.arange(sides)
    j = (i + 1) % sides
    faces = [np.stack([np.full(sides, top), i, j], axis=1)]
    for ring in range(rings - 2):
        a, b = ring * sides, (ring + 1) * sides
        faces += [np.stack([a + i, b + i, b + j], axis=1), np.stack([a + i, b + j, a + j], axis=1)]
    last = (rings - 2) * sides
    faces.append(np.stack([np.full(sides, bottom), last + j, last + i], axis=1))
    return vertices.astype(np.float32), np.concatenate(faces).astype(np.uint32)


def unit_box(chamfer: float = 0.12, taper: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Cube of side 1 with chamfered edges and corners, narrowed towards ``+z``, to draw flat-shaded: ``(vertices
    (24, 3), faces (44, 3))``.

    Each corner is inset by ``chamfer`` along two of its three axes, which gives three vertices per corner; their
    convex hull has six square faces, twelve edge strips and eight corner triangles. Then x and y are scaled from 1 at
    ``z = -0.5`` to ``taper`` at ``z = 0.5``: a linear scale in z maps planes to planes, so the faces stay flat.
    """
    corners = np.array([[x, y, z] for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)])
    kept = np.eye(3, dtype=bool)[None]  # per vertex, the axis that stays on the cube's face
    vertices = np.where(kept, corners[:, None, :], corners[:, None, :] * (1.0 - 2.0 * chamfer)).reshape(24, 3)
    faces = []
    for normal in np.ndindex(3, 3, 3):  # the 26 outward directions: 6 faces, 12 edges, 8 corners
        direction = np.array(normal, dtype=np.float64) - 1.0
        if not direction.any():
            continue
        height = vertices @ direction
        ring = np.flatnonzero(height > height.max() - 1e-9)
        across = np.cross(direction, [1.0, 0.0, 0.0] if abs(direction[0]) < 0.5 else [0.0, 1.0, 0.0])
        spread = vertices[ring] - vertices[ring].mean(axis=0)
        ring = ring[np.argsort(np.arctan2(spread @ np.cross(direction, across), spread @ across))]
        faces += [(ring[0], ring[k], ring[k + 1]) for k in range(1, len(ring) - 1)]
    vertices[:, :2] *= 1.0 + (taper - 1.0) * (vertices[:, 2:] + 0.5)
    return vertices.astype(np.float32), np.array(faces, dtype=np.uint32)
