"""The abstract skeleton of a robot: ``SkeletonRecipe`` and the layout that turns a ``Model`` into blocks.

A skeleton is drawn from three unit meshes (a cylinder, a ball and a chamfered box), each instanced many times.
``build_skeleton`` solves the layout once, at the neutral configuration: sizes from the robot's extent and the mass
each joint carries, one glyph per joint kind (a drum for a revolute joint, a sleeve and rail for a prismatic one, a ball for a
spherical one, a thin plate for a planar one, a block for a translation, a cube for a free-flyer), or with
``glyphs="ball"`` one ball per cluster of close joints, joined by rods or sticks whose ends follow them: a stick
figure. Close joints never overlap: a drum slides along its own axis until it clears its
neighbours and shrinks only when no slide does, so every axis line stays exact and joints that share a point (a
hip, a waist, a knuckle) come out as a short stack of separate drums. Every block is rigid in one joint frame or
follows two anchors, so ``Skeleton.place`` poses a step with one pose composition per block.

Nothing here imports viser: the robot visual sends the arrays. Poses are ``[tx, ty, tz, qx, qy, qz, qw]``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from numbers import Real
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
    JointPlanar,
    JointPrismatic,
    JointRevolute,
    JointRevoluteUnbounded,
    JointSpherical,
    JointTranslation,
    Mesh,
    Model,
    Sphere,
)
from .layer import from_z
from .theme import RGB, Theme

_TWIST = float(np.cos(np.radians(35.0)))  # a joint axis within 35 degrees of its limb twists the limb
_CARRY = 0.026  # m: the glyph radius of a joint that carries 1 kg; it grows with the cube root of the mass
_HALF = 0.8  # half length of a drum over its radius
_SLEEVE = 0.6  # half length of a sleeve along its rail over the joint radius
_MARKER = 0.6  # radius of the plain marker of an unlisted joint kind over the joint radius
_TRUNK_PAD = 0.7  # a trunk's box pads its sphere centres by this share of their median radius
_HEAD_CAP = 0.64  # a second frame of a body is a ball of at most this share of the body's smallest half extent
_HEAD_MIN = 0.3  # and of at least this share: a smaller frame (a sensor) is no head
_TIP = (0.55, 0.7)  # the smallest and the largest ball that ends a leaf, over the leaf joint's radius
_LEAF_CAP = 1.25  # a leaf glyph is at most this share of the half thickness of the body it moves
_LEAF_FLOOR = 0.2  # and at least this share of the reference radius: a thin placeholder shape is not a body
_FIXED_CAP = 0.4  # a glyph that cannot slide is at most this share of its shortest bone
_LOW = 0.65  # a glyph of its kind is at least this many reference radii: a wrist, an ankle, a knee reads as a motor
_REACH = 1.75  # a glyph slides at most this many of its radii from its start, unless no size down to _KEEP clears there
_KEEP = 0.7  # a glyph keeps this share of its size while a slide of any length clears it
_LEAST = 0.1  # the least share of its size a glyph shrinks to when nothing clears it nearer
_SPACING = 0.35  # on rods, a drum is at most this share of its shortest bone, not under _ROD rod radii
_ROD = 2.5  # on rods, a drum is at least this many rod radii, and a rod thins beside a smaller drum
_ARC = 24  # segments of a limit arc
_KINDS = {
    JointFixed: "fixed", JointRevolute: "revolute", JointRevoluteUnbounded: "revolute", JointHelical: "revolute",
    JointPrismatic: "prismatic", JointSpherical: "spherical", JointFreeFlyer: "free", JointPlanar: "planar",
    JointTranslation: "translation",
}  # fmt: skip
#: Per joint kind, its blocks: (unit mesh, part, rigid in the parent group or in the child, scale over the radius).
_GLYPHS = {
    "revolute": (("cylinders", "drum", True, (1.0, 1.0, 2 * _HALF)), ("cylinders", "face", True, (0.8, 0.8, 2.2 * _HALF))),
    "prismatic": (("boxes", "sleeve", False, (1.6, 1.6, 2 * _SLEEVE)), ("cylinders", "rail", True, (0.5, 0.5, 4.4 * _HALF))),
    "spherical": (("balls", "ball", False, (1.0, 1.0, 1.0)),),
    "free": (("boxes", "base", False, (1.5, 1.5, 1.5)),),
    "planar": (("boxes", "plate", False, (2.0, 2.0, 0.4)),),  # a thin plate in the joint's xy plane
    "translation": (("boxes", "block", False, (1.2, 1.2, 1.2)),),
    "other": (("balls", "marker", False, (1.0, 1.0, 1.0)),),
}  # fmt: skip
_Shade = Literal["joint", "face", "body", "bone", "stick"]
#: The shade each part of the skeleton takes from the robot's hue: light rods, mid drums, dark drum faces. A
#: ``head`` (the ball of a head over its trunk, where the glyphs are drawn) and a ``cap`` (the round end of the rod
#: that ends a chain) take the rods' shade: no ball but a spherical joint's reads as a joint. A ``knob`` (the small
#: ball that ends a chain of balls, or their head) takes the shade of the balls but draws no joint.
_ROLE: dict[str, _Shade] = {
    "drum": "joint", "sleeve": "joint", "ball": "joint", "knob": "joint", "plate": "joint", "block": "joint",
    "face": "face", "marker": "face", "base": "face",
    "rod": "bone", "rail": "bone", "head": "bone", "cap": "bone",
    "stick": "stick",
}  # fmt: skip
_MESHES = ("cylinders", "balls", "boxes")
# The blocks that draw a joint; a free base is no motor.
_JOINT_PARTS = ("drum", "face", "sleeve", "ball", "plate", "block", "marker")


@final
@dataclass(frozen=True, kw_only=True)
class SkeletonRecipe:
    """How a robot's skeleton is drawn; ``MEDIUM`` (the defaults) and ``MINIMAL`` are the two presets.

    Parameters
    ----------
    glyphs
        ``"kind"`` draws one glyph per joint kind joined by rods, ``"ball"`` one ball per cluster of close joints
        joined by sticks: a stick figure, whose sticks run on from the last joint of a chain to the end of its body
        (to a small ball there with ``"ball"``, else to a round cap of the rod's width), and whose trunks draw only
        their heads, as balls.
    bones
        Radius of a rod or stick over the reference glyph radius; 0 draws none.
    limits
        Draw the limit arc of each bounded revolute joint, with a tick at the current angle (``glyphs="kind"``).
    cluster
        Joints closer than ``cluster`` times the robot's size share one ball (``glyphs="ball"``; a glyph of its kind
        is always its joint's own).
    twists
        ``False`` draws no glyph for a lone joint whose axis runs along its limb; the rod runs through.
    sides
        Tint the left side of the robot lighter and the right side darker.
    size
        Multiplies every glyph radius.
    joint_size
        Joint name to a multiplier of its glyph's radius, applied after the layout's size rules, so it changes that
        glyph only: 0 hides it and the rods run on through its place, and the glyphs after it keep their size. The
        factor wins over the layout's narrowing towards the tips, so above 1 a glyph may be wider than the one
        before it. A ball of ``glyphs="ball"`` takes the mean factor of the joints it stands for.
    joint_radius
        The radius in metres of every joint's glyph, or of the joints named, in place of the layout's size rules; a
        glyph of its kind keeps it where it meets another. ``joint_size`` still multiplies it; ``None`` keeps the
        layout's sizes.
    bodies
        ``True`` lays the sticks along the robot's bodies: two or more child joints hang from a hub (a left and right
        pair from a bar), a chain runs on to the end of its last body and a trunk shows its head. ``False`` joins
        each joint to its parent joint and stops at the last joint of a chain, the plain joint tree.
    color
        Joint name to the colour of its glyph, in place of the robot's hue.
    """

    glyphs: Literal["kind", "ball"] = "kind"
    bones: float = 0.35
    limits: bool = False
    cluster: float = 0.0
    twists: bool = True
    sides: bool = False
    size: float = 0.7
    joint_size: Mapping[str, float] = field(default_factory=dict)
    joint_radius: float | Mapping[str, float] | None = None
    bodies: bool = True
    color: Mapping[str, RGB] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Raise ``ArgumentError`` for an unknown ``glyphs`` word or a size out of range or not finite."""
        if self.glyphs not in ("kind", "ball"):
            raise ArgumentError(f"glyphs must be 'kind' or 'ball', got {self.glyphs!r}")
        if (
            not all(math.isfinite(value) for value in (self.bones, self.cluster, self.size, *self.joint_size.values()))
            or self.bones < 0.0
            or self.cluster < 0.0
            or self.size <= 0.0
            or min(self.joint_size.values(), default=0.0) < 0.0
        ):
            raise ArgumentError(
                f"bones, cluster and joint_size must be finite and at least 0, size finite and above 0, got bones={self.bones}, "
                f"cluster={self.cluster}, size={self.size}, joint_size={dict(self.joint_size)}"
            )
        radii = self.joint_radius.values() if isinstance(self.joint_radius, Mapping) else (self.joint_radius,)
        if self.joint_radius is not None and not all(isinstance(r, Real) and 0.0 < r < math.inf for r in radii):
            raise ArgumentError(
                f"joint_radius must be a finite radius above 0 or a mapping of them, got {self.joint_radius!r}"
            )


MEDIUM = SkeletonRecipe()
MINIMAL = SkeletonRecipe(glyphs="ball", bones=0.42, cluster=0.06, twists=False, size=0.62)


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
    owner: np.ndarray  # (N,) the joint a block draws or belongs to
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

    ``scale`` is the robot's size (diagonal of the box of its joint origins at neutral), ``radius`` the reference
    glyph radius and ``radii (njoints,)`` the radius each joint's glyph is sized by, before ``joint_size`` (a fixed
    joint's is that of the joint its body is welded to, the universe's the reference). ``slides`` (metres along the joint's own axis) and ``shrinks`` (factor in ``[0, 1]``, 0: no glyph fits) list, by
    joint name, the glyphs the layout moved or shrank.
    """

    scale: float
    radius: float
    radii: np.ndarray  # (njoints,)
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
        meshes: list[Blocks] = [getattr(self, name) for name in _MESHES]
        body = np.concatenate([blocks.body for blocks in meshes])
        local = torch.from_numpy(np.concatenate([blocks.local for blocks in meshes])).to(poses.dtype)
        every = se3.compose(poses[..., body, :], local).numpy()  # one composition for every block
        placed, first = {}, 0
        for name, blocks in zip(_MESHES, meshes, strict=True):
            world, first = every[..., first : first + len(blocks), :], first + len(blocks)
            scale = blocks.scale
            moving = blocks.anchors[:, 0] >= 0
            if anchors is not None and moving.any():
                start, end = anchors[..., blocks.anchors[moving, 0], :], anchors[..., blocks.anchors[moving, 1], :]
                world[..., moving, :3] = (start + end) / 2
                if name == "cylinders":
                    world[..., moving, 3:] = from_z(end - start)
                    scale = np.broadcast_to(scale, (*world.shape[:-1], 3)).astype(world.dtype)
                    scale[..., moving, 2] = np.linalg.norm(end - start, axis=-1)
            placed[name] = (world, scale)
        return placed

    def lines(self, joint_poses: np.ndarray) -> np.ndarray:
        """World end points ``(..., L, 2, 3)`` of the limit arcs and ticks at ``joint_poses (..., njoints, 7)``."""
        poses = torch.from_numpy(np.ascontiguousarray(joint_poses))
        local = torch.from_numpy(self.line_local).to(poses.dtype)
        return se3.act(poses[..., self.line_body, None, :], local).numpy()

    def glyphs(self, mesh: str, njoints: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The blocks of ``mesh`` that draw joints: ``(rows (n,), joints (n, njoints) bool, faces (n,) bool)``.

        A glyph of its kind (a drum and its faces, a sleeve, a ball, a plate, a block, a marker) stands for its
        joint, a ball of ``glyphs="ball"`` for every joint of its cluster; ``faces`` marks the blocks in the dark
        shade of a drum's faces. A stick or rod, a rail, a head, a knob, a cap and the cube of a free base draw no
        joint.
        """
        blocks: Blocks = getattr(self, mesh)
        rows = np.array([i for i, part in enumerate(blocks.part) if part in _JOINT_PARTS], dtype=np.int64)
        joints = np.zeros((len(rows), njoints), dtype=bool)
        for k, i in enumerate(rows):
            anchor = blocks.anchors[i, 0]  # a ball of a cluster sits at the mean of its joints' origins
            joints[k] = self.weights[anchor, :njoints] > 0.0 if anchor >= 0 else np.arange(njoints) == blocks.owner[i]
        faces = np.array([_ROLE[blocks.part[i]] == "face" for i in rows], dtype=bool)
        return rows, joints, faces

    def colors(self, hue: RGB, theme: Theme) -> dict[str, np.ndarray]:
        """Colours ``(N, 3)`` in ``[0, 1]`` of every block: ``theme.shade`` of ``hue`` for its part and side, or the
        recipe's colour for it (a drum face takes that colour's shade)."""
        colors = {}
        for name in _MESHES:
            blocks: Blocks = getattr(self, name)
            rows = np.zeros((len(blocks), 3), dtype=np.float32)
            for i, (part, side, own) in enumerate(zip(blocks.part, blocks.side, blocks.color, strict=True)):
                role = _ROLE[part]
                if own is None:
                    rows[i] = theme.shade(hue, role, side=int(side))
                else:
                    rows[i] = theme.shade(own, role) if role == "face" else own
            colors[name] = rows
        return colors


def build_skeleton(
    model: Model,
    recipe: SkeletonRecipe,
    *,
    spheres: RobotSpheres | None = None,
    geometry: Geometry | None = None,
    mesh_vertices: Callable[[int, Mesh], np.ndarray | None] | None = None,
) -> Skeleton:
    """Lay out the skeleton of ``model`` under ``recipe``, from one forward-kinematics pass at ``q_neutral``.

    Leaf glyphs, heads and the ends of chains are sized from the first source a body has: the collision shapes of
    ``geometry`` (a mesh only through ``mesh_vertices(body, mesh)``, which returns the vertices ``(V, 3)`` of its
    file, or ``None`` to leave it out), the ``spheres`` on its frames, its inertia. A name of ``recipe`` that the
    model does not have raises ``UnknownNameError``.
    """
    given = recipe.joint_radius
    for name in (*recipe.joint_size, *recipe.color, *(given if isinstance(given, Mapping) else ())):
        if name not in model.joint_names:
            raise UnknownNameError("joint", name, valid=model.joint_names)
    tree = _tree(model)
    clouds = _clouds(tree, spheres, geometry, mesh_vertices)
    tree = _leaf_caps(_sizes(tree, recipe, clouds), clouds, kind=recipe.glyphs == "kind")
    factor = {j: recipe.joint_size.get(model.joint_names[j], 1.0) for j in tree.glyphs}
    rows = _Rows(tree, recipe)
    if recipe.glyphs == "ball":
        tree = _given(tree, recipe)
        _stick_figure(tree, recipe, rows, clouds, factor=factor)
        return rows.pack(slides={}, shrinks={}, radii=tree.radius)
    # on rods, a row of close drums (a wrist) keeps rod between them
    spaced = {j: _SPACING * min(tree.bones(j), default=np.inf) for j in tree.glyphs}
    least = _ROD * recipe.bones * tree.reference
    radius = {
        j: r if tree.kind[j] != "revolute" else max(min(r, spaced[j]), min(r, least)) for j, r in tree.radius.items()
    }
    tree = _given(replace(tree, radius=radius), recipe)
    # joint_size scales a glyph after the size rules, so the glyphs after it keep theirs
    drawn = replace(tree, radius={j: factor[j] * tree.radius[j] for j in tree.glyphs})
    hidden = {j for j in tree.glyphs if _mimic(tree, j) or drawn.radius[j] <= 0.0}
    if not recipe.twists:
        hidden |= {j for j, after in tree.children.items() if j and len(after) == 1 and _twists(tree, j, after)}
    slide, shrink = _solve(drawn, hidden, clouds)
    if given is not None:  # a given radius is drawn as given, also where glyphs meet
        named = given if isinstance(given, Mapping) else model.joint_names
        shrink = {j: 1.0 if model.joint_names[j] in named else k for j, k in shrink.items()}
    reach = {j: 0.0 if j in hidden else shrink[j] * drawn.radius[j] for j in tree.glyphs}
    ends = {j: max(shrink[j], _LEAST) * tree.radius[j] for j in tree.glyphs}  # a chain's end is sized by it: never 0
    _stick_figure(tree, recipe, rows, clouds, factor=factor, glyphs=(slide, reach, ends))
    return rows.pack(
        slides={model.joint_names[j]: s for j, s in slide.items() if abs(s) > 1e-9},
        shrinks={model.joint_names[j]: k for j, k in shrink.items() if k < 1.0},
        radii=ends,
    )


# ------------------------------------------------------------------------------------------------ numpy helpers


def _numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy().astype(np.float64)


def _matrix(quaternion: np.ndarray) -> np.ndarray:
    """Rotation matrices ``(..., 3, 3)`` of ``xyzw`` quaternions ``(..., 4)`` (``so3.to_matrix``)."""
    return so3.to_matrix(torch.from_numpy(np.ascontiguousarray(quaternion, dtype=np.float64))).numpy()


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

    A rigid group is a moving joint plus the bodies welded to it by fixed joints; group 0 is the world, and what is
    rigid in group ``g`` is rigid in joint ``g``'s frame (the universe's for the world, also for a joint that hangs
    from the world itself). ``anchor[j]`` is the group of joint ``j``, ``glyphs`` the moving joints, ``group[j]`` the
    parent group of glyph ``j`` and ``children[g]`` the glyphs whose parent group is ``g``.
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

    def bones(self, joint: int) -> list[float]:
        """Distances from glyph ``joint`` to its parent and child glyphs, those under 1 % of the robot ignored."""
        ends = (*((self.group[joint],) if self.group[joint] else ()), *self.children[joint])
        lengths = [float(np.linalg.norm(self.origin[joint] - self.origin[i])) for i in ends]
        return [length for length in lengths if length > 0.01 * self.scale]


def _tree(model: Model) -> _Tree:
    """The tree of ``model`` at ``q_neutral``, without sizes yet (``_sizes``)."""
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
    return _Tree(
        model, kind, tuple(anchor), tuple(root), glyphs, group, children, origin, rotation, neutral, axis, 0.0, 0.0, {}
    )


def _sizes(tree: _Tree, recipe: SkeletonRecipe, clouds: Mapping[int, list[_Cloud]]) -> _Tree:
    """``tree`` with the sizes of ``recipe``.

    The robot's size is the diagonal of the box of its joint origins; when they span nothing (one moving joint,
    joints that share a point, a lone floating body) the box also takes in the ``clouds`` and the solid box of each
    rigid group's inertia, and only a robot with none of those is 1 m. The reference radius is a third of the mean
    bone, kept between 2 % and 3 % of the size. A joint's radius grows with the mass it carries, as an actuator
    does: 26 mm times the cube root of the mass of its subtree in kg (a 1 kg forearm 26 mm, a 10 kg leg 56 mm, a
    50 g finger 10 mm), kept under 1.2 reference radii, so a wrist is smaller than its elbow, a finger than its
    wrist, a chain narrows towards its tips and mirror sides match; a robot without mass takes the reference. A ball
    of ``glyphs="ball"`` is at least 0.5 reference radii. A glyph of its kind is at least 0.65 of them and 2.5 rod
    radii (0.875 of them), so a wrist, an ankle or a knee reads as a motor at the distance of the whole robot;
    a joint inside a chain that carries grams (a finger of a hand on an arm) is at most twice its mass size instead,
    and a leaf at least half a reference radius (``_leaf_caps`` caps it by its body). Both times ``size``. A glyph
    that cannot slide is no thicker than 0.4 of its shortest bone. Every style shares the reference, so a small
    robot's balls are as small for it as a large robot's.
    """
    model, origin, glyphs, group = tree.model, tree.origin, tree.glyphs, tree.group
    points = [origin[1:] if model.njoints > 1 else origin]
    if np.ptp(points[0], axis=0).max() <= 1e-9:
        for cloud in (cloud for found in clouds.values() for cloud in found):
            pad = 0.0 if cloud.radii is None else cloud.radii[:, None]
            points += [cloud.origin + cloud.points @ cloud.rotation.T + sign * pad for sign in (-1.0, 1.0)]
        for solid in filter(None, (_inertia(tree, g) for g in (0, *glyphs))):
            mass, center, total = solid
            half = _solid(mass, np.diag(total))
            points.append(np.stack([center - half, center + half]))
    extent = np.concatenate(points)
    scale = float(np.linalg.norm(np.ptp(extent, axis=0)))
    scale = scale if scale > 1e-9 else 1.0  # nothing to measure: a unit robot
    spans = [float(np.linalg.norm(origin[j] - origin[group[j]])) for j in glyphs if group[j]]
    spans = [span for span in spans if span > 0.01 * scale]
    reference = float(np.clip(np.mean(spans) / 3.0, 0.02 * scale, 0.03 * scale)) if spans else 0.03 * scale
    tree = replace(tree, scale=scale, reference=reference * recipe.size)
    kind = tree.kind
    carried = _numpy(model.body_inertias)[:, 0].copy()  # per joint, the mass of its subtree
    for j in range(model.njoints - 1, 0, -1):
        carried[model.parent_joint_ids[j]] += carried[j]
    radius = {}
    floor = max(_LOW, _ROD * recipe.bones)
    for j in glyphs:
        own = _CARRY * np.cbrt(carried[j]) if carried[0] > 0.0 and kind[j] != "free" else reference
        low = 0.5
        if recipe.glyphs == "kind":
            low = max(0.0 if tree.children[j] else 0.5, min(floor, 2.0 * own / reference))
        radius[j] = recipe.size * float(np.clip(own, low * reference, 1.2 * reference))
        radius[j] *= _MARKER if kind[j] == "other" else 1.0
        if kind[j] in ("spherical", "planar", "translation", "other") and tree.bones(j):  # glyphs that cannot slide
            radius[j] = min(radius[j], _FIXED_CAP * min(tree.bones(j)))
    return replace(tree, radius=radius)


def _given(tree: _Tree, recipe: SkeletonRecipe) -> _Tree:
    """``tree`` with the radii of ``recipe.joint_radius`` in place of the layout's."""
    given, names = recipe.joint_radius, tree.model.joint_names
    if given is None:
        return tree
    sized = given if isinstance(given, Mapping) else dict.fromkeys(names, given)
    return replace(tree, radius={j: float(sized.get(names[j], r)) for j, r in tree.radius.items()})


def _leaf_caps(tree: _Tree, clouds: Mapping[int, list[_Cloud]], *, kind: bool) -> _Tree:
    """``tree`` with each leaf glyph no thicker than the body it moves: at most 1.25 times the median radius of the
    spheres it carries or, for a glyph of its ``kind``, of the middle half extent of its collision points in its
    own frame, but never under 0.2 reference radii (Tiago's actuated finger joints carry 2 mm placeholder boxes)."""
    radius = dict(tree.radius)
    for j in tree.glyphs:
        if tree.children[j] or j not in clouds:
            continue
        radii = [cloud.radii for cloud in clouds[j] if cloud.radii is not None]
        if radii:
            size = float(np.median(np.concatenate(radii)))
        elif not kind:  # a ball is capped by spheres only
            continue
        else:
            points = np.concatenate([cloud.origin + cloud.points @ cloud.rotation.T for cloud in clouds[j]])
            size = float(np.sort(np.ptp((points - tree.origin[j]) @ tree.rotation[j], axis=0))[1]) / 2
        radius[j] = min(radius[j], max(_LEAF_CAP * size, _LEAF_FLOOR * tree.reference))
    return replace(tree, radius=radius)


def _mimic(tree: _Tree, joint: int) -> bool:
    """Whether ``joint`` is a revolute mimic joint: it moves as its source does and has no motor of its own, so it
    draws no glyph (a gripper's linkage). A mimic slider keeps its sleeve: two fingers that close on each other
    read as a pair."""
    return tree.kind[joint] == "revolute" and tree.model.mimic_source_ids[joint] != joint


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
    distal glyph keeps its place and a proximal one gives way; a leaf gives way last because its drum ends its
    chain. A glyph is tested against every placed glyph closer to it than the sum of their sizes (each
    with room to slide): the glyphs of its rigid groups (parent, siblings, children), where the clearance holds for
    every configuration, and any other, where it holds at neutral (a hip's yaw two hops from its pitch, a waist
    beside a hip, a finger beside its neighbour). The cube of a free base is placed as a glyph that cannot slide:
    the drums of a waist on it slide clear of it.
    """
    origin = tree.origin
    solid = [j for j in tree.glyphs if j not in hidden]
    frame = {j: _matrix(from_z(tree.axis[j])) if j in tree.axis else tree.rotation[j] for j in solid}  # as placed
    slide = {j: 0.0 for j in tree.glyphs}
    shrink = {j: 1.0 for j in tree.glyphs}
    for j in solid:  # a drum on a twist joint starts with its far face on the joint: a collar on the limb
        limb = origin[j] - (origin[tree.group[j]] if tree.group[j] else origin[tree.root[j]])
        if np.linalg.norm(limb) <= tree.radius[j] and tree.children[j]:  # on its parent: away from its children
            limb = origin[list(tree.children[j])].mean(axis=0) - origin[j]
        length = float(np.linalg.norm(limb))
        along = (
            float(tree.axis[j] @ limb) / length if tree.kind[j] == "revolute" and length > 0.01 * tree.scale else 0.0
        )
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

    size = {j: 2.5 * float(np.linalg.norm(_half(tree, j, tree.radius[j]))) for j in solid}  # with room to slide
    placed: list[int] = []

    def across(j: int) -> bool:  # on the root body's mid (x-z) plane, its axis across it: it cannot slide sideways
        root = tree.rotation[tree.root[j]]
        middle = abs(float((origin[j] - origin[tree.root[j]]) @ root[:, 1])) < 0.02 * tree.scale
        return j in tree.axis and middle and abs(float(tree.axis[j] @ root[:, 1])) > _TWIST

    for j in sorted(solid, key=lambda j: (not tree.children[j], not across(j), j)):
        boxes = {i: box(i) for i in placed}
        hard = [b for i, b in boxes.items() if np.linalg.norm(b[0] - origin[j]) < abs(slide[j]) + size[j] + size[i]]
        if hard:
            slide[j], shrink[j] = _fit(tree, j, slide[j], frame[j], hard)
        placed.append(j)
    return slide, shrink


def _half(tree: _Tree, joint: int, reach: float) -> np.ndarray:
    """Half extents of the box around the glyph of ``joint`` drawn with radius ``reach``: a drum with its faces or a
    sleeve along z, a free base's cube, else a ball."""
    along = _SLEEVE if tree.kind[joint] == "prismatic" else 1.1 * _HALF
    if joint not in tree.axis:
        return reach * np.full(3, 0.75 if tree.kind[joint] == "free" else 1.0)
    return reach * np.array([1.0, 1.0, along])


def _fit(
    tree: _Tree, joint: int, base: float, frame: np.ndarray, others: Sequence[_OrientedBox]
) -> tuple[float, float]:
    """The slide and shrink of one glyph against its placed neighbours ``others``.

    Candidates step a quarter of the half length away from ``base``, alternating sides, first the side away from
    the root body (so left and right come out as mirror images), until just past the half diagonal of the widest
    neighbour. The first one clear of every box by 0.1 radius and no farther than 1.75 radii from ``base`` wins (the
    glyph stays by its joint); with none the glyph shrinks by 0.85 and tries again, down to 0.7 of its size. Then
    the farther candidates count as well, from the full size down to 0.7, and only when none clears there, down to
    a tenth: a glyph slides rather than shrinks under 0.7. A glyph that clears at no size down to a tenth is not
    drawn (shrink 0: a hand whose finger joints sit 7 mm apart). A ball or a marker has the one candidate and only
    shrinks.
    """
    radius, origin, axis = tree.radius[joint], tree.origin[joint], tree.axis.get(joint)
    sign = 1.0
    if axis is not None:
        away = float((origin - tree.origin[tree.root[joint]]) @ axis)
        if abs(away) < 1e-6 * tree.scale:  # level with the root: the axis's largest component in the root frame
            in_root = np.round(tree.rotation[tree.root[joint]].T @ axis, 6)
            away = float(in_root[np.abs(in_root).argmax()])
        sign = 1.0 if away >= 0.0 else -1.0
    for reach, least in ((_REACH * radius, _KEEP), (np.inf, _KEEP), (np.inf, _LEAST)):
        shrink = 1.0
        while True:
            half, offsets = _half(tree, joint, shrink * radius), [0.0]
            if axis is not None:
                step = 0.25 * half[2]
                steps = int(1.25 * (half[2] + max(float(np.linalg.norm(other[2])) for other in others)) / step)
                offsets += [side * m * step for m in range(1, steps + 1) for side in (sign, -sign)]
            candidates = base * shrink + np.array(offsets)
            centers = origin + candidates[:, None] * (axis if axis is not None else np.zeros(3))
            gaps = _gaps(centers, frame, half, others)
            clear = (gaps >= 0.1 * shrink * radius) & (np.abs(candidates - base * shrink) <= reach)
            if clear.any():
                return float(candidates[np.argmax(clear)]), shrink
            if shrink <= least:
                break
            shrink = max(least, 0.85 * shrink)
    return base, 0.0  # nothing clears: no glyph


# ------------------------------------------------------------------------------------------------ the bodies' geometry


@final
@dataclass(frozen=True)
class _Cloud:
    """Geometry of one frame: ``points (n, 3)`` in the frame at ``rotation``, ``origin`` (world, at neutral)."""

    joint: int
    rotation: np.ndarray
    origin: np.ndarray
    points: np.ndarray
    radii: np.ndarray | None  # per point for a sphere set; None when the points lie on the surface


_Head = tuple[int, np.ndarray]  # a head: the joint it is rigid in, its centre (world, at neutral)


def _clouds(
    tree: _Tree,
    spheres: RobotSpheres | None,
    geometry: Geometry | None,
    mesh_vertices: Callable[[int, Mesh], np.ndarray | None] | None,
) -> dict[int, list[_Cloud]]:
    """Per rigid group, the geometry its leaf glyph, head and chain end are sized from: its collision shapes, else
    its sphere set. A sphere set is an overlay first: the look of a body with collision shapes does not change with
    it."""
    model = tree.model
    found: dict[int, list[_Cloud]] = {}
    shapes: dict[int, list[np.ndarray]] = {}
    for attached in geometry.shapes if geometry is not None else ():
        points = _shape_points(attached.body, attached.shape, mesh_vertices) if attached.collision else None
        if points is not None:
            placement = np.array(attached.shape.placement)
            shapes.setdefault(attached.body, []).append(points @ _matrix(placement[3:]).T + placement[:3])
    for joint, parts in shapes.items():
        cloud = _Cloud(joint, tree.rotation[joint], tree.origin[joint], np.concatenate(parts), None)
        found.setdefault(tree.anchor[joint], []).append(cloud)
    if spheres is not None:
        frame_ids, offsets, radii = spheres.frame_ids.cpu().numpy(), _numpy(spheres.offsets), _numpy(spheres.radii)
        placements = _numpy(model.frame_placements)
        shaped = set(found)
        for f in np.unique(frame_ids):
            joint, placement, mask = model.frame_parent_joint_ids[f], placements[f], frame_ids == f
            if tree.anchor[joint] in shaped:
                continue
            rotation = tree.rotation[joint] @ _matrix(placement[3:])
            origin = tree.origin[joint] + tree.rotation[joint] @ placement[:3]
            cloud = _Cloud(joint, rotation, origin, offsets[mask], radii[mask])
            found.setdefault(tree.anchor[joint], []).append(cloud)
    return found


def _shape_points(
    body: int, shape: object, mesh_vertices: Callable[[int, Mesh], np.ndarray | None] | None
) -> np.ndarray | None:
    """Points ``(n, 3)`` on the surface of a shape on ``body``, in the shape's own frame; ``None`` for a mesh without
    ``mesh_vertices`` or whose vertices it does not give."""
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
        vertices = mesh_vertices(body, shape)
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


def _solid(mass: float, moments: np.ndarray) -> np.ndarray:
    """Half sides ``(3,)`` of the uniform solid box of ``mass`` whose moments of inertia about its axes are
    ``moments (3,)``."""
    return np.sqrt(np.maximum(6.0 * (moments.sum() - 2.0 * moments) / mass, 0.0)) / 2


def _crowns(tree: _Tree, clouds: Mapping[int, list[_Cloud]], heads: Mapping[int, _Head]) -> dict[int, int]:
    """The leaf groups that are heads, each to the first joint of its neck: of each trunk (the root, or two or more child joints) without a head in
    ``heads``, the end of a chain of at most two joints from it (a neck's pan and tilt) whose first joint is above
    every other joint but the trunk's own along the root body's z, and whose geometry is round (no more than twice as
    long as wide): a Talos, an Atlas, a Tiago."""
    crowned = {}
    for g in (0, *tree.glyphs):
        if g in heads or not (g == 0 or (tree.group[g] == 0 and tree.kind[g] == "free") or len(tree.children[g]) > 1):
            continue
        height = tree.origin @ tree.rotation[tree.root[g]][:, 2]
        for first in tree.children[g]:
            chain = (first, *tree.children[first])
            leaf, others = chain[-1], [j for j in tree.glyphs if j not in (g, *chain)]
            if len(chain) > 2 or tree.children[leaf] or not clouds.get(leaf) or tree.kind[first] == "free":
                continue
            if height[first] <= max((height[j] for j in others), default=-np.inf):
                continue
            points = np.concatenate([c.origin + c.points @ c.rotation.T for c in clouds[leaf]]) @ tree.rotation[leaf]
            long, wide = np.sort(np.ptp(points, axis=0))[:0:-1]
            if long <= 2.0 * wide:
                crowned[leaf] = first
    return crowned


def _head(tree: _Tree, recipe: SkeletonRecipe, group: int, clouds: Sequence[_Cloud]) -> _Head | None:
    """The head of trunk ``group`` (a floating root, or two or more child joints), if it has one.

    The bulkiest frame is the body: a box, axis-aligned in that frame around its points and padded by 0.7 of their
    median radius; the floating root's takes in the solid box of its inertia, so a stand-in shape does not leave the
    legs without a pelvis (an H1's 5 cm sphere), and the box of a moving body reaches its child joints' origins: a
    chest spans its shoulders (an H1, whose torso shape is a small block). One other frame may be the head: the
    bulkiest that sits outside the body, is round (no more than twice as long as wide) and at least 0.3 of the
    body's smallest half extent, and no wider than 0.64 of it; sensors, antennas and columns are no head. A trunk
    without such a head whose frames rise above its body along the root body's z (an H1's cameras and lidar) has a
    head over the body, midway between its top and their mean. Beside the drums of ``glyphs="kind"`` a head stands
    above every joint along the root body's z, its centre at least its radius over the top of its body: a
    quadruped's snout or the lidar on its back, a sensor on a pelvis or on a gripper's wrist is no head there (among
    the balls of ``glyphs="ball"`` it is).
    """

    def bulk(cloud: _Cloud) -> float:
        return (
            float(np.sum(cloud.radii**3)) if cloud.radii is not None else float(np.prod(np.ptp(cloud.points, axis=0)))
        )

    up = tree.rotation[tree.root[group]][:, 2]
    highest = max((float(tree.origin[j] @ up) for j in tree.glyphs), default=-np.inf)

    def rises(center: np.ndarray, radius: float, body: _OrientedBox) -> bool:  # on its body, above every joint
        over = float((center - body[0]) @ up) - float(np.abs(body[1].T @ up) @ body[2])
        return recipe.glyphs == "ball" or (over >= radius and float(center @ up) > highest)

    body: _OrientedBox | None = None
    root = tree.group[group] == 0 and tree.kind[group] == "free"
    for cloud in sorted(clouds, key=bulk, reverse=True):
        pad = _TRUNK_PAD * float(np.median(cloud.radii)) if cloud.radii is not None else 0.0
        low, high = cloud.points.min(axis=0) - pad, cloud.points.max(axis=0) + pad
        if body is None and not root:  # a moving body reaches its branches
            kids = (tree.origin[list(tree.children[group])] - cloud.origin) @ cloud.rotation
            low, high = np.minimum(low, kids.min(axis=0)), np.maximum(high, kids.max(axis=0))
        solid = _inertia(tree, group) if body is None and root else None
        if solid is not None:  # the robot's own body takes in its inertia's box
            middle = cloud.rotation.T @ (solid[1] - cloud.origin)
            half = _solid(solid[0], np.diag(cloud.rotation.T @ solid[2] @ cloud.rotation))
            low, high = np.minimum(low, middle - half), np.maximum(high, middle + half)
        center, half = cloud.origin + cloud.rotation @ ((low + high) / 2), (high - low) / 2
        if body is None:
            body = (center, cloud.rotation, half)
            continue
        radius = min(float(np.median(half)), _HEAD_CAP * float(body[2].min()))
        inside = np.all(np.abs(body[1].T @ (center - body[0])) <= body[2] + 0.5 * radius)  # round a joint
        long, wide = np.sort(half)[:0:-1]
        small = radius < _HEAD_MIN * float(body[2].min()) or long > 2.0 * wide
        if not (inside or small or not rises(center, radius, body)):
            return cloud.joint, center
    if body is None:
        return None
    placements = _numpy(tree.model.frame_placements)
    top = float(np.abs(body[1].T @ up) @ body[2])  # the body's half extent along up
    above = [
        where
        for f, joint in enumerate(tree.model.frame_parent_joint_ids)
        if tree.anchor[joint] == group
        and float(((where := tree.origin[joint] + tree.rotation[joint] @ placements[f, :3]) - body[0]) @ up) > top
    ]
    # over the body, midway between its top and the frames: they mark the top of a head, not its middle
    center = body[0] + up * (top + float((np.mean(above, axis=0) - body[0]) @ up)) / 2 if above else body[0]
    return (group, center) if above and rises(center, tree.reference, body) else None


def _extent(tree: _Tree, group: int, clouds: Sequence[_Cloud]) -> np.ndarray | None:
    """World points ``(n, 3)`` at neutral that span the body of leaf ``group``: the points of its frames' geometry,
    else the joint and twice its centre of mass; ``None`` without either."""
    if clouds:
        return np.concatenate([cloud.origin + cloud.points @ cloud.rotation.T for cloud in clouds])
    solid = _inertia(tree, group)
    return None if solid is None else np.stack([tree.origin[group], 2.0 * solid[1] - tree.origin[group]])


def _tip(tree: _Tree, group: int, clouds: Sequence[_Cloud]) -> tuple[np.ndarray, bool]:
    """Where the body of leaf ``group`` ends (world, at neutral), and whether a ball that ends it is centred there.

    The end is the middle of the points of the leaf's geometry farthest from the joint (the sole of a paw, the end
    of a hand, a finger, a toe): the ball ends there, so a foot stands on the ground. Without geometry it is the
    leaf's frame farthest from the joint when that frame lies past the middle of the body its inertia spans (a foot,
    a tool tip), the ball centred on it, else the far end of that body. Both styles end a leaf here.
    """
    origin, model = tree.origin[group], tree.model
    placements = _numpy(model.frame_placements)
    frames = [
        tree.origin[joint] + tree.rotation[joint] @ placements[f, :3]
        for f, joint in enumerate(model.frame_parent_joint_ids)
        if tree.anchor[joint] == group
    ]
    frame = max(frames, key=lambda where: float(np.linalg.norm(where - origin)), default=origin)
    points, far = _extent(tree, group, clouds), origin
    if points is not None:
        reach = np.linalg.norm(points - origin, axis=1)
        far = points[reach >= reach.max() - 0.01 * tree.scale].mean(axis=0)
    if not clouds and np.linalg.norm(frame - origin) > 0.5 * np.linalg.norm(far - origin):
        return frame, True
    return far, False


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
        owner: int, rotation: np.ndarray | None = None, anchors: tuple[int, int] = (-1, -1),
    ) -> None:  # fmt: skip
        """One block of ``part``, rigid in joint ``body`` at ``center`` and ``rotation`` (world, at neutral) or
        moving between ``anchors``; ``owner`` is the joint it draws or belongs to."""
        tree, side = self.tree, 0
        if self.recipe.sides:  # left is positive y of the root body, by more than 2 % of the robot
            root = tree.root[owner]
            offset = float(tree.rotation[root][:, 1] @ (tree.origin[owner] - tree.origin[root]))
            side = int(np.sign(offset)) if abs(offset) > 0.02 * tree.scale else 0
        named = part not in ("stick", "rod", "cap")  # a recipe colours glyphs, never what joins them
        color = self.recipe.color.get(tree.model.joint_names[owner]) if named else None
        rotation = np.eye(3) if rotation is None else rotation
        self.rows[mesh].append(
            (body, center, rotation, np.asarray(scale, dtype=np.float64), anchors, owner, part, side, color)
        )

    def anchor(self, members: Mapping[int, float], where: np.ndarray) -> int:
        """A new anchor, the weighted mean of the points ``members`` (index to weight), at ``where`` at neutral."""
        self.weights.append(members)
        self.where.append(where)
        return len(self.weights) - 1

    def point(self, body: int, where: np.ndarray) -> int:
        """A new anchor rigid in joint ``body`` at ``where`` (world, at neutral)."""
        self.points.append((body, self.tree.rotation[body].T @ (where - self.tree.origin[body])))
        return self.anchor({len(self.points) - 1: 1.0}, where)

    def stick(self, start: int, end: int, radius: float, owner: int, *, part: str = "stick") -> None:
        """A cylinder that follows anchors ``start`` and ``end``; nothing when they coincide at neutral."""
        if np.linalg.norm(self.where[end] - self.where[start]) > 1e-6 and radius > 0.0:
            center = (self.where[start] + self.where[end]) / 2
            self.add(
                "cylinders",
                part,
                self.tree.root[owner],
                center,
                (radius, radius, 1.0),
                owner=owner,
                anchors=(start, end),
            )

    def pack(
        self, *, slides: Mapping[str, float], shrinks: Mapping[str, float], radii: Mapping[int, float]
    ) -> Skeleton:
        """The finished ``Skeleton``: every world placement expressed in its joint's frame; ``radii`` per glyph."""
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
            radii=np.array([radii.get(group, tree.reference) for group in tree.anchor]),
            **packed,
            points_body=np.array([body for body, _ in self.points], dtype=np.int64),
            points_local=np.stack([local for _, local in self.points]),
            weights=weights,
            line_body=np.array([body for body, _, _ in self.lines], dtype=np.int64),
            line_local=line_local,
            slides=slides,
            shrinks=shrinks,
        )


# ------------------------------------------------------------------------------------------------ a glyph per joint kind


def _placed(tree: _Tree, slide: Mapping[int, float]) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Centre (world, at neutral) and rotation of every glyph: slid along its joint's axis, the axis as its z."""
    center = {j: tree.origin[j] + slide[j] * tree.axis[j] if j in tree.axis else tree.origin[j] for j in tree.glyphs}
    frame = {j: _matrix(from_z(tree.axis[j])) if j in tree.axis else tree.rotation[j] for j in tree.glyphs}
    return center, frame


def _glyph(
    tree: _Tree, recipe: SkeletonRecipe, rows: _Rows, joint: int, center: np.ndarray, frame: np.ndarray, reach: float
) -> None:
    """The blocks of the glyph of ``joint``'s kind at ``center`` and ``frame``, of radius ``reach`` (nothing at 0),
    and its limit arc when ``recipe`` draws them."""
    for mesh, part, in_parent, scale in _GLYPHS[tree.kind[joint]] if reach > 0.0 else ():
        place, size = center, reach * np.array(scale)
        travel = _travel(tree, joint) if part == "rail" else None
        if travel is not None:  # the rail spans where the sleeve can be, so it never leaves its parent
            place, size[2] = center + sum(travel) / 2 * tree.axis[joint], travel[1] - travel[0]
        body = tree.group[joint] if in_parent else joint
        rows.add(mesh, part, body, place, size, rotation=frame, owner=joint)
    if reach > 0.0 and recipe.limits and isinstance(tree.model.joints[joint], JointRevolute) and joint in tree.axis:
        _limit_arc(tree, rows, joint, center, reach)


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
    frame = _matrix(from_z(tree.axis[joint]))
    zero = float(model.q_neutral[index])  # the tick's angle at neutral

    def spoke(angle: float, reach: float) -> np.ndarray:
        return center + reach * radius * (np.cos(angle - zero) * frame[:, 0] + np.sin(angle - zero) * frame[:, 1])

    angles = np.linspace(lower, upper, _ARC + 1)
    parent = tree.group[joint]
    rows.lines += [(parent, spoke(a, 1.15), spoke(b, 1.15)) for a, b in zip(angles[:-1], angles[1:], strict=True)]
    rows.lines.append((joint, spoke(zero, 1.0), spoke(zero, 1.3)))


# ------------------------------------------------------------------------------------------------ a ball per cluster


def _clusters(tree: _Tree, recipe: SkeletonRecipe) -> tuple[dict[int, list[int]], dict[int, int], set[int]]:
    """Clusters of close joints: ``(members per cluster, cluster of each glyph, clusters not drawn)``.

    Two joints share a cluster when they are parent and child, or siblings, and closer than ``cluster`` times the
    robot's size; a joint drawn as a glyph of its kind is a cluster of its own. With ``twists=False`` a lone twist
    joint is not drawn.
    """
    label = {j: j for j in tree.glyphs}
    close = recipe.cluster * tree.scale if recipe.glyphs == "ball" else 0.0

    def find(j: int) -> int:
        while label[j] != j:
            j = label[j]
        return j

    for j in tree.glyphs:
        siblings = [i for i in tree.children[tree.group[j]] if i < j]
        for i in (*siblings, *((tree.group[j],) if tree.group[j] else ())):
            if np.linalg.norm(tree.origin[j] - tree.origin[i]) < close:
                label[find(j)] = find(i)
    cluster = {j: find(j) for j in tree.glyphs}
    members: dict[int, list[int]] = {}
    for j in tree.glyphs:
        members.setdefault(cluster[j], []).append(j)
    lone = [] if recipe.twists else [c for c, joints in members.items() if len(joints) == 1]
    one = {c: tree.children[c] for c in lone if len({cluster[j] for j in tree.children[c]}) == 1}  # one ball after it
    return members, cluster, {c for c, after in one.items() if _twists(tree, c, after)}


def _stick_figure(
    tree: _Tree, recipe: SkeletonRecipe, rows: _Rows, clouds: Mapping[int, list[_Cloud]], *,
    factor: Mapping[int, float],
    glyphs: tuple[Mapping[int, float], Mapping[int, float], Mapping[int, float]] | None = None,
) -> None:  # fmt: skip
    """Sticks between the joints, and a ball per cluster of close joints.

    A ball sits at the mean of its member origins at every step; the cluster of a free-flyer root is rigid in the
    root body and drawn as a cube. A ball is at most 0.4 of its shortest stick, and a stick never thicker than the
    balls it joins (a rod beside glyphs: than 1 / 2.5 of them). A leaf joint gets a stick on to the end of its body
    (``_knob``) and a small ball there, rigid in its frame: a foot stands on the ground, a hand and a finger have their length. A trunk's
    head (``_heads``) is a ball as small as a joint's on a stick from the trunk; neither without ``recipe.bodies``. Then a ball takes the mean
    ``factor`` (``joint_size``) of its joints; a ball of factor 0, or of mimic joints only (``_mimic``), is not
    drawn, and its sticks keep their thickness.

    With ``glyphs``, the slide, drawn radius and undrawn radius of every joint (``build_skeleton``), each joint is
    its own cluster and draws the glyph of its kind instead of a ball, its sticks meeting at the glyph's centre; a
    chain ends in a round cap of its last rod's width and a head is a ``head`` in the rods' shade, as no ball
    beside the drums may read as a joint.
    """
    members, cluster, skipped = _clusters(tree, recipe)
    radius, share, anchor, base = {}, {}, {}, {}
    center, frame = _placed(tree, glyphs[0]) if glyphs is not None else ({}, {})
    for c, joints in members.items():
        if c in skipped:
            continue
        where = center[c] if glyphs is not None else tree.origin[joints].mean(axis=0)
        radius[c] = glyphs[2][c] if glyphs is not None else float(np.mean([tree.radius[j] for j in joints]))
        share[c] = 0.0 if all(_mimic(tree, j) for j in joints) else float(np.mean([factor[j] for j in joints]))
        floating = [j for j in joints if tree.kind[j] == "free" and tree.group[j] == 0]
        if floating:
            base[c] = floating[0]
        rigid = c if glyphs is not None else base.get(c)  # a glyph's centre lies on its joint's axis
        anchor[c] = (
            rows.anchor({j: 1.0 / len(joints) for j in joints}, where) if rigid is None else rows.point(rigid, where)
        )
    tip = "knob" if glyphs is None else "head"  # beside drums a head is pale: a ball there is no joint
    heads = _heads(tree, recipe, rows, clouds, part=tip) if recipe.bodies else {}
    crowned = _crowns(tree, clouds, heads) if recipe.bodies else {}
    sticks = _sticks(tree, rows, heads, members=members, cluster=cluster, anchor=anchor, bodies=recipe.bodies)
    ends: dict[int, list[float]] = {}
    for start, end, _ in sticks if glyphs is None else ():  # a glyph is sized by the slide rule instead
        length = float(np.linalg.norm(rows.where[end] - rows.where[start]))
        for a in (start, end) if length > 0.01 * tree.scale else ():
            ends.setdefault(a, []).append(length)
    drawn = {anchor[c]: min(radius[c], _FIXED_CAP * min(ends.get(anchor[c], [np.inf]))) for c in radius}
    capped: set[int] = set()
    for j in (j for j in tree.glyphs if not tree.children[j] and recipe.bodies):
        a = anchor[cluster[j]]
        knob = _knob(tree, recipe, rows, j, clouds.get(j, []), start=a, radius=drawn[a], part=tip, head=j in crowned)
        capped |= {end for _, end, _ in knob} if glyphs is not None and j not in crowned else set()
        sticks += knob
    ball = {anchor[c]: glyphs[1][c] if glyphs is not None else share[c] * drawn[anchor[c]] for c in radius}
    for c, a in anchor.items():
        if glyphs is not None:
            _glyph(tree, recipe, rows, c, center[c], frame[c], ball[a])
        elif ball[a] > 0.0:
            _ball(tree, rows, a, ball[a], members[c][0], base.get(c))
    part = "stick" if glyphs is None else "rod"  # between glyphs a rod, in the pale bone shade
    for start, end, owner in sticks:
        balls = [ball[a] / (1.0 if glyphs is None else _ROD) for a in (start, end) if ball.get(a, 0.0) > 0.0]
        side = min([recipe.bones * tree.reference, *balls])
        rows.stick(start, end, side, owner, part=part)
        if end in capped:  # the rod that ends a chain beside drums ends round, in a cap no wider than itself
            rows.add("balls", "cap", owner, rows.where[end], (side, side, side), owner=owner)


def _ball(tree: _Tree, rows: _Rows, anchor: int, radius: float, joint: int, base: int | None) -> None:
    """The ball of ``radius`` at ``anchor`` of the cluster of ``joint``, or the cube of the floating root ``base``."""
    if base is None:
        rows.add(
            "balls", "ball", tree.root[joint], rows.where[anchor], (radius,) * 3, owner=joint, anchors=(anchor, anchor)
        )
    else:
        side = (1.7 * radius,) * 3
        rows.add("boxes", "base", base, rows.where[anchor], side, rotation=tree.rotation[base], owner=joint)


def _heads(
    tree: _Tree, recipe: SkeletonRecipe, rows: _Rows, clouds: Mapping[int, list[_Cloud]], *, part: str
) -> dict[int, _Head]:
    """The heads of the trunks (``_head``: a second frame outside the body), each a ball of ``part`` as small as a
    joint's, added; returned per trunk group."""
    heads = {}
    for g in tree.glyphs:
        if (tree.group[g] == 0 and tree.kind[g] == "free") or len(tree.children[g]) >= 2:
            head = _head(tree, recipe, g, clouds.get(g, []))
            if head is not None:
                heads[g] = head
    for g, (body, center) in heads.items():
        rows.add("balls", part, body, center, np.full(3, tree.reference), owner=g)
    return heads


def _mirrored(tree: _Tree, joint: int, other: int) -> bool:
    """Whether the origins of ``joint`` and ``other`` mirror each other across the root body's x-z plane (left and
    right, each more than 2 % of the robot off it, within 2 % of the robot)."""
    root = tree.root[joint]
    a, b = ((tree.origin[j] - tree.origin[root]) @ tree.rotation[root] for j in (joint, other))
    side = 0.02 * tree.scale
    return abs(a[1]) > side and bool(np.linalg.norm(a * [1.0, -1.0, 1.0] - b) < side)


def _knob(
    tree: _Tree, recipe: SkeletonRecipe, rows: _Rows, group: int, clouds: Sequence[_Cloud], *, start: int,
    radius: float, part: str, head: bool,
) -> list[tuple[int, int, int]]:  # fmt: skip
    """The stick from the ball of leaf ``group`` (anchor ``start``, drawn radius ``radius``) on to the end of its
    body (``_tip``), and the small ball of ``part`` that ends it, both rigid in the leaf's frame; the stick is
    returned, the ball added. Nothing when the end is inside the leaf's ball. The ball's radius is 0.55 of the leaf's ball, or less to
    stay clear of it, and at least 1.25 times the stick's; a ``head``'s is a ball of the reference radius in the
    middle of its body. A ``part`` other than ``"knob"`` (the glyphs beside the rods) adds no ball but a head: the
    stick stops a stick's radius short of the end, where ``_stick_figure`` rounds it with a cap.
    """
    end, centred = _tip(tree, group, clouds)
    if head:
        points = (_extent(tree, group, clouds) - tree.origin[group]) @ tree.rotation[group]
        end = tree.origin[group] + tree.rotation[group] @ ((points.min(axis=0) + points.max(axis=0)) / 2)
    toward = end - rows.where[start]
    length = float(np.linalg.norm(toward))
    if length <= radius:
        return []
    clear = min(_TIP[0] * radius, (length - radius) / 2.0)  # clear of the leaf's ball: no blob round a short finger
    ball = tree.reference if head else max(1.25 * min(recipe.bones * tree.reference, radius), clear)
    ball = ball if head or part == "knob" else min(recipe.bones * tree.reference, radius)  # else the cap's radius
    end = end if centred or head else end - ball * toward / length  # a surface end: the ball ends there
    if ball > 0.0 and (head or part == "knob"):  # a knob, wider than its stick
        rows.add("balls", part, group, end, (ball, ball, ball), owner=group)
    return [(start, rows.point(group, end), group)]


def _sticks(
    tree: _Tree, rows: _Rows, heads: Mapping[int, _Head], *,
    members: Mapping[int, list[int]], cluster: Mapping[int, int], anchor: Mapping[int, int], bodies: bool,
) -> list[tuple[int, int, int]]:  # fmt: skip
    """The sticks of the stick figure as ``(anchor, anchor, joint whose side tints it)``.

    Each ball hangs from the ball of the nearest drawn group above it. A group with two or more balls below it
    joins them through a hub that is rigid in the group, the mean of its child joints; a left and right pair of
    them hangs from a bar whose middle joins the hub, so a quadruped has a spine and two shoulder bars and a pelvis
    a hip bar, not a star of sticks from the hub. A fixed base gets a cube at the root body's origin and a stick up
    to each first ball; a head hangs from its group's hub, else from its group's ball. Without ``bodies`` no group
    has a hub: its balls hang from its own.
    """
    origin = tree.origin

    def above(joint: int) -> int:
        while joint and cluster[joint] not in anchor:  # a twist that is not drawn: the stick runs through
            joint = tree.group[joint]
        return joint

    below: dict[int, list[int]] = {}
    for c in anchor:
        below.setdefault(above(tree.group[members[c][0]]), []).append(c)
    sticks, bases, hubs = [], {}, {}
    for group, kids in below.items():
        own = cluster[group] if group else -1
        first = {c: members[c][0] for c in kids if c != own}
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
        if group and len(first) >= 2 and bodies:
            where = np.mean([origin[joint] for joint in first.values()], axis=0)
            hub = hubs[group] = rows.point(group, where)
            sticks.append((anchor[own], hub, group))
            bars: dict[frozenset[int], int] = {}  # a mirror pair of child balls hangs from a bar: hips, shoulders
            for c, joint in first.items():
                twin = next((d for d in first if d != c and _mirrored(tree, joint, first[d])), None)
                if twin is None:
                    sticks.append((hub, anchor[c], joint))
                    continue
                pair = frozenset((c, twin))
                if pair not in bars:
                    bars[pair] = rows.point(group, (rows.where[anchor[c]] + rows.where[anchor[twin]]) / 2)
                    sticks.append((hub, bars[pair], group))
                sticks.append((bars[pair], anchor[c], joint))
        elif group:
            sticks += [(anchor[own], anchor[c], joint) for c, joint in first.items()]
    for group, (body, center) in heads.items():  # a head hangs from its trunk's hub
        start = hubs[group] if group in hubs else anchor[cluster[above(group)]]
        sticks.append((start, rows.point(body, center), group))
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


def unit_box(chamfer: float = 0.12) -> tuple[np.ndarray, np.ndarray]:
    """Cube of side 1 with chamfered edges and corners, to draw flat-shaded: ``(vertices (24, 3), faces (44, 3))``.

    Each corner is inset by ``chamfer`` along two of its three axes, which gives three vertices per corner; their
    convex hull has six square faces, twelve edge strips and eight corner triangles.
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
    return vertices.astype(np.float32), np.array(faces, dtype=np.uint32)
