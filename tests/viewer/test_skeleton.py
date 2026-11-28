"""The skeleton layout: glyphs of one rigid group stay clear, rods and sticks are thinner than their glyphs."""

from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import replace
from functools import partial
from pathlib import Path

import numpy as np
import pytest
import torch

import better_robot as br
from better_robot import collision
from better_robot.model import (
    Box,
    Geometry,
    JointFixed,
    JointPlanar,
    JointPrismatic,
    JointRevolute,
    JointRevoluteUnbounded,
    JointSpherical,
    JointTranslation,
    Model,
)
from better_robot.viewer import LIGHT, RobotVisual
from better_robot.viewer.skeleton import MEDIUM, MINIMAL, Skeleton, SkeletonRecipe, build_skeleton

from better_robot.exceptions import ArgumentError, UnknownNameError
from tests.support.robots import g1_urdf, panda_urdf

_GLYPH_PARTS = ("drum", "sleeve", "ball", "plate", "block", "marker", "base")


def _g1() -> tuple[Model, collision.RobotSpheres]:
    model = br.load(g1_urdf(), root_joint=br.JointFreeFlyer())
    return model, collision.load_spheres(collision.SPHERE_SETS / "g1.json", model)[0]


def _panda() -> tuple[Model, collision.RobotSpheres]:
    model = br.load(panda_urdf())
    return model, collision.load_spheres(collision.SPHERE_SETS / "panda.json", model)[0]


def _mixed() -> tuple[Model, None]:
    """A free-flyer base carrying a prismatic mast, a spherical shoulder, an elbow and a planar puck; no geometry."""
    builder = br.io.ModelBuilder("mixed")

    def at(x: float, y: float, z: float) -> torch.Tensor:
        return torch.tensor([x, y, z, 0.0, 0.0, 0.0, 1.0])

    for name, mass in (("base", 5.0), ("carriage", 1.0), ("upper", 1.0), ("fore", 0.8), ("puck", 0.2)):
        builder.add_body(name, mass=mass, com=torch.tensor([0.1, 0.0, 0.0]), inertia=torch.eye(3) * mass * 0.004)
    builder.add_free_flyer_root("root_joint", child="base")
    builder.add_prismatic(
        "mast", parent="base", child="carriage", axis=(0.0, 0.0, 1.0), origin=at(-0.1, 0.0, 0.2), lower=0.0, upper=0.3,
        effort_limit=200.0,
    )  # fmt: skip
    builder.add_spherical("shoulder", parent="carriage", child="upper", origin=at(0.0, 0.0, 0.2))
    builder.add_revolute(
        "elbow", parent="upper", child="fore", axis=(0.0, 1.0, 0.0), origin=at(0.3, 0.0, 0.0), lower=-2.0, upper=0.5,
        effort_limit=40.0,
    )  # fmt: skip
    builder.add_planar("table", parent="base", child="puck", origin=at(0.05, 0.25, 0.1))
    return br.io.build_model(builder.finalize()), None


def _at(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> torch.Tensor:
    return torch.tensor([x, y, z, 0.0, 0.0, 0.0, 1.0])


_HINGE = {"axis": (0.0, 1.0, 0.0), "lower": -1.5, "upper": 1.5}


def _arm(builder: br.io.ModelBuilder, *, thin: bool = False, boxes: bool = False) -> None:
    """A fixed base and four revolute joints 25 cm apart along z, no axis along the line; ``thin``: the leaf's one
    shape is a 2 mm placeholder box; ``boxes``: the base and each link carry a box from their joint to the next."""
    builder.add_body("base", mass=1.0)
    builder.add_fixed("mount", parent="world", child="base")
    for k, axis in enumerate(((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))):
        builder.add_body(f"l{k}", mass=1.0)
        builder.add_revolute(f"j{k}", parent=f"l{k - 1}" if k else "base", child=f"l{k}", axis=axis,
                             origin=_at(z=0.25), lower=-2.0, upper=2.0)  # fmt: skip
    for body in ("base", "l0", "l1", "l2", "l3") if boxes else ():
        builder.add_shape(body, Box(size=(0.06, 0.06, 0.25), placement=(0, 0, 0.125, 0, 0, 0, 1.0)))
    if thin:
        builder.add_shape("l3", Box(size=(0.002, 0.002, 0.002), placement=(0, 0, 0.1, 0, 0, 0, 1.0)))


def _on_world(builder: br.io.ModelBuilder, *, slider: bool) -> None:
    """A hinge or a slider that hangs from the world itself, not from a fixed base, and a hinge on it."""
    builder.add_body("cart", mass=1.0, com=torch.tensor([0.2, 0.0, 0.0]))
    builder.add_body("pole", mass=0.2)
    add, axis = (builder.add_prismatic, (1.0, 0.0, 0.0)) if slider else (builder.add_revolute, (0.0, 0.0, 1.0))
    add("rider", parent="world", child="cart", axis=axis, lower=-1.0, upper=1.0)
    builder.add_revolute("tip", parent="cart", child="pole", origin=_at(z=0.1), **_HINGE)


def _gantry(builder: br.io.ModelBuilder) -> None:
    """Three sliders from the world at one point: the joint origins span nothing."""
    for body, parent, axis in (
        ("x", "world", (1.0, 0.0, 0.0)),
        ("y", "x", (0.0, 1.0, 0.0)),
        ("z", "y", (0.0, 0.0, 1.0)),
    ):
        builder.add_body(body, mass=1.0, inertia=0.01 * torch.eye(3))
        builder.add_prismatic(f"p{body}", parent=parent, child=body, axis=axis, lower=-1.0, upper=1.0)


def _planar(builder: br.io.ModelBuilder) -> None:
    """A planar root, then a spherical, a helical, a revolute and a translation joint."""
    for body in ("base", "a", "b", "c", "d"):
        builder.add_body(body, mass=1.0)
    builder.add_planar("slide", parent="world", child="base")
    builder.add_spherical("ball", parent="base", child="a", origin=_at(z=0.3))
    builder.add_helical("screw", parent="a", child="b", axis=(0.0, 0.0, 1.0), pitch=0.01, origin=_at(z=0.3),
                        lower=-3.0, upper=3.0)  # fmt: skip
    builder.add_revolute("tip", parent="b", child="c", origin=_at(z=0.3), **_HINGE)
    builder.add_joint("shift", kind=JointTranslation(), parent="c", child="d", origin=_at(z=0.3))


def _hand(builder: br.io.ModelBuilder) -> None:
    """A floating palm, five fingers of three joints, and a gripper whose second jaw mimics the first."""
    builder.add_body("palm", mass=1.0)
    builder.add_free_flyer_root(child="palm")
    for finger, k in itertools.product(range(5), range(3)):
        builder.add_body(f"f{finger}{k}", mass=0.05)
        origin = _at(0.08, 0.02 * (finger - 2)) if k == 0 else _at(0.03)
        builder.add_revolute(f"f{finger}{k}_joint", parent=f"f{finger}{k - 1}" if k else "palm", child=f"f{finger}{k}",
                             origin=origin, axis=(0.0, 1.0, 0.0), lower=0.0, upper=1.5)  # fmt: skip
    for jaw in ("jaw_a", "jaw_b"):
        builder.add_body(jaw, mass=0.02)
        builder.add_shape(jaw, Box(size=(0.01, 0.01, 0.04), placement=(0, 0, 0.02, 0, 0, 0, 1.0)))
    builder.add_prismatic("grip", parent="palm", child="jaw_a", axis=(0.0, 1.0, 0.0), origin=_at(-0.05, 0.01),
                          lower=0.0, upper=0.04)  # fmt: skip
    builder.add_joint("grip_mimic", kind=JointPrismatic(axis=(0.0, 1.0, 0.0)), parent="palm", child="jaw_b",
                      origin=_at(-0.05, -0.01), mimic_source="grip", mimic_multiplier=-1.0)  # fmt: skip


def _fixed(builder: br.io.ModelBuilder) -> None:
    """Nothing moves."""
    builder.add_body("a", mass=1.0)
    builder.add_body("b", mass=1.0)
    builder.add_fixed("mount", parent="world", child="a")
    builder.add_fixed("weld", parent="a", child="b", origin=_at(z=0.5))


def _box(builder: br.io.ModelBuilder) -> None:
    """A lone floating body."""
    builder.add_body("box", mass=1.0)
    builder.add_shape("box", Box(size=(0.1, 0.2, 0.3), placement=(0, 0, 0, 0, 0, 0, 1.0)))
    builder.add_free_flyer_root(child="box")


def _servo(builder: br.io.ModelBuilder) -> None:
    """A 2 cm servo and its horn."""
    builder.add_body("case", mass=0.05)
    builder.add_body("horn", mass=0.01)
    builder.add_shape("horn", Box(size=(0.04, 0.008, 0.004), placement=(0.015, 0, 0, 0, 0, 0, 1.0)))
    builder.add_fixed("mount", parent="world", child="case")
    builder.add_revolute("spin", parent="case", child="horn", origin=_at(z=0.02), **_HINGE)


def _forearm(builder: br.io.ModelBuilder) -> None:
    """A hand on a fixed forearm as large as a Shadow hand's (15 x 15 x 23 cm), its wrist at the forearm's end."""
    for body in ("forearm", "palm", "finger"):
        builder.add_body(body, mass=1.0)
    builder.add_fixed("mount", parent="world", child="forearm")
    builder.add_shape("forearm", Box(size=(0.15, 0.15, 0.23), placement=(0, 0, 0.115, 0, 0, 0, 1.0)))
    builder.add_revolute("wrist", parent="forearm", child="palm", origin=_at(z=0.23), **_HINGE)
    builder.add_shape("palm", Box(size=(0.03, 0.08, 0.1), placement=(0, 0, 0.05, 0, 0, 0, 1.0)))
    builder.add_revolute("knuckle", parent="palm", child="finger", origin=_at(z=0.1), **_HINGE)
    builder.add_shape("finger", Box(size=(0.02, 0.02, 0.05), placement=(0, 0, 0.025, 0, 0, 0, 1.0)))


def _bust(builder: br.io.ModelBuilder) -> None:
    """A floating chest with two arms and a head on a pan and tilt neck that rises above them."""
    for name in ("chest", "l_arm", "r_arm", "neck", "head"):
        builder.add_body(name, mass=1.0)
    builder.add_free_flyer_root(child="chest")
    builder.add_shape("chest", Box(size=(0.2, 0.4, 0.5), placement=(0, 0, 0, 0, 0, 0, 1.0)))
    for side, y in (("l", 0.25), ("r", -0.25)):
        builder.add_revolute(f"{side}_shoulder", parent="chest", child=f"{side}_arm", origin=_at(y=y, z=0.2), **_HINGE)
        builder.add_shape(f"{side}_arm", Box(size=(0.06, 0.06, 0.3), placement=(0, 0, -0.15, 0, 0, 0, 1.0)))
    builder.add_revolute("pan", parent="chest", child="neck", origin=_at(z=0.3), **{**_HINGE, "axis": (0.0, 0.0, 1.0)})
    builder.add_revolute("tilt", parent="neck", child="head", origin=_at(), **_HINGE)
    builder.add_shape("head", Box(size=(0.16, 0.16, 0.18), placement=(0, 0, 0.12, 0, 0, 0, 1.0)))


#: Small models at the edges of the layout.
_SYNTHETIC: dict[str, Callable[[br.io.ModelBuilder], None]] = {
    "arm": _arm, "thin_tip": partial(_arm, thin=True), "boxed_arm": partial(_arm, boxes=True),
    "world_hinge": partial(_on_world, slider=False),
    "world_slider": partial(_on_world, slider=True), "gantry": _gantry, "planar": _planar, "hand": _hand,
    "fixed": _fixed, "box": _box, "servo": _servo, "bust": _bust, "forearm": _forearm,
}  # fmt: skip


def _synthetic(name: str) -> tuple[Model, Geometry]:
    """The model ``name`` of ``_SYNTHETIC`` and the geometry its builder attached."""
    builder = br.io.ModelBuilder(name)
    _SYNTHETIC[name](builder)
    ir = builder.finalize()
    model = br.io.build_model(ir)
    return model, br.io.build_geometry(ir, model)


_ROBOTS = {"g1": _g1, "panda": _panda, "mixed": _mixed}


def _poses(model: Model, q: torch.Tensor | None = None) -> np.ndarray:
    return br.forward_kinematics(model, model.q_neutral if q is None else q).joint_poses.numpy()


def _rotations(quaternions: np.ndarray) -> np.ndarray:
    """Rotation matrices ``(N, 3, 3)`` of quaternions ``(N, 4)``, scalar last."""
    x, y, z, w = quaternions.T
    rows = [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]  # fmt: skip
    return np.stack(rows, axis=1).reshape(-1, 3, 3)


def _drums(skeleton: Skeleton, poses: np.ndarray) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Per joint, its drum at ``poses``: centre ``(3,)``, rotation ``(3, 3)``, half extents ``(radius, radius, half)``."""
    world, scale = skeleton.place(poses)["cylinders"]
    rotations = _rotations(world[:, 3:])
    blocks = skeleton.blocks["cylinders"]
    return {
        int(blocks.owner[i]): (world[i, :3], rotations[i], scale[i] * np.array([1.0, 1.0, 0.5]))
        for i in range(len(blocks))
        if blocks.part[i] == "drum"
    }


def _anchors(model: Model) -> list[int]:
    """Per joint, the moving joint its body is welded to: itself, or through fixed joints; 0 is the world."""
    anchor = [0] * model.njoints
    for j in range(1, model.njoints):
        anchor[j] = anchor[model.parent_joint_ids[j]] if isinstance(model.joints[j], JointFixed) else j
    return anchor


def _same_group_pairs(model: Model, joints: set[int]) -> list[tuple[int, int]]:
    """Pairs of moving joints whose glyphs share a rigid group: a joint with its child joints, and siblings."""
    anchor = _anchors(model)
    group = {j: anchor[model.parent_joint_ids[j]] for j in joints}
    return [(a, b) for a, b in itertools.combinations(sorted(joints), 2) if group[b] == a or group[a] == group[b]]


def _overlap(first: tuple, second: tuple, *, round_: bool) -> bool:
    """Whether a grid of points filling ``first`` has a point inside ``second`` (boxes, or the cylinders in them)."""
    center, rotation, half = first
    axis = np.linspace(-1.0, 1.0, 13)
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    if round_:
        grid = grid[np.hypot(grid[:, 0], grid[:, 1]) <= 1.0]
    points = center + (grid * half) @ rotation.T
    other_center, other_rotation, other_half = second
    local = (points - other_center) @ other_rotation / other_half
    radial = np.hypot(local[:, 0], local[:, 1]) if round_ else np.abs(local[:, :2]).max(axis=1)
    return bool(((radial < 1.0 - 1e-9) & (np.abs(local[:, 2]) < 1.0 - 1e-9)).any())


def _solids(skeleton: Skeleton, poses: np.ndarray) -> dict[int, list[tuple[np.ndarray, np.ndarray, np.ndarray, bool]]]:
    """Per joint, the solids of its glyph at ``poses``: a drum and its faces (round), a sleeve and its rail, a ball
    (its box), a planar joint's plate, a translation's block, each as centre ``(3,)``, rotation ``(3, 3)``, half
    extents ``(3,)`` and whether it is round."""
    solids: dict[int, list] = {}
    for mesh, parts in (
        ("cylinders", ("drum", "face", "rail")),
        ("boxes", ("sleeve", "plate", "block")),
        ("balls", ("ball",)),
    ):
        world, scale = skeleton.place(poses)[mesh]
        blocks, rotations = skeleton.blocks[mesh], _rotations(world[:, 3:])
        for i in (i for i in range(len(blocks)) if blocks.part[i] in parts):
            half = scale[i] * {"cylinders": [1.0, 1.0, 0.5], "boxes": [0.5, 0.5, 0.5], "balls": [1.0, 1.0, 1.0]}[mesh]
            solids.setdefault(int(blocks.owner[i]), []).append((world[i, :3], rotations[i], half, mesh == "cylinders"))
    return solids


def _touching(first: list, second: list) -> bool:
    """Whether a solid of ``first`` and one of ``second`` overlap: any grid point of one inside the other."""
    for (center, rotation, half, round_), (other, turn, reach, also) in itertools.product(first, second):
        if np.linalg.norm(center - other) < np.linalg.norm(half) + np.linalg.norm(reach) and (
            _overlap((center, rotation, half), (other, turn, reach), round_=round_ and also)
            or _overlap((other, turn, reach), (center, rotation, half), round_=round_ and also)
        ):
            return True
    return False


#: Per robot, the joints whose glyphs meet by construction, each with its reason. The mixed robot's shoulder ball sits
#: on its mast's line 20 cm above the carriage, within the mast's 30 cm travel: the rail spans that travel, so it runs
#: through the ball, which has no axis to slide along, and no shrink takes it off the line.
_MEETING = {"mixed": {("mast", "shoulder")}}


@pytest.mark.parametrize("robot", ["g1", "panda", "mixed", "planar"])
def test_no_joint_glyph_goes_inside_another(robot: str) -> None:
    """No two joint glyphs overlap at neutral, neighbours in the tree or not (a hip's three drums, a waist beside the
    hips, the fingers, a planar plate or a translation block beside a drum): each slides along its own axis or
    shrinks. The glyphs of one rigid group (a joint with its child joints, and siblings) stay clear in every pose,
    because a drum is rigid in its parent group and symmetric about its axis."""
    if robot in _SYNTHETIC:
        model, geometry = _synthetic(robot)
        skeleton = build_skeleton(model, MEDIUM, geometry=geometry)
    else:
        model, spheres = _ROBOTS[robot]()
        skeleton = build_skeleton(model, MEDIUM, spheres=spheres)
    names, meeting = model.joint_names, _MEETING.get(robot, set())
    solids = _solids(skeleton, _poses(model))
    pairs = itertools.combinations(sorted(solids), 2)
    assert {(names[a], names[b]) for a, b in pairs if _touching(solids[a], solids[b])} == meeting
    velocity = 0.8 * torch.randn(model.nv, generator=torch.Generator().manual_seed(0), dtype=model.q_neutral.dtype)
    solids = _solids(skeleton, _poses(model, br.kinematics.integrate(model, model.q_neutral, velocity)))
    for a, b in _same_group_pairs(model, set(solids)):
        assert (names[a], names[b]) in meeting or not _touching(solids[a], solids[b]), (names[a], names[b])


def test_g1_slides_are_mirror_images() -> None:
    model, spheres = _g1()
    skeleton = build_skeleton(model, MEDIUM, spheres=spheres)
    assert skeleton.slides  # the hips, shoulders and wrists are close enough to need slides
    for name, slide in skeleton.slides.items():
        if name.startswith("left_"):
            assert abs(skeleton.slides[name.replace("left_", "right_")]) == pytest.approx(abs(slide), abs=1e-9)
    for name, shrink in skeleton.shrinks.items():
        twin = name.replace("left_", "right_") if name.startswith("left_") else name.replace("right_", "left_")
        assert skeleton.shrinks.get(twin) == pytest.approx(shrink) or name.startswith("waist")
    poses = _poses(model)
    drums = _drums(skeleton, poses)
    mirror = np.array([1.0, -1.0, 1.0])  # the neutral root frame is the world: left is positive y
    lefts = [j for j in drums if model.joint_names[j].startswith("left_")]
    assert lefts
    for j in lefts:  # each drum's offset from its joint (the description's own origins differ by 10 micrometres)
        other = model.joint_names.index(model.joint_names[j].replace("left_", "right_"))
        offset, mirrored = drums[j][0] - poses[j, :3], drums[other][0] - poses[other, :3]
        np.testing.assert_allclose(offset * mirror, mirrored, atol=1e-6)
        np.testing.assert_allclose(drums[j][2], drums[other][2], atol=1e-9)


@pytest.mark.parametrize("robot", ["g1", "panda"])
def test_the_drums_of_one_robot_keep_a_narrow_range_of_sizes(robot: str) -> None:
    """No joint is too small beside the others: before the layout shrinks them, a robot's smallest drum is at least
    0.7 of its largest. Measured: 0.75 (G1) and 0.79 (Panda), where the floor of 2.5 rod radii lifts the G1's wrists
    and ankles (0.43 under a floor of half the reference radius)."""
    model, spheres = _ROBOTS[robot]()
    skeleton = build_skeleton(model, MEDIUM, spheres=spheres)
    blocks = skeleton.blocks["cylinders"]
    radii = [
        blocks.scale[i, 0] / skeleton.shrinks.get(model.joint_names[blocks.owner[i]], 1.0)
        for i in range(len(blocks))
        if blocks.part[i] == "drum"
    ]
    assert min(radii) >= 0.7 * max(radii)
    floor = 2.5 * MEDIUM.bones  # reference radii: no wrist or ankle reads as a bead
    assert min(radii) >= floor * skeleton.radius * (1 - 1e-6)


@pytest.mark.parametrize("robot", ["g1", "panda", "mixed"])
def test_medium_draws_a_glyph_per_joint_kind_on_rods(robot: str) -> None:
    """The medium skeleton is the stick figure with a glyph per joint kind (the mixed robot has a drum, a sleeve on
    its rail, a ball, a free-flyer cube and a planar plate) on rods that follow their ends; no stick, and a chain ends
    in a round cap of its rod."""
    model, spheres = _ROBOTS[robot]()
    medium = build_skeleton(model, MEDIUM, spheres=spheres)
    parts = {mesh: set(medium.blocks[mesh].part) for mesh in ("cylinders", "balls", "boxes")}
    assert "stick" not in set.union(*parts.values())
    rods = np.array(medium.blocks["cylinders"].part) == "rod"
    assert rods.any() and (medium.blocks["cylinders"].anchors[rods, 0] >= 0).all()
    if robot == "mixed":
        assert set.union(*parts.values()) == {"drum", "face", "sleeve", "rail", "ball", "base", "plate", "rod", "cap"}


@pytest.mark.parametrize("name", [*_SYNTHETIC, "mixed", "g1", "panda"])
def test_a_hinge_is_a_drum_and_a_slider_a_sleeve_never_a_ball(name: str) -> None:
    """With a glyph per joint kind, every revolute joint with a motor of its own is a drum, every prismatic joint
    a sleeve, alone or beside others, a planar joint a thin plate and a translation a block: a ball is the glyph of a
    spherical joint (the owner on a G1 drawn with balls at its hips: "not acceptable"; R2 m3: a planar base too). The
    only other balls are at most one head per trunk and the round cap of a chain's last rod, no wider than the rod;
    both in the rods' pale shade. A chain's end, a hidden joint and a quadruped's snout draw no ball."""
    if name in _SYNTHETIC:
        model, geometry = _synthetic(name)
        skeleton = build_skeleton(model, MEDIUM, geometry=geometry)
    else:
        model, spheres = _ROBOTS[name]()
        skeleton = build_skeleton(model, MEDIUM, spheres=spheres)
    hinges = {j for j, joint in enumerate(model.joints) if isinstance(joint, (JointRevolute, JointRevoluteUnbounded))}
    hinges -= {j for j in hinges if model.mimic_source_ids[j] != j}  # a linkage that follows its source: no motor
    sliders = {j for j, joint in enumerate(model.joints) if isinstance(joint, JointPrismatic)}
    drawn = {(part, owner) for mesh in ("cylinders", "boxes") for part, owner in _parts(skeleton, mesh)}
    assert {j for j in hinges if ("drum", j) not in drawn} == set(), name
    assert {j for j in sliders if ("sleeve", j) not in drawn} == set(), name
    for kind, part in ((JointPlanar, "plate"), (JointTranslation, "block")):
        joints = {j for j, joint in enumerate(model.joints) if isinstance(joint, kind)}
        assert {j for j in joints if (part, j) not in drawn} == set(), (name, part)
    spherical = {j for j, joint in enumerate(model.joints) if isinstance(joint, JointSpherical)}
    kids = np.bincount(np.asarray(model.parent_joint_ids[1:]), minlength=model.njoints)
    trunks = 1 + int((kids[1:] >= 2).sum())  # the root, and every joint with two or more child joints
    hue = (0.2, 0.6, 0.4)
    colors = skeleton.colors(hue, LIGHT)["balls"]
    balls = zip(_parts(skeleton, "balls"), skeleton.blocks["balls"].scale, colors, strict=True)
    for (part, owner), scale, color in balls:
        assert part in ("ball", "head", "cap"), (name, part)
        assert part != "ball" or owner in spherical, (name, model.joint_names[owner])
        assert part != "cap" or scale.max() <= MEDIUM.bones * skeleton.radius + 1e-12, name
        if part in ("head", "cap"):
            np.testing.assert_allclose(color, LIGHT.shade(hue, "bone"), atol=1e-6)
    assert skeleton.blocks["balls"].part.count("head") <= trunks, name


def test_a_neck_ends_in_a_head_that_stands_above_its_trunk() -> None:
    """A chain of at most two joints that leaves a trunk above every other joint and ends in round geometry is a
    neck: it ends in one pale ball of the reference radius in the middle of that geometry (a Talos, an Atlas)."""
    model, geometry = _synthetic("bust")
    skeleton = build_skeleton(model, MEDIUM, geometry=geometry)
    tilt = model.joint_names.index("tilt")
    balls, _ = skeleton.place(_poses(model))["balls"]
    heads = [i for i, (_, owner) in enumerate(_parts(skeleton, "balls")) if owner == tilt]
    assert len(heads) == 1 and skeleton.blocks["balls"].part[heads[0]] == "head"
    np.testing.assert_allclose(balls[heads[0], :3], [0.0, 0.0, 0.42], atol=1e-6)
    np.testing.assert_allclose(skeleton.blocks["balls"].scale[heads[0]], skeleton.radius)


def test_g1_medium_wrist_keeps_rod_between_its_drums() -> None:
    """On rods a drum is at most 0.35 of its shortest bone, never under 2.5 rod radii (``MEDIUM.bones`` reference
    radii each): the G1's forearm reads rod, wrist roll, rod, wrist pitch, rod, wrist yaw, not a cluster of motors."""
    model, spheres = _g1()
    skeleton = build_skeleton(model, MEDIUM, spheres=spheres)
    poses = _poses(model)
    origin, drums = poses[:, :3], _drums(skeleton, poses)
    chain = [
        model.joint_names.index(f"left_{name}_joint") for name in ("elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")
    ]

    def reach(joint: int, direction: np.ndarray) -> tuple[float, float]:  # a placed drum's span along the bone
        center, rotation, half = drums[joint]
        along = abs(float(rotation[:, 2] @ direction))
        extent = along * half[2] + np.sqrt(max(1.0 - along**2, 0.0)) * half[0]
        return float(center @ direction) - extent, float(center @ direction) + extent

    for before, joint in itertools.pairwise(chain):
        bone = float(np.linalg.norm(origin[joint] - origin[before]))
        radius = max(drums[before][2][0], drums[joint][2][0])
        assert radius <= max(0.35 * bone, 2.5 * MEDIUM.bones * skeleton.radius) + 1e-9
        direction = (origin[joint] - origin[before]) / bone
        assert reach(joint, direction)[0] > reach(before, direction)[1]  # rod shows between them


def _parts(skeleton: Skeleton, mesh: str) -> list[tuple[str, int]]:
    blocks = skeleton.blocks[mesh]
    return [(part, int(owner)) for part, owner in zip(blocks.part, blocks.owner, strict=True)]


def test_panda_rails_span_the_finger_travel_in_the_hand() -> None:
    model, spheres = _panda()
    skeleton = build_skeleton(model, MEDIUM, spheres=spheres)
    cylinders = skeleton.blocks["cylinders"]
    rails = [i for i, part in enumerate(cylinders.part) if part == "rail"]
    assert len(rails) == 2 and {int(cylinders.body[i]) for i in rails} == {model.joint_names.index("panda_joint7")}
    np.testing.assert_allclose(cylinders.scale[rails, 2], 0.04, atol=1e-6)  # the 4 cm each finger travels


@pytest.mark.parametrize("robot", ["g1", "panda", "mixed"])
@pytest.mark.parametrize("recipe", [MEDIUM, MINIMAL], ids=["medium", "minimal"])
def test_bone_is_never_thicker_than_the_glyphs_it_joins(robot: str, recipe: SkeletonRecipe) -> None:
    """A rod's or a stick's end inside a glyph is no wider than that glyph."""
    model, spheres = _ROBOTS[robot]()
    skeleton = build_skeleton(model, recipe, spheres=spheres)
    placed = skeleton.place(_poses(model))
    glyphs = []  # (centre, radius): half the side for a box
    for mesh, per_scale in (("cylinders", 1.0), ("balls", 1.0), ("boxes", 0.5)):
        blocks, (world, scale) = skeleton.blocks[mesh], placed[mesh]
        glyphs += [
            (world[i, :3], per_scale * scale[i, 0]) for i in range(len(blocks)) if blocks.part[i] in _GLYPH_PARTS
        ]
    centers, radii = np.stack([center for center, _ in glyphs]), np.array([radius for _, radius in glyphs])
    blocks, (world, scale) = skeleton.blocks["cylinders"], placed["cylinders"]
    axes = _rotations(world[:, 3:])[:, :, 2]
    bones = [  # (centre, axis, length, radius) of each rod or stick
        (world[i, :3], axes[i], scale[i, 2], scale[i, 0])
        for i in range(len(blocks))
        if blocks.part[i] in ("stick", "rod")
    ]
    assert bones
    joined = np.zeros(len(glyphs), dtype=bool)
    for center, axis, length, half in bones:
        for sign in (-1.0, 1.0):
            distance = np.linalg.norm(centers - (center + 0.5 * sign * length * axis), axis=1)
            inside = distance < 0.5 * radii  # the bone's cap is in these glyphs
            assert (half <= radii[inside] + 1e-12).all(), (center, half, radii[inside])
            joined |= inside | ((distance <= 1.3 * radii) & (half > 0.8 * radii))
    assert joined.sum() >= len(glyphs) - 1  # every glyph but at most one (a free-flyer's cube) holds a bone end


def test_g1_minimal_clusters_close_joints_alike_on_both_sides_and_without_geometry() -> None:
    model, spheres = _g1()

    def clusters(skeleton: Skeleton) -> list[str]:  # a cluster's ball follows its joints: heads and knobs are rigid
        return [
            model.joint_names[o]
            for o, (a, _) in zip(skeleton.blocks["balls"].owner, skeleton.blocks["balls"].anchors)
            if a >= 0
        ]

    owners = clusters(build_skeleton(model, MINIMAL, spheres=spheres))
    assert "left_hip_pitch_joint" in owners and "left_hip_roll_joint" not in owners  # one ball for the pair
    assert sorted(o.replace("left_", "right_") for o in owners if o.startswith("left_")) == sorted(
        o for o in owners if o.startswith("right_")
    )
    assert sorted(clusters(build_skeleton(model, MINIMAL))) == sorted(
        owners
    )  # the clusters do not depend on the geometry


@pytest.mark.parametrize("robot", ["g1", "panda"])
def test_a_drum_stays_on_its_axis_within_its_length_of_its_joint(robot: str) -> None:
    """Every revolute joint draws one drum on its own axis (close joints too: their drums stand apart), slid along
    that axis by at most 1.75 of its radii before the layout shrank it, after a twist joint's collar offset of one
    half length: it never leaves its joint."""
    model, spheres = _ROBOTS[robot]()
    poses = _poses(model)
    skeleton = build_skeleton(model, MEDIUM, spheres=spheres)
    drums = _drums(skeleton, poses)
    revolute = {j for j in range(1, model.njoints) if type(model.joints[j]).__name__ == "JointRevolute"}
    assert set(drums) == revolute
    for joint, (center, rotation, half) in drums.items():
        axis = _rotations(poses[joint : joint + 1, 3:])[0] @ np.asarray(model.joints[joint].axis)
        assert abs(float(axis @ rotation[:, 2])) == pytest.approx(1.0, abs=1e-5), model.joint_names[joint]
        offset = center - poses[joint, :3]
        along = float(offset @ rotation[:, 2])
        np.testing.assert_allclose(offset, along * rotation[:, 2], atol=1e-6)
        radius = half[0] / skeleton.shrinks.get(model.joint_names[joint], 1.0)
        assert abs(along) <= half[2] + 1.75 * radius + 1e-9, (model.joint_names[joint], along, radius)


def test_go2_minimal_legs_run_on_to_the_feet() -> None:
    """The Go2's knees are its leaf joints: the minimal skeleton runs a stick from each knee to its foot, and the
    ball that ends it reaches the lowest point of that leg's meshes within 1 cm (the foot's ball is 2.2 cm wide).
    The Go2 comes from the ``robot_descriptions`` cache, as in the examples."""
    import trimesh  # noqa: PLC0415 - optional dependency of the viewer
    from robot_descriptions import go2_description  # noqa: PLC0415 - the examples' robots, not a fixture

    from better_robot.viewer.robot import _mesh_file  # noqa: PLC0415 - the viewer's own mesh lookup

    urdf = go2_description.URDF_PATH
    model = br.load(urdf, root_joint=br.JointFreeFlyer())
    geometry = br.io.load_geometry(urdf, model)
    skeleton = build_skeleton(model, MINIMAL, geometry=geometry)
    poses = _poses(model)
    placed = skeleton.place(poses)
    (balls, ball_scale), (sticks, stick_scale) = placed["balls"], placed["cylinders"]
    axes = _rotations(sticks[:, 3:])[:, :, 2]
    stick_ends = np.concatenate([sticks[:, :3] + sign * 0.5 * stick_scale[:, 2:] * axes for sign in (-1.0, 1.0)])
    anchor = _anchors(model)
    for leg in ("FL", "FR", "RL", "RR"):
        knee = model.joint_names.index(f"{leg}_calf_joint")
        tips = [
            i
            for i, owner in enumerate(skeleton.blocks["balls"].owner)
            if owner == knee and skeleton.blocks["balls"].anchors[i, 0] < 0
        ]
        assert len(tips) == 1, leg
        tip = tips[0]
        assert np.linalg.norm(stick_ends - balls[tip, :3], axis=1).min() < 1e-5  # a stick ends in the ball
        lowest = np.inf
        for attached in geometry.shapes:
            if attached.collision or anchor[attached.body] != knee:
                continue
            vertices = trimesh.load(_mesh_file(attached.body, attached.shape.uri, Path(urdf)), force="mesh").vertices
            world = torch.from_numpy(poses[attached.body]).double()
            world = br.lie.se3.compose(world, torch.tensor(attached.shape.placement, dtype=torch.float64))
            points = br.lie.se3.act(world[None], torch.from_numpy(vertices * np.asarray(attached.shape.scale)))
            lowest = min(lowest, float(points[:, 2].min()))
        assert abs(balls[tip, 2] - ball_scale[tip, 2] - lowest) < 0.01, (
            leg,
            balls[tip, 2] - ball_scale[tip, 2],
            lowest,
        )


def _described(name: str, *, spheres: bool = False) -> tuple[Model, dict]:
    """A robot of the examples from the ``robot_descriptions`` cache with what ``RobotVisual`` gives its skeleton:
    the geometry, the collision mesh loader and, when asked, the robot's sphere set."""
    import importlib  # noqa: PLC0415 - the examples' robots, not a fixture

    urdf = importlib.import_module(f"robot_descriptions.{name}_description").URDF_PATH
    model = br.load(urdf, root_joint=br.JointFreeFlyer())
    geometry = br.io.load_geometry(urdf, model)
    visual = RobotVisual(model, model.q_neutral[None], geometry=geometry, root=urdf)
    sources = {"geometry": geometry, "mesh_vertices": lambda body, mesh: visual._mesh(body, mesh).vertices}
    if spheres:
        sources["spheres"] = collision.load_spheres(collision.SPHERE_SETS / f"{name}.json", model)[0]
    return model, sources


def test_g1_layout_does_not_change_with_a_sphere_set() -> None:
    """A sphere set given to show as the collision overlay leaves the layout alone: the G1's glyphs are sized from
    its collision meshes either way."""
    model, sources = _described("g1", spheres=True)
    plain = {key: value for key, value in sources.items() if key != "spheres"}
    for recipe in (MEDIUM, MINIMAL):
        first, second = build_skeleton(model, recipe, **plain), build_skeleton(model, recipe, **sources)
        for mesh in ("cylinders", "balls", "boxes"):
            np.testing.assert_array_equal(first.blocks[mesh].local, second.blocks[mesh].local)
            np.testing.assert_array_equal(first.blocks[mesh].scale, second.blocks[mesh].scale)


@pytest.mark.parametrize("name", ["g1", "go2"])
def test_minimal_follows_the_medium_size_rule_and_draws_the_head(name: str) -> None:
    """Both styles share one size rule (a reference radius between 2 % and 3 % of the robot's size, each style at
    its recipe's ``size`` of it: the Go2's minimal balls were 3.7 % of its size), and the minimal skeleton draws the
    trunk's head that the medium one finds as one ball of its radius, rigid in the trunk, on a stick."""
    model, sources = _described(name)
    medium, minimal = build_skeleton(model, MEDIUM, **sources), build_skeleton(model, MINIMAL, **sources)
    assert 0.02 <= medium.radius / (MEDIUM.size * medium.scale) <= 0.03 + 1e-12
    assert minimal.radius == pytest.approx(MINIMAL.size / MEDIUM.size * medium.radius)
    trunk = model.joint_names.index("waist_pitch_joint" if name == "g1" else "root_joint")
    heads = [
        i
        for i, owner in enumerate(minimal.blocks["balls"].owner)
        if owner == trunk and minimal.blocks["balls"].anchors[i, 0] < 0
    ]
    assert len(heads) == 1
    assert _anchors(model)[minimal.blocks["balls"].body[heads[0]]] == trunk
    np.testing.assert_allclose(minimal.blocks["balls"].scale[heads[0]], minimal.radius)
    placed = minimal.place(_poses(model))
    (balls, _), (sticks, scale) = placed["balls"], placed["cylinders"]
    axes = _rotations(sticks[:, 3:])[:, :, 2]
    ends = np.concatenate([sticks[:, :3] + sign * 0.5 * scale[:, 2:] * axes for sign in (-1.0, 1.0)])
    assert np.linalg.norm(ends - balls[heads[0], :3], axis=1).min() < 1e-5  # a stick ends in the head
    if name == "g1":  # above every other ball
        assert balls[heads[0], 2] == balls[:, 2].max()


@pytest.mark.parametrize("recipe", [MEDIUM, MINIMAL], ids=["medium", "minimal"])
def test_place_on_a_clip_equals_place_per_step(recipe: SkeletonRecipe) -> None:
    model, spheres = _g1()
    skeleton = build_skeleton(model, recipe, spheres=spheres)
    velocity = 0.5 * torch.randn(3, model.nv, generator=torch.Generator().manual_seed(1), dtype=model.q_neutral.dtype)
    poses = _poses(model, br.kinematics.integrate(model, model.q_neutral.expand(3, -1), velocity))
    clip = skeleton.place(poses)
    for step in range(3):
        for mesh, (world, scale) in skeleton.place(poses[step]).items():
            np.testing.assert_allclose(clip[mesh][0][step], world, atol=1e-12)
            np.testing.assert_allclose(np.broadcast_to(clip[mesh][1], (3, *scale.shape))[step], scale, atol=1e-12)


def test_recipe_rejects_what_it_cannot_draw() -> None:
    with pytest.raises(ArgumentError, match="glyphs"):
        SkeletonRecipe(glyphs="cube")  # type: ignore[arg-type]
    with pytest.raises(ArgumentError, match="size"):
        SkeletonRecipe(size=0.0)
    with pytest.raises(ArgumentError, match="finite"):
        SkeletonRecipe(size=float("nan"))
    with pytest.raises(ArgumentError, match="finite"):
        SkeletonRecipe(bones=float("inf"))
    model, _ = _mixed()
    with pytest.raises(UnknownNameError, match="knee"):
        build_skeleton(model, SkeletonRecipe(joint_size={"knee": 2.0}))
    with pytest.raises(UnknownNameError, match="unknown joint 'base'"):  # a body or frame name colours nothing
        build_skeleton(model, SkeletonRecipe(color={"base": (1.0, 0.0, 0.0)}))


@pytest.mark.parametrize("recipe", [MEDIUM, MINIMAL], ids=["medium", "minimal"])
@pytest.mark.parametrize("factors", [{"j1": 0.0}, {"j1": 0.5}, {"j3": 2.0}, {"j0": 0.0, "j2": 0.0}], ids=str)
def test_joint_size_scales_its_own_glyph_and_the_bones_run_on(recipe: SkeletonRecipe, factors: dict) -> None:
    """A factor scales one glyph after the size rules, so the glyphs after it keep theirs (a 2 at the tip wins over
    the narrowing towards it), and a glyph of factor 0 goes while every rod and stick stays."""
    model, _ = _synthetic("arm")
    mesh, glyph = ("cylinders", "drum") if recipe.glyphs == "kind" else ("balls", "ball")

    def glyphs(skeleton: Skeleton) -> dict[int, float]:  # joint -> radius of its drum or of its cluster's ball
        blocks = skeleton.blocks[mesh]
        rows = zip(blocks.owner, blocks.part, blocks.scale, blocks.anchors[:, 0], strict=True)
        drawn = [
            (owner, scale) for owner, part, scale, moving in rows if part == glyph and (moving >= 0 or mesh != "balls")
        ]
        return {int(owner): float(scale[0]) for owner, scale in drawn}

    def bones(skeleton: Skeleton) -> list[tuple[str, int]]:
        blocks = skeleton.blocks["cylinders"]
        return sorted(("cylinders", int(o)) for o, p in zip(blocks.owner, blocks.part) if p in ("stick", "rod"))

    plain = build_skeleton(model, recipe)
    sized = build_skeleton(model, replace(recipe, joint_size=factors))
    expected = {j: factors.get(model.joint_names[j], 1.0) * radius for j, radius in glyphs(plain).items()}
    assert len(expected) == 4
    assert glyphs(sized) == pytest.approx({j: radius for j, radius in expected.items() if radius > 0.0})
    assert bones(sized) == bones(plain)


@pytest.mark.parametrize("name", [*_SYNTHETIC, "mixed", "g1", "panda"])
def test_every_block_line_and_bound_is_finite(name: str) -> None:
    """The skeleton styles of the presets and of recipes that hide every glyph, with limit arcs and sides, place
    finite blocks and lines and give finite bounds at two configurations: a hidden leaf whose centre of mass is on
    its joint gave a NaN end block and a NaN camera."""
    if name in _SYNTHETIC:
        (model, geometry), root = _synthetic(name), None
    elif name == "panda":
        model, root = br.load(panda_urdf()), panda_urdf()
        geometry = br.io.load_geometry(root, model)
    else:
        model, geometry, root = _ROBOTS[name]()[0], None, None
    moving = [name for j, name in enumerate(model.joint_names) if j and not isinstance(model.joints[j], JointFixed)]
    hidden = dict.fromkeys(moving, 0.0)
    velocity = torch.randn(model.nv, generator=torch.Generator().manual_seed(0), dtype=model.q_neutral.dtype)
    q = torch.stack([model.q_neutral, br.kinematics.integrate(model, model.q_neutral, velocity)])
    for medium, minimal in (
        (MEDIUM, MINIMAL),
        (replace(MEDIUM, limits=True, sides=True, joint_size=hidden), replace(MINIMAL, joint_size=hidden)),
    ):
        visual = RobotVisual(model, q, geometry=geometry, root=root, medium=medium, minimal=minimal)
        for style in ("minimal", "medium"):
            skeleton = visual._skeleton(style)
            for mesh, (world, scale) in skeleton.place(visual._poses).items():
                assert np.isfinite(world).all() and np.isfinite(scale).all(), (style, mesh)
            assert np.isfinite(skeleton.lines(visual._poses)).all(), style
        for step in range(2):
            assert np.isfinite(np.concatenate(visual.bounds(step))).all(), step


@pytest.mark.parametrize(("name", "still"), [("world_slider", "rail"), ("world_hinge", "drum")])
def test_a_joint_on_the_world_keeps_its_parent_side_in_the_world(name: str, still: str) -> None:
    """ModelBuilder's ``parent="world"`` and a top-level MJCF joint hang a joint from the world itself, not from a
    fixed base: what the layout makes rigid in the parent (a slider's rail, a drum and its faces, a limit arc) stays
    put while the joint moves, and the ticks turn with their joints."""
    model, _ = _synthetic(name)
    skeleton = build_skeleton(model, replace(MEDIUM, limits=True))
    poses = _poses(model, torch.tensor([[-0.5, 0.0], [0.5, 1.0]], dtype=model.q_neutral.dtype))
    rider, blocks = model.joint_names.index("rider"), skeleton.blocks["cylinders"]
    world = skeleton.place(poses)["cylinders"][0]
    parent = [
        i for i, (owner, part) in enumerate(zip(blocks.owner, blocks.part)) if owner == rider and part != "sleeve"
    ]
    assert [blocks.part[i] for i in parent] == [still, "face"][: len(parent)] and parent
    np.testing.assert_allclose(world[0, parent], world[1, parent], atol=1e-6)
    lines, arc = skeleton.lines(poses), skeleton.line_body == 0
    assert arc.sum() == (24 if name == "world_hinge" else 0)  # the hinge's arc in the world, the tip's in the rider
    np.testing.assert_allclose(lines[0, arc], lines[1, arc], atol=1e-6)
    assert not np.allclose(lines[0], lines[1])


def test_a_thin_placeholder_shape_does_not_shrink_a_leaf_glyph_to_nothing() -> None:
    """A leaf glyph is capped by the body it moves, but a 2 mm placeholder box (Tiago's actuated finger joints carry
    them) leaves it at 0.2 reference radii, not 1.25 times 1 mm."""
    model, geometry = _synthetic("thin_tip")
    skeleton = build_skeleton(model, MEDIUM, geometry=geometry)
    blocks, tip = skeleton.blocks["cylinders"], model.joint_names.index("j3")
    drum = next(i for i, (owner, part) in enumerate(zip(blocks.owner, blocks.part)) if owner == tip and part == "drum")
    assert blocks.scale[drum, 0] == pytest.approx(0.2 * skeleton.radius)


@pytest.mark.parametrize(("name", "size"), [("box", np.sqrt(0.1**2 + 0.2**2 + 0.3**2)), ("gantry", np.sqrt(3 * 0.06))])
def test_a_robot_whose_joints_share_one_point_takes_its_size_from_its_bodies(name: str, size: float) -> None:
    """When the joint origins span nothing (a lone floating body, sliders that share a point) the robot's size is the
    diagonal of its geometry's box and of its inertia boxes (sides of 0.245 m for an inertia of 0.01 kg m2 about each
    axis of 1 kg), not a default 1 m."""
    model, geometry = _synthetic(name)
    assert build_skeleton(model, MEDIUM, geometry=geometry).scale == pytest.approx(size)


def test_colors_shade_each_part_and_tint_the_sides() -> None:
    model, spheres = _g1()
    hue = (0.2, 0.6, 0.4)
    plain = build_skeleton(model, MEDIUM, spheres=spheres)
    colors = plain.colors(hue, LIGHT)
    parts = np.array(plain.blocks["cylinders"].part)
    np.testing.assert_allclose(
        colors["cylinders"][parts == "drum"], np.broadcast_to(LIGHT.shade(hue, "joint"), ((parts == "drum").sum(), 3))
    )
    np.testing.assert_allclose(colors["cylinders"][parts == "face"][0], LIGHT.shade(hue, "face"), atol=1e-6)
    np.testing.assert_allclose(colors["cylinders"][parts == "rod"][0], LIGHT.shade(hue, "bone"), atol=1e-6)
    red = (1.0, 0.0, 0.0)
    sided = build_skeleton(model, SkeletonRecipe(sides=True, color={"left_knee_joint": red}), spheres=spheres)
    drums = {}  # a glyph's colour per joint: its drum or its ball
    for mesh in ("cylinders", "balls"):
        blocks, rows = sided.blocks[mesh], sided.colors(hue, LIGHT)[mesh]
        drums |= {model.joint_names[o]: c for o, p, c in zip(blocks.owner, blocks.part, rows) if p in ("drum", "ball")}
    np.testing.assert_allclose(drums["left_knee_joint"], red, atol=1e-6)  # a recipe colour wins over hue and side
    lighter, darker = tuple(c + 0.35 * (1.0 - c) for c in hue), tuple(0.75 * c for c in hue)  # left, right
    np.testing.assert_allclose(drums["left_elbow_joint"], LIGHT.shade(lighter, "joint"), atol=1e-6)
    np.testing.assert_allclose(drums["right_elbow_joint"], LIGHT.shade(darker, "joint"), atol=1e-6)
    np.testing.assert_allclose(drums["waist_yaw_joint"], LIGHT.shade(hue, "joint"), atol=1e-6)


def test_a_chain_on_rods_ends_in_a_round_cap_and_no_body_draws_a_block() -> None:
    """The boxed arm draws its glyphs, its rods and one round cap at the end of its chain, on its last link."""
    model, geometry = _synthetic("boxed_arm")
    skeleton = build_skeleton(model, MEDIUM, geometry=geometry)
    drawn = {
        (part, model.body_names[body])
        for blocks in (skeleton.blocks["boxes"], skeleton.blocks["balls"])
        for part, body in zip(blocks.part, blocks.body, strict=True)
        if part not in _GLYPH_PARTS
    }
    assert drawn == {("cap", "l3")}


def test_a_limit_arc_spans_its_joint_s_limits_and_its_tick_sits_at_the_joint_s_angle() -> None:
    """Each bounded revolute joint gets 24 segments rigid in its parent, round the joint's axis about its drum's
    centre, from its lower to its upper limit, and a tick rigid in its child at the joint's angle."""
    model, _ = _synthetic("arm")
    skeleton = build_skeleton(model, replace(MEDIUM, limits=True))
    q = torch.tensor([0.3, -1.0, 1.1, -0.6], dtype=model.q_neutral.dtype)  # each within pi of both limits
    poses = _poses(model, q)
    lines, drums, anchor = skeleton.lines(poses), _drums(skeleton, poses), _anchors(model)
    joints = [j for j in range(model.njoints) if not isinstance(model.joints[j], JointFixed)]
    assert lines.shape == (25 * len(joints), 2, 3)
    for k, j in enumerate(joints):  # per joint, in order: its arc, then its tick
        arc, tick = lines[25 * k : 25 * k + 24], lines[25 * k + 24]
        assert (skeleton.line_body[25 * k : 25 * k + 24] == anchor[model.parent_joint_ids[j]]).all()
        assert skeleton.line_body[25 * k + 24] == j
        center, axis = drums[j][0], _rotations(poses[j : j + 1, 3:])[0] @ np.asarray(model.joints[j].axis)
        points = np.concatenate([arc[:, 0], arc[-1:, 1]]) - center
        np.testing.assert_allclose(arc[1:, 0], arc[:-1, 1], atol=1e-6)  # one unbroken arc
        np.testing.assert_allclose(points @ axis, 0.0, atol=1e-6)  # round the axis, in the drum's middle plane
        np.testing.assert_allclose(np.linalg.norm(points, axis=1), np.linalg.norm(points[0]), atol=1e-6)
        assert np.linalg.norm(points[0]) > drums[j][2][0]  # outside the drum

        def turn(start: np.ndarray, end: np.ndarray) -> np.ndarray:  # signed angles about the axis
            return np.arctan2(np.cross(start, end) @ axis, np.sum(start * end, axis=-1))

        lower, upper = float(model.lower_position_limit[k]), float(model.upper_position_limit[k])
        assert turn(points[:-1], points[1:]).sum() == pytest.approx(upper - lower, abs=1e-5)
        pointer = tick[1] - tick[0]
        np.testing.assert_allclose(np.cross(tick[0] - center, pointer), 0.0, atol=1e-6)  # radial
        assert turn(points[0], pointer) == pytest.approx(float(q[k]) - lower, abs=1e-5)
        assert turn(pointer, points[-1]) == pytest.approx(upper - float(q[k]), abs=1e-5)
