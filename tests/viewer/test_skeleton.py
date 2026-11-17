"""The skeleton layout: glyphs of one rigid group stay clear, bones are thinner than their glyphs, end blocks small."""

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
from better_robot.model import Box, Geometry, JointFixed, JointPrismatic, Model
from better_robot.viewer import RobotVisual
from better_robot.viewer.skeleton import MAXIMAL, MINIMAL, Skeleton, SkeletonRecipe, build_skeleton

from better_robot.exceptions import ArgumentError, UnknownNameError
from tests.support.robots import g1_urdf, panda_urdf

_GLYPH_PARTS = ("drum", "sleeve", "ball", "marker", "base")


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


def _arm(builder: br.io.ModelBuilder, *, thin: bool = False) -> None:
    """A fixed base and four revolute joints 25 cm apart along z, no axis along the line; ``thin``: the leaf's one
    shape is a 2 mm placeholder box."""
    builder.add_body("base", mass=1.0)
    builder.add_fixed("mount", parent="world", child="base")
    for k, axis in enumerate(((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))):
        builder.add_body(f"l{k}", mass=1.0)
        builder.add_revolute(f"j{k}", parent=f"l{k - 1}" if k else "base", child=f"l{k}", axis=axis,
                             origin=_at(z=0.25), lower=-2.0, upper=2.0)  # fmt: skip
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
    """A planar root, then a spherical, a helical and a revolute joint."""
    for body in ("base", "a", "b", "c"):
        builder.add_body(body, mass=1.0)
    builder.add_planar("slide", parent="world", child="base")
    builder.add_spherical("ball", parent="base", child="a", origin=_at(z=0.3))
    builder.add_helical("screw", parent="a", child="b", axis=(0.0, 0.0, 1.0), pitch=0.01, origin=_at(z=0.3),
                        lower=-3.0, upper=3.0)  # fmt: skip
    builder.add_revolute("tip", parent="b", child="c", origin=_at(z=0.3), **_HINGE)


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


#: Small models at the edges of the layout.
_SYNTHETIC: dict[str, Callable[[br.io.ModelBuilder], None]] = {
    "arm": _arm, "thin_tip": partial(_arm, thin=True), "world_hinge": partial(_on_world, slider=False),
    "world_slider": partial(_on_world, slider=True), "gantry": _gantry, "planar": _planar, "hand": _hand,
    "fixed": _fixed, "box": _box, "servo": _servo,
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
    blocks = skeleton.cylinders
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


def test_g1_glyphs_of_one_rigid_group_do_not_overlap() -> None:
    model, spheres = _g1()
    skeleton = build_skeleton(model, MAXIMAL, spheres=spheres)
    drums = _drums(skeleton, _poses(model))
    pairs = _same_group_pairs(model, set(drums))
    assert len(drums) == 29 and len(pairs) >= 28
    for a, b in pairs:  # the layout's own test is on the boxes around the drums, at neutral
        names = (model.joint_names[a], model.joint_names[b])
        assert not _overlap(drums[a], drums[b], round_=False), names
        assert not _overlap(drums[b], drums[a], round_=False), names
    # A drum is rigid in its parent group and symmetric about its axis, so the drums stay clear in every pose.
    velocity = 0.8 * torch.randn(model.nv, generator=torch.Generator().manual_seed(0), dtype=model.q_neutral.dtype)
    q = br.kinematics.integrate(model, model.q_neutral, velocity)
    drums = _drums(skeleton, _poses(model, q))
    for a, b in pairs:
        names = (model.joint_names[a], model.joint_names[b])
        assert not _overlap(drums[a], drums[b], round_=True), names
        assert not _overlap(drums[b], drums[a], round_=True), names


def test_g1_slides_are_mirror_images() -> None:
    model, spheres = _g1()
    skeleton = build_skeleton(model, MAXIMAL, spheres=spheres)
    assert len(skeleton.slides) == 14
    for name, slide in skeleton.slides.items():
        if name.startswith("left_"):
            assert abs(skeleton.slides[name.replace("left_", "right_")]) == pytest.approx(abs(slide), abs=1e-9)
    hips = dict.fromkeys(("left_hip_pitch_joint", "right_hip_pitch_joint"), 0.85)
    wrists = dict.fromkeys(("left_wrist_yaw_joint", "right_wrist_yaw_joint"), 0.85**4)  # a leaf shrinks in place
    assert dict(skeleton.shrinks) == pytest.approx(hips | wrists)
    poses = _poses(model)
    drums = _drums(skeleton, poses)
    mirror = np.array([1.0, -1.0, 1.0])  # the neutral root frame is the world: left is positive y
    lefts = [j for j in drums if model.joint_names[j].startswith("left_")]
    assert len(lefts) == 13
    for j in lefts:  # each drum's offset from its joint (the description's own origins differ by 10 micrometres)
        other = model.joint_names.index(model.joint_names[j].replace("left_", "right_"))
        offset, mirrored = drums[j][0] - poses[j, :3], drums[other][0] - poses[other, :3]
        np.testing.assert_allclose(offset * mirror, mirrored, atol=1e-6)
        np.testing.assert_allclose(drums[j][2], drums[other][2], atol=1e-9)


def test_g1_maximal_glyphs_narrow_towards_the_tips() -> None:
    model, spheres = _g1()
    skeleton = build_skeleton(model, MAXIMAL, spheres=spheres)
    drums, anchor = _drums(skeleton, _poses(model)), _anchors(model)
    size = {j: half[0] / skeleton.shrinks.get(model.joint_names[j], 1.0) for j, (_, _, half) in drums.items()}
    pairs = [(anchor[model.parent_joint_ids[j]], j) for j in drums if anchor[model.parent_joint_ids[j]] in drums]
    assert len(pairs) == 26  # every drum but the three on the free-flyer base
    for parent, child in pairs:
        assert size[child] <= size[parent] + 1e-12, (model.joint_names[parent], model.joint_names[child])


def test_panda_rails_span_the_finger_travel_in_the_hand_and_the_sleeves_mirror_each_other() -> None:
    model, spheres = _panda()
    skeleton = build_skeleton(model, MAXIMAL, spheres=spheres)
    cylinders = skeleton.cylinders
    rails = [i for i, part in enumerate(cylinders.part) if part == "rail"]
    assert len(rails) == 2 and {int(cylinders.body[i]) for i in rails} == {model.joint_names.index("panda_joint7")}
    np.testing.assert_allclose(cylinders.scale[rails, 2], 0.04, atol=1e-6)  # the 4 cm each finger travels
    assert skeleton.slides["panda_finger_joint1"] == pytest.approx(skeleton.slides["panda_finger_joint2"], abs=1e-9)


@pytest.mark.parametrize("robot", ["g1", "panda", "mixed"])
@pytest.mark.parametrize("recipe", [MAXIMAL, MINIMAL], ids=["maximal", "minimal"])
def test_bone_is_never_thicker_than_the_glyphs_it_joins(robot: str, recipe: SkeletonRecipe) -> None:
    model, spheres = _ROBOTS[robot]()
    skeleton = build_skeleton(model, recipe, spheres=spheres)
    placed = skeleton.place(_poses(model))
    glyphs = []  # (centre, radius): half the side for a box
    for mesh, per_scale in (("cylinders", 1.0), ("balls", 1.0), ("boxes", 0.5)):
        blocks, (world, scale) = getattr(skeleton, mesh), placed[mesh]
        glyphs += [
            (world[i, :3], per_scale * scale[i, 0]) for i in range(len(blocks)) if blocks.part[i] in _GLYPH_PARTS
        ]
    centers, radii = np.stack([center for center, _ in glyphs]), np.array([radius for _, radius in glyphs])
    bones = []  # (centre, axis, length, half thickness): a stick is a cylinder, a limb a box at its wide end
    for mesh, per_scale in (("cylinders", 1.0), ("boxes", 0.5)):
        blocks, (world, scale) = getattr(skeleton, mesh), placed[mesh]
        axes = _rotations(world[:, 3:])[:, :, 2]
        bones += [
            (world[i, :3], axes[i], scale[i, 2], per_scale * scale[i, 0])
            for i in range(len(blocks))
            if blocks.part[i] in ("bone", "stick")
        ]
    assert bones
    joined = np.zeros(len(glyphs), dtype=bool)
    for center, axis, length, half in bones:
        for end in (center - 0.5 * length * axis, center + 0.5 * length * axis):
            inside = np.linalg.norm(centers - end, axis=1) < 0.5 * radii  # the bone's cap is in these glyphs
            assert (half <= radii[inside] + 1e-12).all(), (center, half, radii[inside])
            joined |= inside
    assert joined.sum() >= len(glyphs) - 1  # every glyph but a free-flyer's cube inside its hull holds a bone end


def test_g1_minimal_has_13_clusters() -> None:
    model, spheres = _g1()
    clusters = build_skeleton(model, MINIMAL, spheres=spheres).clusters
    assert len(clusters) == 13
    assert {"left_hip_pitch_joint", "left_hip_roll_joint"} in [set(cluster) for cluster in clusters]
    assert len(build_skeleton(model, MINIMAL).clusters) == 13  # the clusters do not depend on the geometry


@pytest.mark.parametrize("robot", ["g1", "panda", "mixed"])
def test_end_blocks_are_no_larger_than_the_root_hull(robot: str) -> None:
    model, spheres = _ROBOTS[robot]()
    boxes = build_skeleton(model, MAXIMAL, spheres=spheres).boxes
    volume = boxes.scale.prod(axis=1)
    anchor = _anchors(model)
    roots = {j for j in range(model.njoints) if anchor[j] == 0 or model.parent_joint_ids[anchor[j]] == 0}
    root = [volume[i] for i, part in enumerate(boxes.part) if part == "hull" and boxes.body[i] in roots]
    ends = [volume[i] for i, part in enumerate(boxes.part) if part == "end"]
    assert root and ends
    assert max(ends) <= max(root)


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
        tips = [i for i, owner in enumerate(skeleton.balls.owner) if owner == knee and skeleton.balls.anchors[i, 0] < 0]
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

    from better_robot.viewer import RobotVisual  # noqa: PLC0415 - its collision mesh loader

    urdf = importlib.import_module(f"robot_descriptions.{name}_description").URDF_PATH
    model = br.load(urdf, root_joint=br.JointFreeFlyer())
    geometry = br.io.load_geometry(urdf, model)
    vertices = RobotVisual(model, model.q_neutral[None], geometry=geometry, root=urdf)._collision_vertices
    sources = {"geometry": geometry, "mesh_vertices": vertices}
    if spheres:
        sources["spheres"] = collision.load_spheres(collision.SPHERE_SETS / f"{name}.json", model)[0]
    return model, sources


def test_g1_layout_does_not_change_with_a_sphere_set() -> None:
    """A sphere set given to show as the collision overlay leaves the layout alone: the G1's bodies are sized from
    its collision meshes either way, so its feet stay soles (sized from the spheres they were limbs with toe balls)."""
    model, sources = _described("g1", spheres=True)
    plain = {key: value for key, value in sources.items() if key != "spheres"}
    for recipe in (MAXIMAL, MINIMAL):
        first, second = build_skeleton(model, recipe, **plain), build_skeleton(model, recipe, **sources)
        for mesh in ("cylinders", "balls", "boxes"):
            np.testing.assert_array_equal(getattr(first, mesh).local, getattr(second, mesh).local)
            np.testing.assert_array_equal(getattr(first, mesh).scale, getattr(second, mesh).scale)
    maximal = build_skeleton(model, MAXIMAL, **sources)
    for side in ("left", "right"):
        ankle = model.joint_names.index(f"{side}_ankle_roll_joint")
        blocks = zip(maximal.boxes.part, maximal.boxes.owner, strict=True)
        assert [part for part, owner in blocks if owner == ankle and part != "bone"] == ["end"]  # one sole
        assert ankle not in maximal.balls.owner  # and no toe ball


@pytest.mark.parametrize("name", ["g1", "go2"])
def test_minimal_follows_the_maximal_size_rule_and_draws_the_head(name: str) -> None:
    """Both styles share one size rule (a reference radius between 2 % and 4 % of the robot's size, the minimal one
    at ``MINIMAL.size`` of it: the Go2's minimal balls were 3.7 % of its size), and the minimal skeleton draws the
    trunk's head that the maximal one finds as one ball of the reference radius, rigid in the trunk, on a stick."""
    model, sources = _described(name)
    maximal, minimal = build_skeleton(model, MAXIMAL, **sources), build_skeleton(model, MINIMAL, **sources)
    assert 0.02 <= maximal.radius / maximal.scale <= 0.04
    assert minimal.radius == pytest.approx(MINIMAL.size * maximal.radius)
    trunk = model.joint_names.index("waist_pitch_joint" if name == "g1" else "root_joint")
    heads = [i for i, owner in enumerate(minimal.balls.owner) if owner == trunk and minimal.balls.anchors[i, 0] < 0]
    assert len(heads) == 1
    assert _anchors(model)[minimal.balls.body[heads[0]]] == trunk
    np.testing.assert_allclose(minimal.balls.scale[heads[0]], minimal.radius)
    placed = minimal.place(_poses(model))
    (balls, _), (sticks, scale) = placed["balls"], placed["cylinders"]
    axes = _rotations(sticks[:, 3:])[:, :, 2]
    ends = np.concatenate([sticks[:, :3] + sign * 0.5 * scale[:, 2:] * axes for sign in (-1.0, 1.0)])
    assert np.linalg.norm(ends - balls[heads[0], :3], axis=1).min() < 1e-5  # a stick ends in the head
    if name == "g1":  # above every other ball
        assert balls[heads[0], 2] == balls[:, 2].max()


@pytest.mark.parametrize("recipe", [MAXIMAL, MINIMAL], ids=["maximal", "minimal"])
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
    model, _ = _mixed()
    with pytest.raises(UnknownNameError, match="knee"):
        build_skeleton(model, SkeletonRecipe(joint_size={"knee": 2.0}))


@pytest.mark.parametrize("recipe", [MAXIMAL, MINIMAL], ids=["maximal", "minimal"])
@pytest.mark.parametrize("factors", [{"j1": 0.0}, {"j1": 0.5}, {"j3": 2.0}, {"j0": 0.0, "j2": 0.0}], ids=str)
def test_joint_size_scales_its_own_glyph_and_the_bones_run_on(recipe: SkeletonRecipe, factors: dict) -> None:
    """A factor scales one glyph after the size rules, so the glyphs after it keep theirs (a 2 at the tip wins over
    the narrowing towards it), and a glyph of factor 0 goes while every bone and stick stays."""
    model, _ = _synthetic("arm")
    mesh, glyph = ("cylinders", "drum") if recipe is MAXIMAL else ("balls", "ball")

    def glyphs(skeleton: Skeleton) -> dict[int, float]:  # joint -> radius of its drum or of its cluster's ball
        blocks = getattr(skeleton, mesh)
        rows = zip(blocks.owner, blocks.part, blocks.scale, blocks.anchors[:, 0], strict=True)
        drawn = [
            (owner, scale) for owner, part, scale, moving in rows if part == glyph and (moving >= 0 or mesh != "balls")
        ]
        return {int(owner): float(scale[0]) for owner, scale in drawn}

    def bones(skeleton: Skeleton) -> list[tuple[str, int]]:
        rows = [(name, getattr(skeleton, name)) for name in ("cylinders", "boxes")]
        return sorted((name, int(o)) for name, b in rows for o, p in zip(b.owner, b.part) if p in ("bone", "stick"))

    plain = build_skeleton(model, recipe)
    sized = build_skeleton(model, replace(recipe, joint_size=factors))
    expected = {j: factors.get(model.joint_names[j], 1.0) * radius for j, radius in glyphs(plain).items()}
    assert len(expected) == 4
    assert glyphs(sized) == pytest.approx({j: radius for j, radius in expected.items() if radius > 0.0})
    assert bones(sized) == bones(plain)


@pytest.mark.parametrize("name", [*_SYNTHETIC, "mixed", "g1", "panda"])
def test_every_block_line_and_bound_is_finite(name: str) -> None:
    """Both styles of the presets and of a recipe that hides every glyph, with limit arcs, sides and every hull, place
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
    for maximal, minimal in (
        (MAXIMAL, MINIMAL),
        (
            replace(MAXIMAL, limits=True, sides=True, hulls="all", joint_size=hidden),
            replace(MINIMAL, joint_size=hidden),
        ),
    ):
        visual = RobotVisual(model, q, geometry=geometry, root=root, maximal=maximal, minimal=minimal)
        for style in ("minimal", "maximal"):
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
    skeleton = build_skeleton(model, replace(MAXIMAL, limits=True))
    poses = _poses(model, torch.tensor([[-0.5, 0.0], [0.5, 1.0]], dtype=model.q_neutral.dtype))
    rider, blocks = model.joint_names.index("rider"), skeleton.cylinders
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
    skeleton = build_skeleton(model, MAXIMAL, geometry=geometry)
    blocks, tip = skeleton.cylinders, model.joint_names.index("j3")
    drum = next(i for i, (owner, part) in enumerate(zip(blocks.owner, blocks.part)) if owner == tip and part == "drum")
    assert blocks.scale[drum, 0] == pytest.approx(0.2 * skeleton.radius)


@pytest.mark.parametrize(("name", "size"), [("box", np.sqrt(0.1**2 + 0.2**2 + 0.3**2)), ("gantry", np.sqrt(3 * 0.06))])
def test_a_robot_whose_joints_share_one_point_takes_its_size_from_its_bodies(name: str, size: float) -> None:
    """When the joint origins span nothing (a lone floating body, sliders that share a point) the robot's size is the
    diagonal of its geometry's box and of its inertia boxes (sides of 0.245 m for an inertia of 0.01 kg m2 about each
    axis of 1 kg), not a default 1 m."""
    model, geometry = _synthetic(name)
    assert build_skeleton(model, MAXIMAL, geometry=geometry).scale == pytest.approx(size)


def test_colors_shade_each_part_and_tint_the_sides() -> None:
    model, spheres = _g1()
    hue = (0.2, 0.6, 0.4)
    roles = {"joint": 0.0, "face": -0.5, "bone": 0.5, "stick": 1.0}  # towards black below 0, towards white above

    def shade(color: tuple[float, float, float], role: str) -> tuple[float, float, float]:
        mix, target = abs(roles[role]), float(roles[role] > 0.0)
        red, green, blue = (channel + mix * (target - channel) for channel in color)
        return red, green, blue

    plain = build_skeleton(model, MAXIMAL, spheres=spheres)
    colors = plain.colors(hue, shade)
    parts = np.array(plain.cylinders.part)
    np.testing.assert_allclose(colors["cylinders"][parts == "drum"], np.broadcast_to(hue, (29, 3)), atol=1e-6)
    np.testing.assert_allclose(colors["cylinders"][parts == "face"][0], shade(hue, "face"), atol=1e-6)
    np.testing.assert_allclose(colors["boxes"][np.array(plain.boxes.part) == "hull"][0], shade(hue, "bone"), atol=1e-6)
    red = (1.0, 0.0, 0.0)
    recipe = SkeletonRecipe(sides=True, color={"left_knee_joint": red})
    sided = build_skeleton(model, recipe, spheres=spheres)
    drums = {
        model.joint_names[owner]: color
        for owner, part, color in zip(
            sided.cylinders.owner, sided.cylinders.part, sided.colors(hue, shade)["cylinders"]
        )
        if part == "drum"
    }
    np.testing.assert_allclose(drums["left_knee_joint"], red, atol=1e-6)  # a recipe colour wins over hue and side
    assert drums["left_elbow_joint"].sum() > drums["waist_yaw_joint"].sum() > drums["right_elbow_joint"].sum()
    np.testing.assert_allclose(drums["waist_yaw_joint"], hue, atol=1e-6)
