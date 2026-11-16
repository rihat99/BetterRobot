"""The skeleton layout: glyphs of one rigid group stay clear, bones are thinner than their glyphs, end blocks small."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
import torch

import better_robot as br
from better_robot import collision
from better_robot.model import JointFixed, Model
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
    assert len(skeleton.slides) == 16
    for name, slide in skeleton.slides.items():
        if name.startswith("left_"):
            assert abs(skeleton.slides[name.replace("left_", "right_")]) == pytest.approx(abs(slide), abs=1e-9)
    assert dict(skeleton.shrinks) == pytest.approx({"left_hip_pitch_joint": 0.85, "right_hip_pitch_joint": 0.85})
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
    world, scale = placed["cylinders"]
    axes = _rotations(world[:, 3:])[:, :, 2]
    joined = np.zeros(len(glyphs), dtype=bool)
    bones = [i for i, part in enumerate(skeleton.cylinders.part) if part in ("bone", "stick")]
    assert bones
    for i in bones:
        for end in (world[i, :3] - 0.5 * scale[i, 2] * axes[i], world[i, :3] + 0.5 * scale[i, 2] * axes[i]):
            inside = np.linalg.norm(centers - end, axis=1) < 0.5 * radii  # the bone's cap is in these glyphs
            assert (scale[i, 0] <= radii[inside] + 1e-12).all(), (skeleton.cylinders.owner[i], scale[i], radii[inside])
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
