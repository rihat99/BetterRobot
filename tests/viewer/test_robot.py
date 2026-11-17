"""Robot adapter and mesh layers: poses compose correctly, FK runs once per clip."""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import ClassVar, final

import numpy as np
import pytest
import torch

import better_robot as br
from better_robot.lie import so3
from better_robot.lie.se3 import compose, from_translation
from better_robot.model import JointManifold, JointModel, JointPrismatic
from better_robot.viewer.layers.mesh import MeshLayer, MeshSpec, RigidMeshLayer, SpheresLayer
from better_robot.viewer.robot import RobotVisual, Skin
from better_robot.viewer.skeleton import MAXIMAL
from better_robot.viewer.theme import DARK, Theme, linear_uint8
from better_robot.viewer.viewer import Viewer

from better_robot import collision
from better_robot.exceptions import ArgumentError, ModelError, ShapeError, UnknownNameError
from better_robot.testing import FakeServer
from tests.support.robots import g1_urdf, panda_urdf


def _random_poses(shape: tuple[int, ...], seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    quat = rng.normal(size=(*shape, 4))
    quat /= np.linalg.norm(quat, axis=-1, keepdims=True)
    return np.concatenate([rng.normal(size=(*shape, 3)), quat], axis=-1).astype(np.float32)


def test_rigid_mesh_layer_moves_each_mesh_with_its_body() -> None:
    scene = FakeServer().scene
    body_poses = _random_poses((4, 3))
    origin = _random_poses((), 7)
    triangle = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    specs = [
        MeshSpec(triangle, np.array([[0, 1, 2]]), body=2, origin=origin),
        MeshSpec(triangle, np.array([[0, 1, 2]]), body=0, origin=np.array([0, 0, 0, 0, 0, 0, 1.0])),
    ]
    layer = RigidMeshLayer("meshes", specs, body_poses=body_poses)
    layer.build(scene, "/r/meshes")
    layer.set_step(3)
    expected = compose(torch.from_numpy(body_poses[3, 2]), torch.from_numpy(origin)).numpy()
    node = scene.nodes["/r/meshes/mesh_0"]
    np.testing.assert_allclose(node.position, expected[:3], atol=1e-6)
    np.testing.assert_allclose(node.wxyz, expected[[6, 3, 4, 5]], atol=1e-6)
    np.testing.assert_allclose(scene.nodes["/r/meshes/mesh_1"].position, body_poses[3, 0, :3])

    deforming = MeshLayer("soft", np.zeros((6, 3, 3)), faces=np.array([[0, 1, 2]]))
    deforming.build(scene, "/r/soft")
    deforming.set_step(5)
    assert deforming.nsteps == 6 and ("/r/soft/mesh", "vertices") in scene.log
    SpheresLayer("s", np.zeros((2, 4, 3)), radii=np.ones(4)).build(scene, "/r/s")
    assert scene.nodes["/r/s/spheres"].props["batched_scales"].shape == (4,)


def test_panda_visual_follows_forward_kinematics() -> None:
    """The tracked Panda ships its collision meshes only, so the overlay draws those, one node per shape."""
    model = br.load(panda_urdf())
    geometry = br.io.load_geometry(panda_urdf(), model)
    q = torch.stack([model.random_configuration(generator=torch.Generator().manual_seed(i)) for i in range(5)])
    state = br.forward_kinematics(model, q)
    server = FakeServer()
    viewer = Viewer(server=server, ground=False)
    robot = viewer.add(
        RobotVisual(
            model, q, geometry=geometry, root=panda_urdf(), style="maximal", collision=True, frames=["panda_hand"]
        )
    )
    viewer.show(4)
    nodes = server.scene.nodes
    shapes = [attached for attached in geometry.shapes if attached.collision]
    assert len(shapes) == 17  # nine STL meshes and eight finger boxes
    assert sum(name.startswith("/panda/panda/collision/mesh_") for name in nodes) == 17
    hand = model.body_id("panda_hand")
    index = next(i for i, attached in enumerate(shapes) if attached.body == hand)
    origin = np.asarray(shapes[index].shape.placement, dtype=np.float32)
    expected = compose(state.joint_poses[4, hand].float(), torch.from_numpy(origin)).numpy()
    np.testing.assert_allclose(nodes[f"/panda/panda/collision/mesh_{index}"].position, expected[:3], atol=1e-5)
    frame = state.frame_poses[4, model.frame_id("panda_hand")].numpy()
    np.testing.assert_allclose(nodes["/panda/panda/frames/axes"].batched_positions[0], frame[:3], atol=1e-5)
    assert robot.nsteps == 5 and robot.q.shape == (5, model.nq)


@final
@dataclass(frozen=True)
class _JointDrop(JointModel):
    """A joint kind defined outside the library: a translation along -z of the joint frame."""

    nq: ClassVar[int] = 1
    nv: ClassVar[int] = 1
    manifold: ClassVar[JointManifold] = JointPrismatic.manifold
    axis: ClassVar[None] = None

    def placement(self, q: torch.Tensor) -> torch.Tensor:
        return from_translation(torch.cat((q.new_zeros(*q.shape[:-1], 2), -q), dim=-1))

    def motion_subspace(self, q: torch.Tensor) -> torch.Tensor:
        return q.new_tensor([0.0, 0.0, -1.0, 0.0, 0.0, 0.0]).expand(*q.shape[:-1], 6).unsqueeze(-1)

    def neutral(self) -> torch.Tensor:
        return torch.zeros(1)

    def random(self, generator: torch.Generator, *, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
        return lower + (upper - lower) * torch.rand(1, generator=generator)


def test_robot_visual_draws_a_joint_kind_from_outside_the_library_as_a_ball() -> None:
    """C-EXT-2: a kind outside the built-ins takes the generic path, a marker ball, in both skeleton styles."""
    builder = br.ModelBuilder("drop")
    builder.add_body("base")
    builder.add_body("tip")
    builder.add_joint("drop", kind=_JointDrop(), parent="base", child="tip", lower=0.0, upper=0.5)
    model = br.io.build_model(builder.finalize())
    server = FakeServer()
    robot = Viewer(server=server, ground=False).add(RobotVisual(model, torch.tensor([[0.1], [0.2]])))
    assert robot.style == "maximal"
    assert len(server.scene.nodes["/drop/drop/maximal/balls"].props["batched_positions"]) == 1
    robot.set_style("minimal")
    assert "/drop/drop/minimal/balls" in server.scene.nodes


# ---------------------------------------------------------------------------------------------- the robot visual


def _g1_visual(q: torch.Tensor | None = None, **options: object) -> tuple[RobotVisual, FakeServer, Viewer]:
    model = br.load(g1_urdf(), root_joint=br.JointFreeFlyer())
    spheres, _ = collision.load_spheres(collision.SPHERE_SETS / "g1.json", model)
    server = FakeServer()
    viewer = Viewer(server=server, ground=False)
    robot = viewer.add(RobotVisual(model, q, spheres=spheres, **options))  # type: ignore[arg-type]
    return robot, server, viewer


def _drawn(server: FakeServer) -> dict[str, object]:
    """The robot's nodes that draw something: everything under its tag but the frames that group them."""
    nodes = server.scene.nodes.items()
    return {name: node for name, node in nodes if name.startswith("/g1_29dof/") and node.kind != "FrameHandle"}


def _positions(node: object, key: str = "batched_positions") -> np.ndarray:
    """What a batched node shows: its last write, or what it was created with."""
    return vars(node).get(key, node.props[key])  # type: ignore[attr-defined]


def _matrices(wxyz: np.ndarray) -> np.ndarray:
    """Rotation matrices ``(N, 3, 3)`` of viser's scalar-first quaternions ``(N, 4)``."""
    return so3.to_matrix(torch.from_numpy(np.asarray(wxyz, dtype=np.float64)[:, [1, 2, 3, 0]])).numpy()


def test_robot_visual_maximal_with_every_overlay_is_six_nodes_that_a_step_writes_once() -> None:
    model = br.load(g1_urdf(), root_joint=br.JointFreeFlyer())
    robot, server, viewer = _g1_visual(
        model.q_neutral.repeat(2, 1), collision=True, frames=True, maximal=replace(MAXIMAL, limits=True)
    )
    assert robot.style == "maximal" and robot.styles == ("minimal", "maximal")  # no geometry: no meshes
    path, moved = "/g1_29dof/g1_29dof", ["batched_positions", "batched_wxyzs"]
    server.scene.log.clear()
    viewer.show(1)
    writes = {name: sorted(event for node, event in server.scene.log if node == name) for name in _drawn(server)}
    assert writes == {
        **{f"{path}/maximal/{mesh}": moved for mesh in ("balls", "boxes", "cylinders")},
        f"{path}/maximal/limits": ["points"],  # the arcs and ticks
        f"{path}/frames/axes": moved,
        f"{path}/collision/spheres": ["batched_positions"],  # a ball needs no rotation
    }


@pytest.mark.parametrize("style", ["maximal", "minimal"])
def test_robot_visual_turns_its_skeleton_with_forward_kinematics(style: str) -> None:
    """A step writes rotations that follow forward kinematics, here far from the neutral configuration the skeleton is
    laid out at: a block rigid in a joint keeps its pose in that joint's frame, a drum's axis is its joint's axis,
    and each end of a stick is the centre of a ball or a point rigid in a joint."""
    model = br.load(g1_urdf(), root_joint=br.JointFreeFlyer())
    velocity = 0.8 * torch.randn(model.nv, generator=torch.Generator().manual_seed(5), dtype=model.q_neutral.dtype)
    q = torch.stack([model.q_neutral, br.kinematics.integrate(model, model.q_neutral, velocity)])
    robot, server, viewer = _g1_visual(q, style=style)
    viewer.show(1)
    poses = br.forward_kinematics(model, q).joint_poses.numpy().astype(np.float64)  # (2, njoints, 7)
    joints = [_matrices(poses[step][:, [6, 3, 4, 5]]) for step in range(2)]  # (njoints, 3, 3) per step
    skeleton, nodes = robot._skeleton(style), server.scene.nodes
    shown = {}  # mesh -> at steps 0 (as built) and 1: positions (N, 3), rotations (N, 3, 3), scales (N, 3)
    for mesh in ("cylinders", "balls", "boxes"):
        node, blocks = nodes[f"/g1_29dof/g1_29dof/{style}/{mesh}"], getattr(skeleton, mesh)
        shown[mesh] = [
            tuple(read(f"batched_{key}") for key in ("positions", "wxyzs", "scales"))
            for read in (node.props.__getitem__, partial(_positions, node))
        ]
        shown[mesh] = [(position, _matrices(wxyz), scale) for position, wxyz, scale in shown[mesh]]
        rigid = blocks.anchors[:, 0] < 0
        body = blocks.body[rigid]
        local = [
            (
                np.einsum("nki,nk->ni", joints[step][body], position[rigid] - poses[step, body, :3]),
                joints[step][body].transpose(0, 2, 1) @ rotation[rigid],
            )
            for step, (position, rotation, _) in enumerate(shown[mesh])
        ]
        np.testing.assert_allclose(local[1][0], local[0][0], atol=1e-5, err_msg=mesh)
        np.testing.assert_allclose(local[1][1], local[0][1], atol=1e-5, err_msg=mesh)
    blocks, rotation = skeleton.cylinders, shown["cylinders"][1][1]
    if style == "maximal":
        drums = [i for i, part in enumerate(blocks.part) if part == "drum"]
        axes = [joints[1][j] @ np.asarray(model.joints[j].axis) for j in blocks.owner[drums]]
        assert len(drums) == 29
        np.testing.assert_allclose(rotation[drums, :, 2], np.stack(axes), atol=1e-5)
        return
    sticks = blocks.anchors[:, 0] >= 0
    ends = [  # (2 N, 3) per step: both ends of each stick
        np.concatenate([p[sticks] + sign * 0.5 * s[sticks, 2:] * r[sticks, :, 2] for sign in (-1.0, 1.0)])
        for p, r, s in shown["cylinders"]
    ]
    in_joints = np.einsum("jki,njk->nji", joints[0], ends[0][:, None] - poses[0, None, :, :3])  # (2 N, njoints, 3)
    carried = np.einsum("jik,njk->nji", joints[1], in_joints) + poses[1, None, :, :3]
    rigid = np.linalg.norm(carried - ends[1][:, None], axis=-1).min(axis=1) < 1e-5
    balls = np.linalg.norm(ends[1][:, None] - shown["balls"][1][0][None], axis=-1).min(axis=1) < 1e-5
    assert sticks.sum() >= 12 and (rigid | balls).all(), np.flatnonzero(~(rigid | balls))


def test_robot_visual_builds_a_style_once_and_switches_by_visibility() -> None:
    robot, server, _ = _g1_visual()
    robot.set_style("minimal")
    robot.set_style("maximal")
    nodes = set(server.scene.nodes)
    path = "/g1_29dof/g1_29dof"
    assert {f"{path}/minimal", f"{path}/maximal"} <= nodes and f"{path}/collision" not in nodes  # parts are lazy
    server.scene.log.clear()
    robot.set_style("minimal")
    assert set(server.scene.nodes) == nodes  # the second switch creates nothing
    assert server.scene.nodes[f"{path}/minimal"].visible and not server.scene.nodes[f"{path}/maximal"].visible
    assert not any(name.startswith(f"{path}/maximal/") for name, _ in server.scene.log)  # a hidden part gets no write
    assert any(name.startswith(f"{path}/minimal/") for name, _ in server.scene.log)  # the shown one its step
    with pytest.raises(ArgumentError, match="meshes"):
        robot.set_style("meshes")


def test_robot_visual_update_equals_building_with_that_configuration() -> None:
    model = br.load(g1_urdf(), root_joint=br.JointFreeFlyer())
    velocity = 0.5 * torch.randn(model.nv, generator=torch.Generator().manual_seed(3), dtype=model.q_neutral.dtype)
    q = br.kinematics.integrate(model, model.q_neutral, velocity)
    updated, server_a, _ = _g1_visual(collision=True, frames=True)
    updated.set_style("minimal")
    updated.set_style("maximal")
    updated.update(q)
    built, server_b, _ = _g1_visual(q, collision=True, frames=True)
    built.set_style("minimal")
    built.set_style("maximal")
    for name, node in _drawn(server_b).items():
        other = server_a.scene.nodes[name]
        if "minimal" in name:  # hidden when the update came: it catches up when it is shown again
            continue
        np.testing.assert_allclose(_positions(other), _positions(node), atol=1e-6, err_msg=name)
    updated.set_style("minimal")
    built.set_style("minimal")
    for mesh in ("cylinders", "balls"):
        name = f"/g1_29dof/g1_29dof/minimal/{mesh}"
        np.testing.assert_allclose(
            _positions(server_a.scene.nodes[name]), _positions(server_b.scene.nodes[name]), atol=1e-6
        )
    np.testing.assert_allclose(updated.bounds(0)[0], built.bounds(0)[0], atol=1e-6)
    with pytest.raises(ShapeError, match="steps"):
        updated.update(torch.stack([q, q]))


def test_robot_visuals_take_one_hue_each_also_after_a_replace_and_share_the_sidebar_controls() -> None:
    model = br.load(panda_urdf())
    server = FakeServer()
    viewer = Viewer(server=server, ground=False)
    first = viewer.add(RobotVisual(model, name="left", style="maximal"))
    second = viewer.add(RobotVisual(model, name="right", style="maximal"))
    colors = [
        server.scene.nodes[f"/{name}/{name}/maximal/cylinders"].props["batched_colors"] for name in ("left", "right")
    ]
    assert not np.array_equal(colors[0], colors[1])
    styles = [control for control in server.gui.controls if control.name == "Style"]
    assert len(styles) == 1  # one Style control for both robots
    styles[0].value = "minimal"
    styles[0].fire(None)
    assert first.style == second.style == "minimal"
    third = RobotVisual(model, model.q_neutral.repeat(3, 1), name="left")  # a clip of another length replaces one
    viewer.replace(first, third)
    assert third.hue == first.hue != second.hue


def test_robot_visual_shows_a_skin_bound_at_its_own_configuration() -> None:
    model = br.load(panda_urdf())
    generator = torch.Generator().manual_seed(0)
    bind, q = (model.random_configuration(generator=generator) for _ in range(2))
    poses = br.forward_kinematics(model, torch.stack([bind, q])).joint_poses.numpy()
    hand = model.body_id("panda_hand")
    vertices = poses[0, hand, :3] + 0.05 * np.eye(3, dtype=np.float32)  # a triangle that rides on the hand
    weights = np.zeros((3, model.njoints), dtype=np.float32)
    weights[:, hand] = 1.0
    skin = Skin(vertices, np.array([[0, 1, 2]]), weights=weights, bind=bind.numpy())
    server = FakeServer()
    robot = Viewer(server=server, ground=False).add(RobotVisual(model, q, skin=skin))
    assert robot.style == "meshes" and robot.styles == ("meshes", "minimal", "maximal")
    mesh = server.scene.nodes["/panda/panda/meshes/mesh"]
    np.testing.assert_allclose(mesh.props["bone_positions"], poses[0, :, :3], atol=1e-6)
    np.testing.assert_allclose(mesh.bones[hand].position, poses[1, hand, :3], atol=1e-6)


def test_robot_visual_frames_are_every_moving_joint_or_the_named_frames() -> None:
    model = br.load(panda_urdf())
    server = FakeServer()
    robot = Viewer(server=server, ground=False).add(RobotVisual(model, frames=True))
    axes = "/panda/panda/frames/axes"
    assert len(server.scene.nodes[axes].props["batched_positions"]) == 9  # seven arm joints and two fingers
    robot.set_frames(["panda_hand", model.frame_id("panda_link3")])  # by name or by id
    assert len(server.scene.nodes[axes].props["batched_positions"]) == 2 and not server.scene.nodes[axes].removed
    robot.set_frames(False)
    assert not server.scene.nodes["/panda/panda/frames"].visible and not server.scene.nodes[axes].removed
    robot.set_frames(["panda_hand", "panda_link3"])  # the same set again: shown, not rebuilt
    assert server.scene.nodes["/panda/panda/frames"].visible and not server.scene.nodes[axes].removed
    with pytest.raises(UnknownNameError, match="panda_nose"):
        robot.set_frames(["panda_nose"])


@pytest.mark.parametrize("frames", [True, ["panda_hand"]])
def test_robot_visual_update_moves_its_frames_while_they_are_hidden(frames: bool | list[str]) -> None:
    """Hidden triads follow ``update`` too, so they are in place when they are shown again."""
    model = br.load(panda_urdf())
    server = FakeServer()
    robot = Viewer(server=server, ground=False).add(RobotVisual(model, frames=frames))
    robot.set_frames(False)
    q = model.random_configuration(generator=torch.Generator().manual_seed(0))
    robot.update(q)
    robot.set_frames(frames)
    state = br.forward_kinematics(model, q)
    first = state.joint_poses[2, :3] if frames is True else state.frame_pose_of("panda_hand")[:3]  # panda_joint1
    np.testing.assert_allclose(server.scene.nodes["/panda/panda/frames/axes"].batched_positions[0], first, atol=1e-6)


def test_robot_visual_bounds_are_the_box_of_the_robot_at_the_step() -> None:
    """``Viewer.aim`` fits this box to the view, so it holds the joints and stays within the sphere set's box."""
    model = br.load(g1_urdf(), root_joint=br.JointFreeFlyer())
    spheres, _ = collision.load_spheres(collision.SPHERE_SETS / "g1.json", model)
    q = model.q_neutral.repeat(2, 1)
    q[1, :3] = torch.tensor([3.0, -2.0, 1.0])
    robot = RobotVisual(model, q, spheres=spheres)
    state = br.forward_kinematics(model, q)
    joints, centers = state.joint_poses[:, 1:, :3].numpy(), spheres.centers(state).numpy()
    radii = spheres.radii.numpy()[:, None]
    for step in range(2):
        lows, highs = robot.bounds(step)  # a box per joint and per block
        low, high = lows.min(axis=0), highs.max(axis=0)
        assert (low <= joints[step].min(axis=0)).all() and (joints[step].max(axis=0) <= high).all()
        assert (low >= (centers[step] - radii).min(axis=0) - 0.05).all()
        assert (high <= (centers[step] + radii).max(axis=0) + 0.05).all()


def test_robot_visual_leaves_nothing_behind_when_removed_and_builds_again_when_added() -> None:
    robot, server, viewer = _g1_visual(collision=True, frames=True)
    robot.set_style("minimal")
    built = [name for name in server.scene.nodes if name.startswith("/g1_29dof")]
    viewer.remove(robot)
    assert all(server.scene.nodes[name].removed for name in built)
    viewer.add(robot)
    live = sorted(name for name, node in _drawn(server).items() if not node.removed)  # type: ignore[attr-defined]
    assert robot.style == "minimal" and live == [
        f"/g1_29dof/g1_29dof/{node}"
        for node in ("collision/spheres", "frames/axes", "minimal/balls", "minimal/boxes", "minimal/cylinders")
    ]


def test_robot_visual_follows_a_theme_with_other_hues_and_another_overlay_colour() -> None:
    robot, server, viewer = _g1_visual(collision=True)
    path = "/g1_29dof/g1_29dof"
    before = server.scene.nodes[f"{path}/maximal/cylinders"].props["batched_colors"].copy()
    red, blue = (0.8, 0.1, 0.1), (0.1, 0.2, 0.9)
    viewer.set_theme(Theme(hues=(red,), collision=blue))
    after = server.scene.nodes[f"{path}/maximal/cylinders"].batched_colors
    assert not np.array_equal(before, after) and (after[:, 0] >= after[:, 2]).all()  # shades of red
    spheres = server.scene.nodes[f"{path}/collision/spheres"]
    assert not spheres.removed and (spheres.props["batched_colors"] == linear_uint8(blue)).all()
    assert robot.style == "maximal"


def test_robot_visual_reads_each_mesh_file_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both skeletons size the Panda's bodies from its nine collision meshes, the overlay draws them, and a theme
    switch rebuilds the overlay in its new colour: each file is read once."""
    import trimesh  # noqa: PLC0415 - optional dependency of the viewer

    reads: list[str] = []
    load = trimesh.load
    monkeypatch.setattr(trimesh, "load", lambda path, **options: reads.append(str(path)) or load(path, **options))
    model = br.load(panda_urdf())
    geometry = br.io.load_geometry(panda_urdf(), model)
    viewer = Viewer(server=FakeServer(), ground=False)
    robot = viewer.add(RobotVisual(model, geometry=geometry, root=panda_urdf(), style="maximal", collision=True))
    robot.set_style("minimal")
    viewer.set_theme(DARK)
    assert len(reads) == len(set(reads)) == 9


@pytest.mark.parametrize("content", [None, b"solid empty\nendsolid empty\n"], ids=["missing", "empty"])
def test_a_collision_mesh_that_does_not_load_raises_for_the_skeleton_too(tmp_path: Path, content: bytes | None) -> None:
    """The skeleton sizes a body from its collision mesh: a file that is missing or holds no mesh raises the
    ``ModelError`` the meshes style raises, naming the body and the file, rather than falling back to the inertia."""
    if content is not None:
        (tmp_path / "part.stl").write_bytes(content)
    collision = '<collision><geometry><mesh filename="part.stl"/></geometry></collision>'
    urdf = tmp_path / "r.urdf"
    urdf.write_text(f'<robot name="r"><link name="base">{collision}</link></robot>')
    model = br.load(urdf)
    robot = RobotVisual(model, geometry=br.io.load_geometry(urdf, model), root=urdf, style="maximal")
    with pytest.raises(ModelError, match=r"body \d+: mesh 'part.stl' (is not a file|does not load|has no faces)"):
        Viewer(server=FakeServer(), ground=False).add(robot)


def test_two_robots_of_one_model_need_a_name_each() -> None:
    model = br.load(panda_urdf())
    viewer = Viewer(server=FakeServer(), ground=False)
    viewer.add(RobotVisual(model))
    with pytest.raises(ArgumentError, match="already has a layer named 'panda'"):
        viewer.add(RobotVisual(model))
    viewer.add(RobotVisual(model, name="second"))
    assert [layer.name for layer in viewer.layers] == ["panda", "second"]


def test_robot_visual_minimal_writes_lengths_to_the_stick_node_only() -> None:
    _, server, viewer = _g1_visual(style="minimal")
    server.scene.log.clear()
    viewer.show(0)
    assert [name.rsplit("/", 1)[-1] for name, event in server.scene.log if event == "batched_scales"] == ["cylinders"]


def test_aim_frames_a_moving_robot_by_what_it_covers() -> None:
    """A fixed-base G1 sweeps its joints: over the clip the robot spans 60 to 80 % of the view's limiting side.

    One box round the clip, seen from above, put it at 45 %: the far and near corners of that box lie far outside
    what a thin robot covers.
    """
    model = br.load(g1_urdf())
    phase = torch.linspace(0.0, 2.0 * torch.pi, 60)[:, None]
    middle = (model.lower_position_limit + model.upper_position_limit) / 2.0
    half = (model.upper_position_limit - model.lower_position_limit) / 2.0
    q = middle + 0.4 * half * torch.sin(phase + torch.arange(model.nq) * 0.7)
    viewer = Viewer(server=FakeServer())
    robot = viewer.add(RobotVisual(model, q))
    position, look_at = viewer.aim()
    points = np.concatenate([end for step in range(60) for end in robot.bounds(step)])  # two corners of every box
    sight = (look_at - position) / np.linalg.norm(look_at - position)
    right = np.cross(sight, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    rays = points - position
    depth, tan = rays @ sight, np.tan(np.radians(20.0))  # the browsers' vertical field of view is 40 degrees
    tall, wide = rays @ np.cross(right, sight) / depth / tan, rays @ right / depth / (tan * 16.0 / 9.0)
    assert max(np.abs(tall).max(), np.abs(wide).max()) < 1.0  # every step inside the view, sampled or not
    assert 0.6 < max(np.ptp(tall), np.ptp(wide)) / 2.0 <= 0.8
