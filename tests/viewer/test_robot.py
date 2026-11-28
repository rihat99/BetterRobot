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
from better_robot.model import JointManifold, JointModel, JointPrismatic, JointRevolute
from better_robot.viewer.layers.mesh import MeshLayer, MeshSpec, RigidMeshLayer, SpheresLayer
from better_robot.viewer.robot import RobotVisual, Skin, load_meshes
from better_robot.viewer.skeleton import MEDIUM, build_skeleton
from better_robot.viewer.theme import DARK, LIGHT, Theme, linear_uint8
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
            model, q, geometry=geometry, root=panda_urdf(), style="medium", collision=True, frames=["panda_hand"]
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


def test_the_collision_glass_stands_a_little_off_its_shapes() -> None:
    """viser has no depth offset: glass drawn on the very surface of a display mesh (a G1 names one file for both, a
    Panda's hull meets its mesh within a millimetre) showed stripes of each in turn."""
    model = br.load(panda_urdf())
    geometry = br.io.load_geometry(panda_urdf(), model)
    server = FakeServer()
    Viewer(server=server, ground=False).add(
        RobotVisual(model, geometry=geometry, root=panda_urdf(), style="medium", collision=True)
    )
    for k, shape in enumerate(load_meshes(geometry, root=panda_urdf(), collision=True)):  # the shapes as they are
        drawn = server.scene.nodes[f"/panda/panda/collision/mesh_{k}"].props["vertices"]
        moved = drawn - shape.vertices
        np.testing.assert_allclose(np.linalg.norm(moved, axis=1), 0.002 * 1.044, rtol=2e-3)  # of the arm's size
        assert np.mean(np.sum(moved * (shape.vertices - shape.vertices.mean(axis=0)), axis=1) > 0) > 0.9  # outwards


def test_load_meshes_gives_the_shapes_as_arrays_the_overlay_draws() -> None:
    """The tracked Panda's collision shapes as arrays outside a viewer: each STL file's vertices and faces, scaled,
    with its body and placement, which the collision overlay draws a little off the surface (the test above); its
    display meshes are not tracked, so asking for them names the first missing file."""
    import trimesh  # noqa: PLC0415 - optional dependency of the viewer

    model = br.load(panda_urdf())
    geometry = br.io.load_geometry(panda_urdf(), model)
    specs = load_meshes(geometry, root=panda_urdf(), collision=True)
    shapes = [attached for attached in geometry.shapes if attached.collision]
    assert [spec.body for spec in specs] == [attached.body for attached in shapes]
    for spec, attached in zip(specs, shapes, strict=True):
        np.testing.assert_array_equal(spec.origin, np.asarray(attached.shape.placement, dtype=np.float32))
        assert spec.vertices.ndim == spec.faces.ndim == 2 and spec.faces.max() < len(spec.vertices)
    link0 = next(i for i, attached in enumerate(shapes) if attached.body == model.body_id("panda_link0"))
    stl = trimesh.load(panda_urdf().parent / "meshes" / "collision" / "link0.stl", force="mesh")
    np.testing.assert_allclose(specs[link0].vertices, stl.vertices * shapes[link0].shape.scale, rtol=1e-6)
    np.testing.assert_array_equal(specs[link0].faces, stl.faces)
    with pytest.raises(ModelError, match=r"body 1: mesh 'package://example-robot-data/.*link0.dae' is not a file"):
        load_meshes(geometry, root=panda_urdf())


def test_a_mesh_file_whose_material_file_is_missing_takes_the_shapes_colour(tmp_path: Path) -> None:
    """An OBJ with texture coordinates that names a missing material file (the menagerie's Spot) loads as a texture
    of trimesh's grey stand-in, which drew black: it has no look of its own and takes the shape's colour."""
    (tmp_path / "part.obj").write_text(
        "mtllib material.mtl\nusemtl material_0\nv 0 0 0\nv 0.1 0 0\nv 0 0.1 0\nv 0 0 0.1\n"
        "vt 0 0\nvt 1 0\nvt 0 1\nvt 1 1\nf 1/1 3/3 2/2\nf 1/1 2/2 4/4\nf 1/1 4/4 3/3\nf 2/2 3/3 4/4\n"
    )
    urdf = tmp_path / "r.urdf"
    urdf.write_text(
        '<robot name="r"><link name="base"><visual><geometry><mesh filename="part.obj"/></geometry>'
        '<material name="wrap"><color rgba="0.9 0.7 0.2 1"/></material></visual></link></robot>'
    )
    (spec,) = load_meshes(br.io.load_geometry(urdf, br.load(urdf)), root=urdf)
    assert spec.trimesh is None and spec.color == pytest.approx((0.9, 0.7, 0.2))


def test_load_meshes_refuses_a_model_for_the_geometry_and_a_collision_that_is_not_a_bool() -> None:
    """A ``Model`` in place of its ``Geometry`` raised the builtin ``AttributeError`` of a missing ``shapes``, and
    ``collision="yes"`` matched no shape and returned ``[]`` (R1 n4): both are the library's ``ArgumentError``."""
    model = br.load(panda_urdf())
    geometry = br.io.load_geometry(panda_urdf(), model)
    with pytest.raises(ArgumentError, match="takes a Geometry.*got Model"):
        load_meshes(model, root=panda_urdf())  # pyright: ignore[reportArgumentType]
    with pytest.raises(ArgumentError, match="collision must be True or False, got 'yes'"):
        load_meshes(geometry, root=panda_urdf(), collision="yes")  # pyright: ignore[reportArgumentType]


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
    assert robot.style == "medium"
    assert len(server.scene.nodes["/drop/drop/medium/balls"].props["batched_positions"]) == 1
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


def test_robot_visual_skeleton_with_every_overlay_is_six_nodes_that_a_step_writes_once() -> None:
    model = br.load(g1_urdf(), root_joint=br.JointFreeFlyer())
    robot, server, viewer = _g1_visual(
        model.q_neutral.repeat(2, 1), collision=True, frames=True, style="medium", medium=replace(MEDIUM, limits=True)
    )
    assert robot.styles == ("minimal", "medium")  # no geometry: no meshes
    path, moved = "/g1_29dof/g1_29dof", ["batched_positions", "batched_wxyzs"]
    server.scene.log.clear()
    viewer.show(1)
    writes = {name: sorted(event for node, event in server.scene.log if node == name) for name in _drawn(server)}
    assert writes == {
        **{f"{path}/medium/{mesh}": moved for mesh in ("balls", "boxes")},
        f"{path}/medium/cylinders": sorted([*moved, "batched_scales"]),  # the rods follow their ends, as sticks do
        f"{path}/medium/limits": ["points"],  # the arcs and ticks
        f"{path}/frames/axes": moved,
        f"{path}/collision/spheres": ["batched_positions"],  # a ball needs no rotation
    }


@pytest.mark.parametrize("style", ["medium", "minimal"])
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
        node, blocks = nodes[f"/g1_29dof/g1_29dof/{style}/{mesh}"], skeleton.blocks[mesh]
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
    blocks, rotation = skeleton.blocks["cylinders"], shown["cylinders"][1][1]
    if style != "minimal":
        drums = [i for i, part in enumerate(blocks.part) if part == "drum"]
        axes = [joints[1][j] @ np.asarray(model.joints[j].axis) for j in blocks.owner[drums]]
        assert len(drums) == sum(isinstance(joint, JointRevolute) for joint in model.joints)
        np.testing.assert_allclose(rotation[drums, :, 2], np.stack(axes), atol=1e-5)
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
    robot.set_style("medium")
    nodes = set(server.scene.nodes)
    path = "/g1_29dof/g1_29dof"
    assert {f"{path}/minimal", f"{path}/medium"} <= nodes and f"{path}/collision" not in nodes  # parts are lazy
    server.scene.log.clear()
    robot.set_style("minimal")
    assert set(server.scene.nodes) == nodes  # the second switch creates nothing
    assert server.scene.nodes[f"{path}/minimal"].visible and not server.scene.nodes[f"{path}/medium"].visible
    assert not any(name.startswith(f"{path}/medium/") for name, _ in server.scene.log)  # a hidden part gets no write
    assert any(name.startswith(f"{path}/minimal/") for name, _ in server.scene.log)  # the shown one its step
    with pytest.raises(ArgumentError, match="meshes"):
        robot.set_style("meshes")


def test_a_hidden_robot_builds_a_part_switched_on_only_when_it_is_shown() -> None:
    """The showcase keeps seven robots hidden: a Style or Collision switch built the new part of each of them."""
    robot, server, viewer = _g1_visual(style="medium")
    viewer.set_visible(robot, False)
    nodes = set(server.scene.nodes)
    robot.set_style("minimal")
    robot.set_collision(visible=True)
    robot.set_frames(True)
    assert set(server.scene.nodes) == nodes  # nothing built while hidden
    viewer.set_visible(robot, True)
    path = "/g1_29dof/g1_29dof"
    for part in ("minimal", "collision", "frames"):
        assert server.scene.nodes[f"{path}/{part}"].visible
    assert not server.scene.nodes[f"{path}/medium"].visible


def test_a_robot_shown_by_its_own_switch_shows_the_viewers_step() -> None:
    """``robot.set_visible(True)`` catches the robot up to the step the viewer moved to while it was hidden, as
    ``Viewer.set_visible`` does (R2 m1: its skeleton, overlay and triads stayed at the step they were hidden at)."""
    model = br.load(g1_urdf(), root_joint=br.JointFreeFlyer())
    velocity = torch.randn(model.nv, generator=torch.Generator().manual_seed(3), dtype=model.q_neutral.dtype)
    q = torch.stack([br.kinematics.integrate(model, model.q_neutral, k * velocity) for k in range(3)])
    shown, server_a, viewer_a = _g1_visual(q, style="medium", collision=True, frames=True)
    hidden, server_b, viewer_b = _g1_visual(q, style="medium", collision=True, frames=True)
    hidden.set_visible(False)
    viewer_a.show(2)
    viewer_b.show(2)
    hidden.set_visible(True)
    for name, node in _drawn(server_a).items():
        np.testing.assert_array_equal(_positions(server_b.scene.nodes[name]), _positions(node), err_msg=name)


def test_robot_visual_update_equals_building_with_that_configuration() -> None:
    model = br.load(g1_urdf(), root_joint=br.JointFreeFlyer())
    velocity = 0.5 * torch.randn(model.nv, generator=torch.Generator().manual_seed(3), dtype=model.q_neutral.dtype)
    q = br.kinematics.integrate(model, model.q_neutral, velocity)
    updated, server_a, _ = _g1_visual(collision=True, frames=True)
    updated.set_style("minimal")
    updated.set_style("medium")
    updated.update(q)
    built, server_b, _ = _g1_visual(q, collision=True, frames=True)
    built.set_style("minimal")
    built.set_style("medium")
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


def test_robot_visual_update_with_one_configuration_holds_it_at_every_step_of_the_clip() -> None:
    """``update(q (nq,))`` on a clip of three steps draws ``q`` at each step, as ``joint_values (njoints,)`` does; a
    ``(T, nq)`` of another length still raises."""
    model = br.load(panda_urdf())
    generator = torch.Generator().manual_seed(1)
    clip = torch.stack([model.random_configuration(generator=generator) for _ in range(3)])
    q = model.random_configuration(generator=generator)
    servers, viewers = (FakeServer(), FakeServer()), []
    for server in servers:
        viewers.append(Viewer(server=server, ground=False))
    updated = viewers[0].add(RobotVisual(model, clip, style="medium"))
    built = viewers[1].add(RobotVisual(model, q.expand(3, -1), style="medium"))
    updated.update(q)
    assert torch.equal(updated.q, built.q)
    for step in range(3):
        for viewer in viewers:
            viewer.show(step)
        np.testing.assert_allclose(updated.bounds(step)[0], built.bounds(step)[0], atol=1e-6)
        for name, node in servers[1].scene.nodes.items():
            if "batched_positions" in node.props:
                np.testing.assert_allclose(_positions(servers[0].scene.nodes[name]), _positions(node), atol=1e-6)
    with pytest.raises(ShapeError, match="steps"):
        updated.update(q[None])


def test_robot_visual_refuses_a_hue_that_is_not_three_numbers() -> None:
    model = br.load(panda_urdf())
    with pytest.raises(ShapeError, match="hue"):
        RobotVisual(model, hue=(1.0, 0.0))  # type: ignore[arg-type]


def test_robot_visual_keeps_a_detached_copy_of_q() -> None:
    model = br.load(panda_urdf())
    q = model.q_neutral.clone().requires_grad_()
    robot = RobotVisual(model, q)
    with torch.no_grad():
        q += 0.1  # the caller goes on with its tensor
    assert not robot.q.requires_grad and torch.equal(robot.q[0], model.q_neutral)
    robot.update(q)
    assert not robot.q.requires_grad and robot.q.data_ptr() != q.data_ptr()


def test_robot_visuals_take_one_hue_each_also_after_a_replace_and_share_the_sidebar_controls() -> None:
    model = br.load(panda_urdf())
    server = FakeServer()
    viewer = Viewer(server=server, ground=False)
    first = viewer.add(RobotVisual(model, name="left", style="medium"))
    second = viewer.add(RobotVisual(model, name="right", style="medium"))
    colors = [
        server.scene.nodes[f"/{name}/{name}/medium/cylinders"].props["batched_colors"] for name in ("left", "right")
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


def test_a_robot_that_joins_the_shared_controls_keeps_its_own_state_and_moves_them_to_it() -> None:
    """The showcase's case: a robot added after a choice in the sidebar kept its own style while the Style control
    still showed the choice, and code could not read a robot's collision and frames switches."""
    model = br.load(panda_urdf())
    geometry = br.io.load_geometry(panda_urdf(), model)
    server = FakeServer()
    viewer = Viewer(server=server, ground=False)
    panda = partial(RobotVisual, model, geometry=geometry, root=panda_urdf())  # the tracked one draws no meshes
    first = viewer.add(panda(name="left", style="minimal"))
    controls = {label: server.gui.control(label) for label in ("Style", "Collision", "Frames")}
    for label, value in (("Style", "medium"), ("Collision", True), ("Frames", True)):
        controls[label].value = value  # a choice in the browser
        controls[label].fire(None)
    assert (first.style, first.collision, first.frames) == ("medium", True, True)
    hand = model.frame_id("panda_hand")
    second = viewer.add(panda(name="right", style="minimal", frames=[hand]))
    assert (second.style, second.collision, second.frames) == ("minimal", False, (hand,))  # its own arguments win
    assert [controls[label].value for label in ("Style", "Collision", "Frames")] == ["minimal", False, True]
    assert (first.style, first.collision, first.frames) == ("medium", True, True)  # and no other robot changes


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
    assert robot.style == "meshes" and robot.styles == ("meshes", "minimal", "medium")
    mesh = server.scene.nodes["/panda/panda/meshes/mesh"]
    np.testing.assert_allclose(mesh.props["bone_positions"], poses[0, :, :3], atol=1e-6)
    np.testing.assert_allclose(mesh.bones[hand].position, poses[1, hand, :3], atol=1e-6)


def test_a_ghost_robot_draws_every_style_see_through_and_without_shadows() -> None:
    """``opacity`` reaches every surface node of every style, those built before ``set_opacity`` and after it, and a
    see-through node casts and catches no shadow; the collision glass keeps the theme's opacity."""
    model = br.load(panda_urdf())
    spheres, _ = collision.load_spheres(collision.SPHERE_SETS / "panda.json", model)
    hand = model.body_id("panda_hand")
    weights = np.zeros((3, model.njoints), dtype=np.float32)
    weights[:, hand] = 1.0
    skin = Skin(0.05 * np.eye(3, dtype=np.float32), np.array([[0, 1, 2]]), weights=weights)
    server = FakeServer()
    robot = Viewer(server=server, ground=False).add(
        RobotVisual(model, skin=skin, spheres=spheres, opacity=0.35, style="medium", collision=True)
    )
    robot.set_style("meshes")

    def drawn(*parts: str) -> list[str]:
        under = tuple(f"/panda/panda/{part}/" for part in parts)
        return [
            name for name, node in server.scene.nodes.items() if name.startswith(under) and node.kind != "FrameHandle"
        ]

    def look(name: str) -> tuple[object, ...]:
        node = server.scene.nodes[name]
        return tuple(vars(node).get(key, node.props[key]) for key in ("opacity", "cast_shadow", "receive_shadow"))

    assert len(drawn("meshes", "medium")) == 4  # the skin, and the cylinders, balls and boxes of the glyphs and rods
    assert {look(name) for name in drawn("meshes", "medium")} == {(0.35, False, False)}
    robot.set_opacity(0.6)  # live, and in the styles built after it
    robot.set_style("minimal")
    robot.set_style("medium")
    assert len(drawn(*robot.styles)) == 7 and {look(name) for name in drawn(*robot.styles)} == {(0.6, False, False)}
    assert look("/panda/panda/collision/spheres") == (LIGHT.collision_opacity, False, False)  # the glass keeps its own
    robot.set_opacity(1.0)  # solid again: viser's opaque, and shadows
    assert {look(name) for name in drawn(*robot.styles)} == {(None, True, True)}
    with pytest.raises(ArgumentError, match="opacity"):
        robot.set_opacity(0.0)


def test_a_ghost_keeps_the_own_colours_and_texture_of_its_mesh_files_see_through(tmp_path: Path) -> None:
    """A mesh file with its own texture or colours (a Panda's, a Go2's) is sent as a GLB whose material takes the
    ghost's opacity, blended, with the texture and the colours kept and no shadow; opaque again, it is the file's
    own GLB with shadows (R2 M1: such a robot stayed solid)."""
    import json  # noqa: PLC0415 - one test reads a GLB's JSON chunk

    import trimesh  # noqa: PLC0415 - optional dependency of the viewer
    from PIL import Image  # noqa: PLC0415

    textured = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
    texture = Image.new("RGB", (2, 2), (200, 30, 30))
    textured.visual = trimesh.visual.TextureVisuals(uv=textured.vertices[:, :2] + 0.5, image=texture)
    textured.export(tmp_path / "textured.glb")
    painted = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
    painted.visual.face_colors = np.where(np.arange(12)[:, None] < 6, (200, 30, 30, 255), (30, 30, 200, 255))
    painted.export(tmp_path / "painted.ply")
    urdf = tmp_path / "r.urdf"
    urdf.write_text(
        '<robot name="r"><link name="base"><visual><geometry><mesh filename="textured.glb"/></geometry></visual>'
        '</link><link name="arm"><visual><geometry><mesh filename="painted.ply"/></geometry></visual></link>'
        '<joint name="hinge" type="revolute"><parent link="base"/><child link="arm"/><origin xyz="0 0 0.3"/>'
        '<axis xyz="0 1 0"/><limit lower="-1" upper="1" effort="10" velocity="1"/></joint></robot>'
    )
    model = br.load(urdf)
    server = FakeServer()
    robot = Viewer(server=server, ground=False).add(
        RobotVisual(model, geometry=br.io.load_geometry(urdf, model), root=urdf, opacity=0.35)
    )
    nodes = [server.scene.nodes[f"/r/r/meshes/mesh_{k}"] for k in range(2)]

    def look(node: object) -> tuple[dict[str, object], tuple[object, object]]:
        """The node's glTF JSON and its two shadow switches."""
        glb = vars(node).get("glb_data", node.props["glb_data"])  # type: ignore[attr-defined]
        shadows = tuple(vars(node).get(key, node.props[key]) for key in ("cast_shadow", "receive_shadow"))  # type: ignore[attr-defined]
        return json.loads(glb[20 : 20 + int.from_bytes(glb[12:16], "little")]), shadows

    for opacity in (0.35, 0.6):
        robot.set_opacity(opacity)
        for node in nodes:
            gltf, shadows = look(node)
            (material,) = gltf["materials"]
            assert node.kind == "GlbHandle" and shadows == (False, False) and material["alphaMode"] == "BLEND"
            assert material["pbrMetallicRoughness"]["baseColorFactor"][3] == pytest.approx(opacity, abs=0.003)
        assert len(look(nodes[0])[0]["images"]) == 1  # the texture stays
        assert "COLOR_0" in look(nodes[1])[0]["meshes"][0]["primitives"][0]["attributes"]  # so do the face colours
    robot.set_opacity(1.0)
    for node in nodes:
        gltf, shadows = look(node)
        assert shadows == (True, True) and all("alphaMode" not in material for material in gltf.get("materials", []))


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


def test_a_triad_is_as_large_as_the_glyph_of_its_joint() -> None:
    """One length for every triad drew a Shadow hand's fingers as a bed of spikes: a triad now follows its joint's
    glyph in the medium skeleton, as large on a hip as a hip's drum and on a wrist as a wrist's (on the G1 the hip's
    is 1.18 times the wrist's). Over the meshes, which reach about twice a glyph's radius and hid the triads, every
    triad is 1.75 times that, and a switch to a skeleton style returns it to its glyph's size."""
    model = br.load(g1_urdf(), root_joint=br.JointFreeFlyer())
    weights = np.zeros((3, model.njoints), dtype=np.float32)
    weights[:, 1] = 1.0
    skin = Skin(0.05 * np.eye(3, dtype=np.float32), np.array([[0, 1, 2]]), weights=weights)  # the meshes style
    robot, server, _ = _g1_visual(frames=True, skin=skin, style="medium")
    moving = [j for j in range(1, model.njoints) if model.nqs[j] == 1 or j == 1]  # the base and the 29 revolute ones
    radii = robot._skeleton("medium").radii[moving]

    def scales() -> np.ndarray:
        return server.scene.nodes["/g1_29dof/g1_29dof/frames/axes"].props["batched_scales"]

    np.testing.assert_allclose(scales(), radii, rtol=1e-6)
    size = dict(zip((model.joint_names[j] for j in moving), scales(), strict=True))
    assert size["left_hip_pitch_joint"] > 1.15 * size["left_wrist_yaw_joint"] > 0.0
    assert size["left_hip_pitch_joint"] == size["right_hip_pitch_joint"]  # mirror sides match
    for style, factor in (("meshes", 1.75), ("minimal", 1.0), ("meshes", 1.75), ("medium", 1.0)):
        robot.set_style(style)
        np.testing.assert_allclose(scales(), factor * radii, rtol=1e-6, err_msg=style)
    robot.set_frames(["left_wrist_yaw_link"])  # a frame takes the glyph of its joint
    np.testing.assert_allclose(scales(), [size["left_wrist_yaw_joint"]])


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


def test_robot_visual_bounds_reach_the_collision_meshes_and_no_further() -> None:
    """A body with only collision meshes is bounded by each mesh's furthest vertices, not by the thinner skeleton:
    along every axis the boxes reach within 1 % of the robot's size of its shapes (a Cassie was cut by the top of
    the view) and never past its shapes and joints (a box round each mesh stood a Cassie 8 cm off the ground)."""
    model = br.load(panda_urdf())
    geometry = br.io.load_geometry(panda_urdf(), model)
    q = torch.stack([model.random_configuration(generator=torch.Generator().manual_seed(i)) for i in range(4)])
    robot = RobotVisual(model, q, geometry=geometry, root=panda_urdf())
    poses = br.forward_kinematics(model, q).joint_poses.double()
    specs = load_meshes(geometry, root=panda_urdf(), collision=True)  # the nine meshes and the finger boxes
    for step in range(4):
        lows, highs = robot.bounds(step)
        world = np.concatenate([
            br.lie.se3.act(
                compose(poses[step, spec.body], torch.from_numpy(spec.origin).double()),
                torch.from_numpy(spec.vertices).double(),
            ).numpy()
            for spec in specs
        ] + [poses[step, 1:, :3].numpy()])  # fmt: skip
        assert (lows.min(axis=0) <= world.min(axis=0) + 0.01 * 1.044).all()  # of the arm's size
        assert (highs.max(axis=0) >= world.max(axis=0) - 0.01 * 1.044).all()
        assert (lows.min(axis=0) >= world.min(axis=0) - 1e-5).all() and (
            highs.max(axis=0) <= world.max(axis=0) + 1e-5
        ).all()


def test_robot_visual_bounds_hold_a_skin_that_reaches_past_the_joints_also_after_update() -> None:
    """A sail of a skin on the Panda's hand reaches 0.6 m past every joint: the boxes per bone hold each skinned
    vertex at each step, in a skeleton style too, and follow ``update``."""
    model = br.load(panda_urdf())
    generator = torch.Generator().manual_seed(0)
    q = torch.stack([model.random_configuration(generator=generator) for _ in range(2)])
    hand = model.body_id("panda_hand")
    bind = br.forward_kinematics(model, model.q_neutral).joint_poses[hand]
    sail = np.array([[0.0, 0.0, 0.0], [0.6, 0.0, 0.0], [0.0, 0.0, 0.6]], dtype=np.float32)
    weights = np.zeros((3, model.njoints), dtype=np.float32)
    weights[:, hand] = 1.0
    skin = Skin(bind[:3].numpy() + sail, np.array([[0, 1, 2]]), weights=weights)
    robot = RobotVisual(model, q, skin=skin, style="medium")
    for clip in (q, q.flip(0)):
        robot.update(clip)
        posed = br.forward_kinematics(model, clip).joint_poses[:, hand]  # (2, 7)
        turn = so3.compose(posed[:, None, 3:], so3.inverse(bind[3:]))  # the sail rides on the hand
        vertices = (so3.act(turn, torch.from_numpy(sail).to(posed)) + posed[:, None, :3]).numpy()  # (2, 3, 3)
        for step in range(2):
            lows, highs = robot.bounds(step)  # (N, 3) each
            inside = (lows[:, None] <= vertices[step] + 1e-5) & (vertices[step] <= highs[:, None] + 1e-5)
            assert inside.all(axis=-1).any(axis=0).all()  # each vertex in a box


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
    before = server.scene.nodes[f"{path}/medium/cylinders"].props["batched_colors"].copy()
    red, blue = (0.8, 0.1, 0.1), (0.1, 0.2, 0.9)
    viewer.set_theme(Theme(hues=(red,), collision=blue))
    after = server.scene.nodes[f"{path}/medium/cylinders"].batched_colors
    assert not np.array_equal(before, after) and (after[:, 0] >= after[:, 2]).all()  # shades of red
    spheres = server.scene.nodes[f"{path}/collision/spheres"]
    assert not spheres.removed and (spheres.props["batched_colors"] == linear_uint8(blue)).all()
    assert robot.style == "medium"


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
    robot = viewer.add(RobotVisual(model, geometry=geometry, root=panda_urdf(), style="medium", collision=True))
    robot.set_style("minimal")
    viewer.set_theme(DARK)
    assert len(reads) == len(set(reads)) == 9


@pytest.mark.parametrize("content", [None, b"solid empty\nendsolid empty\n"], ids=["missing", "empty"])
def test_a_collision_mesh_that_does_not_load_raises_for_the_overlay_and_the_skeleton_sizes_its_body_without_it(
    tmp_path: Path, content: bytes | None
) -> None:
    """The collision overlay draws the file: one that is missing or holds no mesh raises ``ModelError``, naming the
    body and the file. A skeleton only sizes the body by it: it gives the same words as a warning and sizes the body
    from its inertia, as for a body without a collision shape."""
    if content is not None:
        (tmp_path / "part.stl").write_bytes(content)
    inertial = (
        '<inertial><mass value="2"/><origin xyz="0 0 0.1"/>'
        '<inertia ixx="0.02" iyy="0.02" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial>'
    )
    collision = '<collision><geometry><mesh filename="part.stl"/></geometry></collision>'
    urdf = tmp_path / "r.urdf"
    urdf.write_text(
        f'<robot name="r"><link name="base">{inertial}{collision}</link><link name="arm">{inertial}</link>'
        '<joint name="hinge" type="revolute"><parent link="base"/><child link="arm"/><origin xyz="0 0 0.3"/>'
        '<axis xyz="0 1 0"/><limit lower="-1" upper="1" effort="10" velocity="1"/></joint></robot>'
    )
    model = br.load(urdf)
    robot = RobotVisual(model, geometry=br.io.load_geometry(urdf, model), root=urdf, style="medium")
    words = r"body \d+: mesh 'part.stl' (is not a file|does not load|has no faces)"
    with pytest.warns(UserWarning, match=words + ".*the skeleton sizes body"):
        Viewer(server=FakeServer(), ground=False).add(robot)
    inertia = build_skeleton(model, MEDIUM)  # no geometry: every body from its inertia
    for mesh in ("cylinders", "balls", "boxes"):
        for array in ("local", "scale"):
            np.testing.assert_array_equal(
                getattr(robot._skeleton("medium").blocks[mesh], array), getattr(inertia.blocks[mesh], array)
            )
    with pytest.raises(ModelError, match=words):
        robot.set_collision(visible=True)


def test_two_robots_of_one_model_need_a_name_each() -> None:
    model = br.load(panda_urdf())
    viewer = Viewer(server=FakeServer(), ground=False)
    viewer.add(RobotVisual(model))
    with pytest.raises(ArgumentError, match="already has a layer named 'panda'"):
        viewer.add(RobotVisual(model))
    viewer.add(RobotVisual(model, name="second"))
    assert [layer.name for layer in viewer.layers] == ["panda", "second"]


@pytest.mark.parametrize("style", ["medium", "minimal"])
def test_joint_values_colour_the_joint_glyphs_with_one_more_write_and_no_new_node(style: str) -> None:
    """A step with values writes the colours of the node that holds the glyphs, once, beside what it writes without
    them, and makes no node; only the knee's glyph changes colour, to the ramp's ends (clamped), its drum face in the
    face shade. Taking the values away gives the hue's colours back; values of another shape are refused."""
    model = br.load(g1_urdf(), root_joint=br.JointFreeFlyer())
    knee = model.joint_names.index("left_knee_joint")
    values = np.full((2, model.njoints), np.nan, dtype=np.float32)  # NaN: no value
    values[:, knee] = (-1.0, 5.0)  # below and above the range
    q = model.q_neutral.repeat(2, 1)
    plain, plain_server, plain_viewer = _g1_visual(q, style=style)
    robot, server, viewer = _g1_visual(q, style=style, joint_values=values, value_range=(0.0, 2.0))
    assert set(_drawn(server)) == set(_drawn(plain_server))
    for each in (server, plain_server):
        each.scene.log.clear()
    viewer.show(1)
    plain_viewer.show(1)
    node = f"/g1_29dof/g1_29dof/{style}/{'balls' if style == 'minimal' else 'cylinders'}"
    assert sorted(set(server.scene.log) - set(plain_server.scene.log)) == [(node, "batched_colors")]
    assert len(server.scene.log) == len(plain_server.scene.log) + 1

    blocks = robot._skeleton(style).blocks[node.rsplit("/", 1)[-1]]
    knees = [i for i, owner in enumerate(blocks.owner) if owner == knee and blocks.part[i] in ("drum", "face", "ball")]
    painted, hue = server.scene.nodes[node].batched_colors, plain_server.scene.nodes[node].props["batched_colors"]
    assert np.flatnonzero((painted != hue).any(axis=1)).tolist() == knees
    high = {"drum": LIGHT.ramp[1], "ball": LIGHT.ramp[1], "face": LIGHT.shade(LIGHT.ramp[1], "face")}
    for i in knees:
        np.testing.assert_array_equal(painted[i], linear_uint8(high[blocks.part[i]]))
    viewer.show(0)
    assert (server.scene.nodes[node].batched_colors[knees[0]] == linear_uint8(LIGHT.ramp[0])).all()
    robot.set_joint_values(values[1])  # one row holds for every step of the clip
    viewer.show(0)
    np.testing.assert_array_equal(server.scene.nodes[node].batched_colors, painted)
    assert robot.joint_values is not None and robot.joint_values.shape == (2, model.njoints)
    assert RobotVisual(model, q, joint_values=torch.from_numpy(values[0])).joint_values.shape == (2, model.njoints)
    robot.set_joint_values(None)
    np.testing.assert_array_equal(server.scene.nodes[node].batched_colors, hue)

    for wrong in (np.zeros((3, model.njoints)), np.zeros(model.nv), torch.zeros(2, model.njoints, 1)):
        with pytest.raises(ShapeError, match="joint_values"):
            robot.set_joint_values(wrong)
    for span in ((1.0, 1.0), (0.0, float("nan"))):
        with pytest.raises(ArgumentError, match="value_range"):
            robot.set_joint_values(values, value_range=span)
    with pytest.raises(ShapeError, match="value_range"):
        robot.set_joint_values(values, value_range=(0.0, 1.0, 2.0))  # type: ignore[arg-type]


@pytest.mark.parametrize("growth", [2.0, 1.4])
def test_joint_values_as_size_grow_a_glyph_to_the_recipe_growth_at_the_top_of_the_range(growth: float) -> None:
    """With values shown as size, the knee's glyph keeps its size at the low end of the range and is the recipe's
    ``growth`` times it at the high end and beyond (clamped); the other glyphs keep theirs. A growth below 1 is
    refused."""
    model = br.load(g1_urdf(), root_joint=br.JointFreeFlyer())
    knee = model.joint_names.index("left_knee_joint")
    values = np.full((2, model.njoints), np.nan, dtype=np.float32)
    values[:, knee] = (0.0, 5.0)
    q = model.q_neutral.repeat(2, 1)
    options = {"joint_values": values, "value_range": (0.0, 2.0), "values_as": "size"}
    robot, server, viewer = _g1_visual(q, medium=replace(MEDIUM, growth=growth), **options)
    node = "/g1_29dof/g1_29dof/medium/cylinders"
    viewer.show(0)
    low = np.array(server.scene.nodes[node].batched_scales)
    viewer.show(1)
    blocks = robot._skeleton("medium").blocks["cylinders"]
    knees = [i for i, owner in enumerate(blocks.owner) if owner == knee and blocks.part[i] in ("drum", "face")]
    assert knees
    expected = low.copy()
    expected[knees] *= growth
    np.testing.assert_allclose(server.scene.nodes[node].batched_scales, expected, rtol=1e-6)
    with pytest.raises(ArgumentError, match="growth"):
        replace(MEDIUM, growth=0.5)


def test_joint_values_as_size_with_a_growth_pair_shrink_a_glyph_at_the_low_end_and_keep_the_rods() -> None:
    """``growth=(low, high)``: the knee's glyph is ``low`` times its size at the low end of the range and ``high``
    times it at the top; every other block, the rods beside it included, keeps its size without values. A pair
    with low above high is refused."""
    model = br.load(g1_urdf(), root_joint=br.JointFreeFlyer())
    knee = model.joint_names.index("left_knee_joint")
    values = np.full((2, model.njoints), np.nan, dtype=np.float32)
    values[:, knee] = (0.0, 2.0)
    options = {"joint_values": values, "value_range": (0.0, 2.0), "values_as": "size"}
    robot, server, viewer = _g1_visual(
        model.q_neutral.repeat(2, 1), medium=replace(MEDIUM, growth=(0.5, 1.4)), **options
    )
    skeleton = robot._skeleton("medium")
    plain = skeleton.place(robot._poses[0])["cylinders"][1].astype(np.float32)
    knees = [
        i
        for i, owner in enumerate(skeleton.blocks["cylinders"].owner)
        if owner == knee and skeleton.blocks["cylinders"].part[i] in ("drum", "face")
    ]
    assert knees and "rod" in skeleton.blocks["cylinders"].part
    for step, factor in ((0, 0.5), (1, 1.4)):
        viewer.show(step)
        expected = plain.copy()
        expected[knees] *= factor
        np.testing.assert_allclose(
            server.scene.nodes["/g1_29dof/g1_29dof/medium/cylinders"].batched_scales, expected, rtol=1e-6
        )
    with pytest.raises(ArgumentError, match="growth"):
        replace(MEDIUM, growth=(1.5, 0.5))


@pytest.mark.parametrize("style", ["minimal", "medium"])
def test_robot_visual_stick_figure_writes_lengths_to_the_stick_node_only(style: str) -> None:
    _, server, viewer = _g1_visual(style=style)
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
