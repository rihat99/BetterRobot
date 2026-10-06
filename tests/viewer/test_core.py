"""Viewer core on a fake server: frame pushes, visibility, groups, timeline."""

from __future__ import annotations

import time

import numpy as np
import pytest

from better_robot.viewer._math import quat_z_to, unit_cone, unit_cylinder, unit_sphere
from better_robot.viewer.layers import ArrowsLayer, FramesLayer, GroundLayer, PointsLayer, SkeletonLayer
from better_robot.viewer.timeline import Timeline
from better_robot.viewer.viewer import Viewer

from .fake_server import FakeServer


def _skeleton(num_frames: int = 10, name: str = "skel") -> SkeletonLayer:
    joints = np.random.default_rng(0).normal(size=(num_frames, 4, 3))
    return SkeletonLayer(name, joints, parents=[-1, 0, 1, 1])


def test_show_pushes_each_visible_layer_once_per_frame_inside_one_atomic_block() -> None:
    server = FakeServer()
    viewer = Viewer(server=server)
    viewer.add(_skeleton(10))
    viewer.add(GroundLayer())
    arrows = viewer.add(ArrowsLayer("f", np.zeros((5, 2, 3)), np.ones((5, 2, 3))), group="forces")
    server.scene.log.clear()
    blocks = server.atomic_blocks
    viewer.show(3)
    assert server.atomic_blocks == blocks + 1
    assert viewer.timeline.frame == 3
    pushed = {node for node, event in server.scene.log if event != "add"}
    assert pushed == {"/scene/skel/joints", "/scene/skel/bones", "/forces/f/shafts", "/forces/f/heads"}
    # The arrows layer has 5 frames: frame 7 clamps to its last frame and the static grid is never touched.
    server.scene.log.clear()
    viewer.timeline.frame = 7
    assert ("/forces/f/shafts", "points") in server.scene.log
    assert not any(node.startswith("/scene/ground") for node, _ in server.scene.log)
    server.scene.log.clear()
    viewer.timeline.frame = 8  # arrows already show their last frame: skipped
    assert not any(node.startswith("/forces") for node, _ in server.scene.log)
    assert ("/scene/skel/joints", "batched_positions") in server.scene.log
    assert arrows.num_frames == 5 and viewer.timeline.num_frames == 10


def test_hidden_layers_are_skipped_and_catch_up_when_shown_again() -> None:
    server = FakeServer()
    viewer = Viewer(server=server)
    layer = viewer.add(_skeleton(10))
    checkbox = server.gui.control("skel")
    checkbox.value = False
    checkbox.fire()
    assert server.scene.nodes["/scene/skel/joints"].visible is False
    server.scene.log.clear()
    viewer.timeline.frame = 6
    assert not any(event != "add" for _, event in server.scene.log)
    viewer.set_visible(layer, True)
    assert ("/scene/skel/joints", "batched_positions") in server.scene.log
    np.testing.assert_array_equal(server.scene.nodes["/scene/skel/joints"].batched_positions, layer.joints[6])


def test_remove_group_disposes_layers_and_the_group_node() -> None:
    server = FakeServer()
    viewer = Viewer(server=server)
    viewer.add(_skeleton(), group="ref")
    viewer.add(PointsLayer("pts", np.zeros((3, 3))), group="ref")
    viewer.add(_skeleton(name="keep"))
    viewer.remove_group("ref")
    assert server.scene.nodes["/ref"].removed
    assert server.scene.nodes["/ref/skel/joints"].removed and server.scene.nodes["/ref/pts/points"].removed
    assert not server.scene.nodes["/scene/keep/joints"].removed
    assert [layer.name for layer in viewer.layers] == ["keep"]
    with pytest.raises(ValueError):
        viewer.add(_skeleton(name="keep"))


def test_timeline_clamps_loops_and_drives_the_gui_slider() -> None:
    timeline = Timeline(5, fps=100.0)
    seen: list[int] = []
    timeline.on_frame(seen.append)
    gui = FakeServer().gui
    timeline.build_gui(gui)
    slider = gui.control("Frame")
    timeline.frame = 99
    assert timeline.frame == 4 and slider.value == 4
    timeline.frame = -3
    assert timeline.frame == 0
    slider.value = 2
    slider.fire()
    assert seen == [4, 0, 2]
    timeline.extend(8)
    assert slider.max == 7
    timeline.extend(3)
    assert timeline.num_frames == 8
    timeline.loop = False
    timeline.frame = 7
    timeline.play()
    deadline = time.perf_counter() + 2.0
    while timeline.playing and time.perf_counter() < deadline:
        time.sleep(0.01)
    assert not timeline.playing and timeline.frame == 7 and seen[-1] == 7 and 0 in seen[-9:-1]

    timeline.loop = True
    timeline.play()
    time.sleep(0.15)
    timeline.pause()
    assert not timeline.playing
    assert any(frame < timeline.frame or frame == 0 for frame in seen[-8:])
    assert gui.control("Play").label == "Play"


def test_frames_points_and_ground_layers_write_the_expected_arrays() -> None:
    scene = FakeServer().scene
    poses = np.zeros((2, 3, 7), dtype=np.float32)
    poses[..., 6] = 1.0
    poses[1, :, 0] = 1.0
    frames = FramesLayer("axes", poses)
    frames.build(scene, "/g/axes")
    frames.set_frame(1)
    node = scene.nodes["/g/axes/axes"]
    np.testing.assert_array_equal(node.batched_positions[:, 0], 1.0)
    np.testing.assert_array_equal(node.batched_wxyzs, np.tile([1.0, 0.0, 0.0, 0.0], (3, 1)))

    points = PointsLayer("pts", np.ones((4, 3)), colors=(1.0, 0.0, 0.0))
    assert points.num_frames == 1
    points.build(scene, "/g/pts")
    assert scene.nodes["/g/pts/points"].props["colors"].tolist() == [[255, 0, 0]] * 4

    GroundLayer(height=0.5).build(scene, "/g/ground")
    assert scene.nodes["/g/ground/grid"].props["position"] == (0.0, 0.0, 0.5)


def test_skeleton_bones_join_parent_and_child_and_arrows_hide_zero_vectors() -> None:
    scene = FakeServer().scene
    joints = np.array([[[0.0, 0.0, 0.0], [0.0, 0.0, 2.0], [1.0, 0.0, 2.0]]], dtype=np.float32)
    skeleton = SkeletonLayer("s", joints, parents=[-1, 0, 1], radius=0.1)
    skeleton.build(scene, "/g/s")
    bones = scene.nodes["/g/s/bones"]
    np.testing.assert_allclose(bones.props["batched_positions"], [[0.0, 0.0, 1.0], [0.5, 0.0, 2.0]])
    np.testing.assert_allclose(bones.props["batched_scales"], [[0.06, 0.06, 2.0], [0.06, 0.06, 1.0]], atol=1e-6)
    np.testing.assert_allclose(bones.props["batched_wxyzs"][0], [1.0, 0.0, 0.0, 0.0])
    half = np.sqrt(0.5)
    np.testing.assert_allclose(bones.props["batched_wxyzs"][1], [half, 0.0, half, 0.0], atol=1e-6)

    arrows = ArrowsLayer("a", np.zeros((1, 2, 3)), [[[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]], scale=2.0, head_size=0.1)
    arrows.build(scene, "/g/a")
    heads = scene.nodes["/g/a/heads"]
    np.testing.assert_allclose(heads.props["batched_scales"][0], [0.05, 0.05, 0.1])
    np.testing.assert_allclose(heads.props["batched_positions"][0], [0.0, 0.0, 1.9])
    np.testing.assert_allclose(heads.props["batched_scales"][1], 0.0)
    np.testing.assert_allclose(scene.nodes["/g/a/shafts"].props["args"][0][0, 1], [0.0, 0.0, 2.0])


def test_quaternion_z_to_direction_and_closed_unit_meshes() -> None:
    directions = np.random.default_rng(1).normal(size=(50, 3))
    directions[0] = [0.0, 0.0, -1.0]
    directions[1] = 0.0
    q = quat_z_to(directions)
    w, x, y, z = q.T
    rotated_z = np.stack([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], axis=-1)
    unit = directions / np.maximum(np.linalg.norm(directions, axis=-1, keepdims=True), 1e-9)
    np.testing.assert_allclose(rotated_z[2:], unit[2:], atol=1e-5)
    np.testing.assert_allclose(rotated_z[0], [0.0, 0.0, -1.0], atol=1e-6)
    np.testing.assert_allclose(rotated_z[1], [0.0, 0.0, 1.0], atol=1e-6)
    for vertices, faces in (unit_sphere(), unit_cylinder(), unit_cone()):
        edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
        _, counts = np.unique(edges, axis=0, return_counts=True)
        assert (counts == 2).all() and faces.max() == len(vertices) - 1
