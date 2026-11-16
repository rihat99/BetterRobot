"""Viewer core on a fake server: frame pushes, visibility, groups, timeline."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pytest

from better_robot.viewer._lie import act, from_z, to_wxyz
from better_robot.viewer.layer import Layer
from better_robot.viewer._primitives import unit_cone, unit_cylinder, unit_sphere
from better_robot.viewer.layers import ArrowsLayer, FramesLayer, GroundLayer, PointsLayer
from better_robot.viewer.layers.image import ImagePanel
from better_robot.viewer.layers.mesh import MeshLayer, MeshSpec, RigidMeshLayer, SpheresLayer
from better_robot.viewer.layers.skinned import SkinnedMeshLayer
from better_robot.viewer.theme import Theme
from better_robot.viewer.timeline import Timeline
from better_robot.viewer.viewer import Viewer

from better_robot.exceptions import ArgumentError, ShapeError, UnknownNameError
from better_robot.testing import FakeServer
from better_robot.testing.fake_server import FakeClient


def _skeleton(nsteps: int = 10, name: str = "skel") -> SpheresLayer:
    joints = np.random.default_rng(0).normal(size=(nsteps, 4, 3))
    return SpheresLayer(name, joints, radii=np.full(4, 0.05))


def test_show_pushes_each_visible_layer_once_per_frame_inside_one_atomic_block() -> None:
    server = FakeServer()
    viewer = Viewer(server=server)
    viewer.add([_skeleton(10)])  # beside the ground the viewer adds by itself
    arrows = ArrowsLayer("f", np.zeros((5, 2, 3)), vectors=np.ones((5, 2, 3)))
    viewer.add([arrows], tags="forces")
    server.scene.log.clear()
    blocks = server.atomic_blocks
    viewer.show(3)
    assert server.atomic_blocks == blocks + 1
    assert viewer.timeline.step == 3
    pushed = {node for node, event in server.scene.log if event != "add"}
    assert pushed == {"/skel/skel/spheres", "/forces/f/shafts", "/forces/f/heads"}
    # The arrows layer has 5 frames: frame 7 clamps to its last frame and the static grid is never touched.
    server.scene.log.clear()
    viewer.timeline.step = 7
    assert ("/forces/f/shafts", "points") in server.scene.log
    assert not any(node.startswith("/ground") for node, _ in server.scene.log)
    server.scene.log.clear()
    viewer.timeline.step = 8  # arrows already show their last frame: skipped
    assert not any(node.startswith("/forces") for node, _ in server.scene.log)
    assert ("/skel/skel/spheres", "batched_positions") in server.scene.log
    assert arrows.nsteps == 5 and viewer.timeline.nsteps == 10


def test_hidden_layers_are_skipped_and_catch_up_when_shown_again() -> None:
    server = FakeServer()
    viewer = Viewer(server=server)
    layer = _skeleton(10)
    viewer.add([layer])
    viewer.add(PointsLayer("cloud", np.zeros((1, 3))))  # a second tag brings the Layers checkboxes
    checkbox = server.gui.control("skel")
    checkbox.value = False
    checkbox.fire()
    assert server.scene.nodes["/skel/skel/spheres"].visible is False
    server.scene.log.clear()
    viewer.timeline.step = 6
    assert not any(event != "add" for _, event in server.scene.log)
    viewer.set_visible(layer, True)
    assert ("/skel/skel/spheres", "batched_positions") in server.scene.log
    np.testing.assert_array_equal(server.scene.nodes["/skel/skel/spheres"].batched_positions, layer.centers[6])


def test_remove_a_tag_disposes_its_layers_and_the_tag_node() -> None:
    server = FakeServer()
    viewer = Viewer(server=server, ground=False)
    viewer.add([_skeleton()], tags="ref")
    viewer.add([PointsLayer("pts", np.zeros((3, 3)))], tags="ref")
    viewer.add([_skeleton(name="keep")])
    viewer.remove("ref")
    assert server.scene.nodes["/ref"].removed
    assert server.scene.nodes["/ref/skel/spheres"].removed and server.scene.nodes["/ref/pts/points"].removed
    assert not server.scene.nodes["/keep/keep/spheres"].removed
    assert [layer.name for layer in viewer.layers] == ["keep"]
    with pytest.raises(ArgumentError, match="already has a layer"):
        viewer.add([_skeleton(name="keep")])


def test_set_offset_moves_a_tag_after_it_was_added() -> None:
    server = FakeServer()
    viewer = Viewer(server=server)
    viewer.add([PointsLayer("p", np.zeros((1, 3)))], tags="robot", offset=(0.0, 2.0, 0.0))
    viewer.set_offset("robot", (0.0, 3.0, 0.0))
    assert server.scene.nodes["/robot"].position == (0.0, 3.0, 0.0)
    np.testing.assert_allclose(viewer.aim()[1], [0.0, 3.0, 0.0])  # record and aim frame the moved tag
    with pytest.raises(UnknownNameError, match="unknown tag 'missing'"):
        viewer.set_offset("missing", (0.0, 0.0, 0.0))


def test_timeline_clamps_loops_and_drives_the_gui_slider() -> None:
    timeline = Timeline(5, fps=100.0)
    seen: list[int] = []
    timeline.on_step(seen.append)
    gui = FakeServer().gui
    timeline.build_gui(gui)
    slider = gui.control("Step")
    timeline.step = 99
    assert timeline.step == 4 and slider.value == 4
    timeline.step = -3
    assert timeline.step == 0
    slider.value = 2
    slider.fire()
    assert seen == [4, 0, 2]
    timeline.extend(8)
    assert slider.max == 7
    timeline.extend(3)
    assert timeline.nsteps == 8
    timeline.loop = False
    timeline.step = 7
    timeline.play()
    deadline = time.perf_counter() + 2.0
    while timeline.playing and time.perf_counter() < deadline:
        time.sleep(0.01)
    assert not timeline.playing and timeline.step == 7 and seen[-1] == 7 and 0 in seen[-9:-1]

    timeline.loop = True
    timeline.play()
    time.sleep(0.15)
    timeline.pause()
    assert not timeline.playing
    assert any(step < timeline.step or step == 0 for step in seen[-8:])
    assert gui.control("Play").label == "Play"


def test_frames_points_and_ground_layers_write_the_expected_arrays() -> None:
    scene = FakeServer().scene
    poses = np.zeros((2, 3, 7), dtype=np.float32)
    poses[..., 6] = 1.0
    poses[1, :, 0] = 1.0
    frames = FramesLayer("axes", poses)
    frames.build(scene, "/g/axes")
    frames.set_step(1)
    node = scene.nodes["/g/axes/axes"]
    np.testing.assert_array_equal(node.batched_positions[:, 0], 1.0)
    np.testing.assert_array_equal(node.batched_wxyzs, np.tile([1.0, 0.0, 0.0, 0.0], (3, 1)))

    # Zero frames: viser's batched-axes client crashes on an empty instance list, so nothing is added.
    empty = FramesLayer("none", np.zeros((1, 0, 7), dtype=np.float32))
    empty.build(scene, "/g/none")
    empty.set_step(0)
    assert "/g/none/axes" not in scene.nodes

    points = PointsLayer("pts", np.ones((4, 3)), colors=(1.0, 0.0, 0.0))
    assert points.nsteps == 1
    points.build(scene, "/g/pts")
    assert scene.nodes["/g/pts/points"].props["colors"].tolist() == [[255, 0, 0]] * 4

    GroundLayer(height=0.5).build(scene, "/g/ground")
    assert scene.nodes["/g/ground/grid"].props["position"] == (0.0, 0.0, 0.5)


def test_arrows_hide_zero_vectors() -> None:
    scene = FakeServer().scene
    arrows = ArrowsLayer(
        "a", np.zeros((1, 2, 3)), vectors=[[[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]], scale=2.0, head_size=0.1
    )
    arrows.build(scene, "/g/a")
    heads = scene.nodes["/g/a/heads"]
    np.testing.assert_allclose(heads.props["batched_scales"][0], [0.05, 0.05, 0.1])
    np.testing.assert_allclose(heads.props["batched_positions"][0], [0.0, 0.0, 1.9])
    np.testing.assert_allclose(heads.props["batched_scales"][1], 0.0)
    np.testing.assert_allclose(scene.nodes["/g/a/shafts"].props["points"][0, 1], [0.0, 0.0, 2.0])


def test_solid_arrows_keep_a_fixed_head_shrink_short_ones_whole_and_hide_zero_and_invalid_vectors() -> None:
    server = FakeServer()
    viewer = Viewer(server=server)
    vectors = np.array([[[0.0, 0.0, 2.0], [0.05, 0.0, 0.0], [0.0, 0.0, 0.0]]] * 2)
    colours = np.zeros((2, 3, 3))
    colours[1] = (0.0, 0.0, 1.0)
    keep = np.ones((2, 3), bool)
    keep[1, 0] = False
    viewer.add(
        [
            ArrowsLayer(
                "f",
                np.zeros((2, 3, 3)),
                vectors=vectors,
                colors=colours,
                head_size=0.1,
                shaft_radius=0.02,
                head_radius=0.04,
            )
        ],
        valid=keep,
    )
    shafts, heads = server.scene.nodes["/f/f/shafts"], server.scene.nodes["/f/f/heads"]
    assert shafts.kind == "BatchedMeshHandle"
    viewer.show(0)
    # Long arrow: shaft up to the head's base, head of full size. Short arrow (0.05 m, under head_size / 0.35): the
    # long one scaled by 0.05 * 0.35 / 0.1 = 0.175, head 35 % of it, so still a shaft and a head.
    np.testing.assert_allclose(shafts.batched_positions[:2], [[0.0, 0.0, 0.95], [0.01625, 0.0, 0.0]], atol=1e-6)
    np.testing.assert_allclose(shafts.batched_scales[:2], [[0.02, 0.02, 1.9], [0.0035, 0.0035, 0.0325]], atol=1e-6)
    np.testing.assert_allclose(heads.batched_positions[:2], [[0.0, 0.0, 1.9], [0.0325, 0.0, 0.0]], atol=1e-6)
    np.testing.assert_allclose(heads.batched_scales[:2], [[0.04, 0.04, 0.1], [0.007, 0.007, 0.0175]], atol=1e-6)
    np.testing.assert_allclose(heads.batched_scales[2], 0.0)
    np.testing.assert_allclose(shafts.batched_scales[2], 0.0)
    viewer.show(1)  # the long arrow is invalid here; per-step colours reach the shafts too
    np.testing.assert_allclose(shafts.batched_scales[0], 0.0)
    np.testing.assert_allclose(heads.batched_scales[0], 0.0)
    np.testing.assert_array_equal(shafts.batched_colors, np.full((3, 3), [0, 0, 255]))


@pytest.mark.parametrize("shaft_radius", [None, 0.01], ids=["line", "solid"])
def test_arrows_draw_nothing_up_to_min_length(shaft_radius: float | None) -> None:
    """Zero and float-noise vectors (default ``min_length``), or ones up to a given ``min_length``, draw no head and no
    shaft: no dot from a line's round caps, no lone solid head."""
    server = FakeServer()
    viewer = Viewer(server=server)
    lengths = np.array([0.0, 1e-7, 5e-3, 0.5], dtype=np.float32)
    vectors = np.zeros((1, 4, 3), dtype=np.float32)
    vectors[..., 2] = lengths
    for name, min_length in (("default", {}), ("coarse", {"min_length": 0.01})):
        viewer.add(
            [
                ArrowsLayer(
                    name, np.ones((1, 4, 3)), vectors=vectors, head_size=0.03, shaft_radius=shaft_radius, **min_length
                )
            ]
        )
    viewer.show(0)
    for name, shown in (("default", [False, False, True, True]), ("coarse", [False, False, False, True])):
        heads, shafts = server.scene.nodes[f"/{name}/{name}/heads"], server.scene.nodes[f"/{name}/{name}/shafts"]
        np.testing.assert_array_equal(heads.batched_scales.any(axis=-1), shown)
        if shaft_radius is None:  # a hidden line is a zero-length segment
            np.testing.assert_array_equal((shafts.points[:, 1] != shafts.points[:, 0]).any(axis=-1), shown)
        else:  # a short solid arrow keeps its shaft
            np.testing.assert_array_equal(shafts.batched_scales.any(axis=-1), shown)


def test_spheres_and_arrows_take_per_step_colours() -> None:
    server = FakeServer()
    viewer = Viewer(server=server)
    colours = np.zeros((3, 2, 3))
    colours[1, 0] = (1.0, 0.0, 0.0)  # contact 0 turns red at step 1
    viewer.add([SpheresLayer("contacts", np.zeros((3, 2, 3)), radii=np.full(2, 0.05), colors=colours)])
    viewer.add([ArrowsLayer("forces", np.zeros((3, 2, 3)), vectors=np.ones((3, 2, 3)), colors=colours)])
    spheres, heads = server.scene.nodes["/contacts/contacts/spheres"], server.scene.nodes["/forces/forces/heads"]
    shafts = server.scene.nodes["/forces/forces/shafts"]
    viewer.show(1)
    np.testing.assert_array_equal(spheres.batched_colors, [[255, 0, 0], [0, 0, 0]])
    np.testing.assert_array_equal(heads.batched_colors, [[255, 0, 0], [0, 0, 0]])
    np.testing.assert_array_equal(shafts.colors[0], [[255, 0, 0], [255, 0, 0]])
    viewer.show(2)
    assert not spheres.batched_colors.any()
    static = SpheresLayer("static", np.zeros((3, 2, 3)), radii=np.full(2, 0.05), colors=(0.0, 1.0, 0.0))
    viewer.add([static])
    assert static._colors.shape == (2, 3)  # one colour stays static: never rewritten per step
    with pytest.raises(ShapeError, match="per-step colors"):
        ArrowsLayer("bad", np.zeros((3, 2, 3)), vectors=np.ones((3, 2, 3)), colors=np.zeros((2, 2, 3)))


def test_quaternion_z_to_direction_and_closed_unit_meshes() -> None:
    directions = np.random.default_rng(1).normal(size=(50, 3))
    directions[0] = [0.0, 0.0, -1.0]
    directions[1] = 0.0
    q = to_wxyz(from_z(directions))
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


def test_plot_panel_lives_in_the_gui_and_marks_the_current_frame() -> None:
    server = FakeServer()
    viewer = Viewer(server=server)
    panel = viewer.add_plot("tau", np.random.default_rng(0).normal(size=(20, 3)), labels=["a", "b", "c"])
    plot = next(control for control in server.gui.controls if control.kind == "GuiUplotHandle")
    assert len(plot.props["data"]) == 4 and plot.props["data"][0][-1] == pytest.approx(19 / 30)
    viewer.show(7)
    assert plot.series[1]["points"]["filter"] == (7,) and plot.series[1]["label"] == "a"
    assert panel.nsteps == 20 and viewer.timeline.nsteps == 20
    viewer.remove("plots")
    assert plot.removed


def test_image_panel_shows_the_image_of_the_current_step_eager_or_lazy() -> None:
    server = FakeServer()
    viewer = Viewer(server=server)
    frames = np.arange(4 * 2 * 3 * 3, dtype=np.uint8).reshape(4, 2, 3, 3)
    viewer.add([ImagePanel("video", frames)], tags="video")
    asked = []
    lazy = ImagePanel("lazy", lambda step: asked.append(step) or frames[step], nsteps=4)
    viewer.add([lazy], tags="video", times=np.arange(4) / 10.0)  # a 10 Hz video on the 30 fps timeline
    eager_image, lazy_image = (control for control in server.gui.controls if control.kind == "GuiImageHandle")
    viewer.show(3)  # 0.1 s: the lazy panel's sample 1
    np.testing.assert_array_equal(eager_image.image, frames[3])
    np.testing.assert_array_equal(lazy_image.image, frames[1])
    assert asked == [0, 0, 1]  # built at sample 0, pushed at 0, then only the shown sample is decoded
    viewer.set_visible(lazy, False)
    assert lazy_image.visible is False
    with pytest.raises(ArgumentError, match="nsteps"):
        ImagePanel("no_length", lambda step: frames[0])
    viewer.remove("video")
    assert eager_image.removed and lazy_image.removed


def test_ground_plane_follows_the_theme_up_axis_unless_given() -> None:
    viewer = Viewer(server=FakeServer(), theme=Theme(up="+y"), ground=False)
    ground, explicit = GroundLayer(), GroundLayer("explicit", plane="xy")
    viewer.add([ground, explicit])
    assert ground.plane == "xz" and explicit.plane == "xy"
    assert GroundLayer().bounds(0) is None  # the grid never drives the camera's framing


def test_ground_tilts_to_a_normal_through_an_origin_and_fills() -> None:
    scene = FakeServer().scene
    normal = np.array([0.0, 1.0, 1.0]) / np.sqrt(2.0)
    GroundLayer(origin=(1.0, 2.0, 3.0), normal=(0.0, 1.0, 1.0), height=0.5, fill=0.4).build(scene, "/g/floor")
    grid = scene.nodes["/g/floor/grid"].props
    w, x, y, z = grid["wxyz"]
    np.testing.assert_allclose(act(np.array([x, y, z, w]), np.array([0.0, 0.0, 1.0])), normal, atol=1e-6)
    np.testing.assert_allclose(grid["position"], np.array([1.0, 2.0, 3.0]) + 0.5 * normal)
    assert grid["plane"] == "xy" and grid["plane_opacity"] == 0.4
    GroundLayer(plane="xz").build(scene, "/g/flat")
    assert scene.nodes["/g/flat/grid"].props["plane_opacity"] == 0.0  # lines only by default
    with pytest.raises(ArgumentError, match="normal"):
        GroundLayer(normal=(0.0, 0.0, 0.0))


def test_serve_stops_playback_when_serving_ends() -> None:
    """Ctrl-C ends viser's serve; a playback thread left running dies inside torch at exit (SIGABRT)."""
    viewer = Viewer(server=FakeServer())  # the stand-in's serve returns at once
    viewer.add([_skeleton(10)])
    viewer.serve()
    assert not viewer.timeline.playing


def test_timeline_controls_appear_only_once_a_layer_has_several_steps() -> None:
    server = FakeServer()
    viewer = Viewer(server=server)
    viewer.add([_skeleton(1, "pose")])
    assert not {"Timeline", "Play", "Step", "Speed", "Loop"} & {control.name for control in server.gui.controls}
    viewer.timeline.play()  # single step: nothing to play
    assert not viewer.timeline.playing
    viewer.add([_skeleton(10, "clip")], tags="motion")
    assert server.gui.control("Step").props["max"] == 9
    assert [control.name for control in server.gui.controls].count("Timeline") == 1


def test_layers_section_has_a_checkbox_per_tag_once_there_are_two_and_follows_set_visible() -> None:
    server = FakeServer()
    viewer = Viewer(server=server)  # the ground's tag is never listed

    def boxes() -> dict[str, object]:
        return {
            c.name: c
            for c in server.gui.controls
            if c.kind == "GuiCheckboxHandle" and c.name != "Loop" and not c.removed
        }

    skel = _skeleton(10, "skel")
    viewer.add([skel])
    assert boxes() == {} and "Layers" not in [c.name for c in server.gui.controls]  # one tag: nothing to toggle
    arm, hand = _skeleton(10, "arm"), PointsLayer("hand", np.zeros((1, 3)))
    viewer.add([arm, hand], tags="robot")
    assert list(boxes()) == ["skel", "robot"]
    assert [control.name for control in server.gui.controls].count("Layers") == 1
    viewer.set_visible(arm, False)
    assert arm.visible is False and boxes()["robot"].value is True  # the hand still shows
    viewer.set_visible(hand, False)
    assert boxes()["robot"].value is False
    boxes()["skel"].value = False
    boxes()["skel"].fire()
    assert skel.visible is False

    viewer.show(4)
    boxes()["robot"].value = True
    boxes()["robot"].fire()
    assert arm.visible and hand.visible and viewer._entry(arm).step == 4
    folder = server.gui.control("Layers")
    viewer.remove("robot")
    assert boxes() == {} and folder.removed  # one tag left


def test_replace_puts_the_new_layer_in_the_old_ones_place_and_visibility() -> None:
    server = FakeServer()
    viewer = Viewer(server=server, ground=False)
    old = _skeleton(10, "body")
    viewer.add([old], tags="human")
    viewer.add([PointsLayer("cloud", np.zeros((3, 2, 3)))], tags="human")
    viewer.set_visible(old, False)
    viewer.show(2)
    new = _skeleton(5, "body2")
    viewer.replace(old, new)
    assert server.scene.nodes["/human/body"].removed and server.scene.nodes["/human/body/spheres"].removed
    assert new.tags == ("human",) and [layer.name for layer in viewer.layers] == ["body2", "cloud"]
    assert new.visible is False and server.scene.nodes["/human/body2/spheres"].visible is False
    assert viewer.timeline.nsteps == 5  # fitted to the layers now present
    viewer.set_visible(new, True)
    assert viewer._entry(new).step == 2
    with pytest.raises(ArgumentError, match="already has a layer named 'cloud'"):
        viewer.replace(new, PointsLayer("cloud", np.zeros((3, 2, 3))))
    with pytest.raises(ArgumentError, match="not in this viewer"):
        viewer.replace(old, _skeleton(5, "again"))


class _Broken(Layer):
    """Builds one node, then raises in ``build`` or on its first push, as a layer with inconsistent arrays does."""

    def __init__(self, name: str, *, fails_in: str) -> None:
        super().__init__(name, 20)
        self.fails_in = fails_in

    def build(self, scene: Any, root: str) -> None:
        self._track(scene.add_frame(f"{root}/part", show_axes=False))
        if self.fails_in == "build":
            raise RuntimeError("broken build")

    def set_step(self, step: int) -> None:
        raise RuntimeError("broken push")


@pytest.mark.parametrize("fails_in", ["build", "set_step"])
def test_a_layer_that_fails_in_add_or_replace_leaves_the_viewer_as_it_was(fails_in: str) -> None:
    server = FakeServer()
    viewer = Viewer(server=server, ground=False)
    kept = _skeleton(4, "kept")
    viewer.add([kept])
    with pytest.raises(RuntimeError, match="broken"):
        viewer.add([_skeleton(8, "fine"), _Broken("bad", fails_in=fails_in)], tags="new")
    assert viewer.layers == [kept] and viewer.timeline.nsteps == 4 and viewer.tags == ["kept"]
    assert all(node.removed for name, node in server.scene.nodes.items() if name.startswith("/new"))
    assert [c.name for c in server.gui.controls if c.name == "new" and not c.removed] == []
    with pytest.raises(RuntimeError, match="broken"):
        viewer.replace(kept, _Broken("kept", fails_in=fails_in))
    assert viewer.layers == [kept] and viewer.timeline.nsteps == 4
    assert not server.scene.nodes["/kept/kept/spheres"].removed and viewer.tags == ["kept"]
    viewer.show(2)
    np.testing.assert_array_equal(server.scene.nodes["/kept/kept/spheres"].batched_positions, kept.centers[2])


def test_set_opacity_fades_mesh_surfaces_live_and_refuses_other_layers() -> None:
    server = FakeServer()
    viewer = Viewer(server=server)
    mesh = MeshLayer("mesh", np.zeros((3, 3)), faces=np.array([[0, 1, 2]]))
    viewer.add([mesh])
    spheres = SpheresLayer("balls", np.zeros((2, 3)), radii=np.ones(2), opacity=1.0)
    viewer.add([spheres])
    mesh.set_opacity(0.4)
    spheres.set_opacity(0.4)
    assert server.scene.nodes["/mesh/mesh/mesh"].opacity == 0.4 and mesh.opacity == 0.4
    assert server.scene.nodes["/balls/balls/spheres"].opacity == 0.4
    mesh.set_opacity(1.0)
    assert server.scene.nodes["/mesh/mesh/mesh"].opacity is None  # viser's opaque
    mesh.set_opacity(0.6)
    viewer.remove(mesh)
    viewer.add(mesh)  # built again with the opacity it keeps
    assert server.scene.nodes["/mesh/mesh/mesh"].props["opacity"] == 0.6
    with pytest.raises(ArgumentError, match=r"\(0, 1\]"):
        mesh.set_opacity(0.0)
    with pytest.raises(ArgumentError, match="draws no surface"):
        PointsLayer("cloud", np.zeros((1, 3))).set_opacity(0.5)


def test_mesh_layers_check_opacity_at_construction_as_set_opacity_does() -> None:
    vertices, faces = np.zeros((3, 3)), np.array([[0, 1, 2]])
    spec = MeshSpec(vertices=vertices, faces=faces, body=0, origin=np.array([0, 0, 0, 0, 0, 0, 1.0]))
    weights, bind = np.ones((3, 4)) / 4, np.tile(np.array([0, 0, 0, 0, 0, 0, 1.0]), (4, 1))
    builders = [
        lambda opacity: MeshLayer("mesh", vertices, faces=faces, opacity=opacity),
        lambda opacity: RigidMeshLayer("rigid", [spec], body_poses=bind[None, :1], opacity=opacity),
        lambda opacity: SpheresLayer("balls", vertices, radii=np.ones(3), opacity=opacity),
        lambda opacity: SkinnedMeshLayer(
            "skin", vertices, faces=faces, weights=weights, bind_poses=bind, bone_poses=bind, opacity=opacity
        ),
    ]
    for build in builders:
        for bad in (0.0, 5.0, float("nan")):
            with pytest.raises(ArgumentError, match=r"opacity must be in \(0, 1\]"):
                build(bad)
        layer = build(0.5)
        assert layer.opacity == 0.5
        with pytest.raises(ArgumentError, match=r"opacity must be in \(0, 1\]"):
            layer.set_opacity(1.5)


class _Box(Layer):
    """A layer that only reports a box per step, ``(step, 0, 0)`` to ``(step, 1, 1)``, and counts the reads."""

    def __init__(self, name: str, nsteps: int) -> None:
        super().__init__(name, nsteps)
        self.read: list[int] = []

    def build(self, scene: Any, root: str) -> None:
        pass

    def set_step(self, step: int) -> None:
        pass

    def bounds(self, step: int) -> tuple[np.ndarray, np.ndarray]:
        self.read.append(step)
        return np.array([step, 0.0, 0.0]), np.array([step, 1.0, 1.0])


def _inside_view(points: np.ndarray, position: np.ndarray, look_at: np.ndarray, fov: float) -> bool:
    """Whether every point lies inside the cone of half-angle ``fov / 2`` around the line of sight."""
    sight = (look_at - position) / np.linalg.norm(look_at - position)
    rays = points - position
    cosines = rays @ sight / np.linalg.norm(rays, axis=-1)
    return bool((cosines >= np.cos(fov / 2)).all())


def test_aim_frames_the_bounding_sphere_of_the_visible_layers_in_new_and_connected_browsers() -> None:
    server = FakeServer()
    server.clients[0] = FakeClient()
    viewer = Viewer(server=server)  # the ground has no bounds: never framed
    points = np.zeros((3, 2, 3))
    points[:, 1, 2] = 1.0
    points[2, :, 0] = 5.0  # the last step moves away
    viewer.add(PointsLayer("pts", points), tags="g", offset=(0.0, 2.0, 0.0))
    hidden = viewer.add(PointsLayer("far", np.full((1, 3), 50.0)))
    viewer.set_visible(hidden, False)
    fov = np.radians(30.0)
    position, look_at = viewer.aim(steps=[0], fov=fov)
    for camera in (server.initial_camera, server.clients[0].camera):
        assert camera.fov == fov
        np.testing.assert_allclose(camera.position, position)
        np.testing.assert_allclose(camera.look_at, look_at)
        assert camera.writes.index("position") < camera.writes.index("look_at")
    np.testing.assert_allclose(look_at, [0.0, 2.0, 0.5])  # the box of step 0, moved by the tag's offset
    # The sphere around that box (radius 0.5) fills the field of view with a 15 % margin.
    np.testing.assert_allclose(np.linalg.norm(position - look_at), 0.5 / np.sin(fov / 2) * 1.15)
    up = viewer.theme.up_vector
    assert np.dot(position - look_at, up) > 0.0  # seen from above
    position, look_at = viewer.aim()  # the whole clip: the last step is in view too
    clip = points.reshape(-1, 3) + [0.0, 2.0, 0.0]
    assert _inside_view(clip, position, look_at, np.radians(40.0))
    np.testing.assert_allclose(look_at, [2.5, 2.0, 0.5])


def test_aim_reads_only_bounds_of_at_most_32_steps_and_skips_invalid_ones() -> None:
    viewer = Viewer(server=FakeServer())
    layer = viewer.add(_Box("box", 100))
    layer.read.clear()
    viewer.aim()
    assert len(layer.read) == 32 and layer.read[0] == 0 and layer.read[-1] == 99
    hidden = viewer.add(_Box("hidden", 3), valid=np.array([True, False, True]))
    viewer.remove(layer)
    hidden.read.clear()
    np.testing.assert_allclose(viewer.aim(steps=range(3)), viewer.aim(steps=[0, 2]))  # step 1 does not count
    assert 1 not in hidden.read
    empty = viewer.aim(steps=[1])  # nothing visible at all: a 2 x 2 x 1 m box on the origin
    np.testing.assert_allclose(empty[1], [0.0, 0.0, 0.5])


def test_the_viewer_aims_after_each_add_until_aim_is_called() -> None:
    server = FakeServer()
    viewer = Viewer(server=server)
    viewer.add(_Box("first", 1))
    first = np.asarray(server.initial_camera.position)
    viewer.add(_Box("second", 4))
    np.testing.assert_allclose(server.initial_camera.look_at, [1.5, 0.5, 0.5])  # both boxes, over the clip
    assert not np.allclose(server.initial_camera.position, first)
    chosen = viewer.aim(steps=[0])
    viewer.add(_Box("third", 1))
    np.testing.assert_allclose(server.initial_camera.position, chosen[0])  # the user's aim holds
