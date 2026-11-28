"""Viewer core on a fake server: frame pushes, visibility, groups, timeline."""

from __future__ import annotations

import dataclasses
import threading
import time
from typing import Any

import numpy as np
import pytest
import torch

from better_robot.lie import so3
from better_robot.viewer.layer import Layer
from better_robot.viewer import ArrowsLayer, FramesLayer, GroundLayer, PlotPanel, PointsLayer
from better_robot.viewer.layers.image import ImagePanel
from better_robot.viewer.layers.mesh import MeshLayer, MeshSpec, RigidMeshLayer, SpheresLayer
from better_robot.viewer.layers.skinned import SkinnedMeshLayer
from better_robot.viewer.sidebar import Sidebar
from better_robot.viewer.theme import DARK, LIGHT, Theme
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
    assert pushed == {"/skel/skel/spheres", "/forces/f/arrows"}
    # The arrows layer has 5 frames: frame 7 clamps to its last frame and the static grid is never touched.
    server.scene.log.clear()
    viewer.timeline.step = 7
    assert ("/forces/f/arrows", "points") in server.scene.log
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
    checkbox = server.gui.control("Skel")
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


def test_set_pose_places_a_tag_after_it_was_added_and_aim_frames_it_turned() -> None:
    """A y-up clip beside a z-up one: the tag's frame turns +y up to +z, the scene node gets the pose (viser's
    scalar-first quaternion), and aim, which record shares, frames the turned box, not the box as given."""
    server = FakeServer()
    viewer = Viewer(server=server, ground=False)
    tall = np.array([[0.0, 0.0, 0.0], [0.1, 2.0, 0.1]])  # 2 m along +y, its up
    viewer.add([PointsLayer("p", tall)], tags="body", pose=(0.0, 2.0, 0.0))
    np.testing.assert_allclose(server.scene.nodes["/body"].position, [0.0, 2.0, 0.0])
    up = so3.from_axis_angle(
        torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64), torch.tensor(np.pi / 2, dtype=torch.float64)
    ).numpy()
    viewer.set_pose("body", [0.0, 3.0, 0.0, *up])
    node = server.scene.nodes["/body"]
    np.testing.assert_allclose(node.position, [0.0, 3.0, 0.0])
    np.testing.assert_allclose(node.wxyz, up[[3, 0, 1, 2]])
    look_at = viewer.aim()[1]
    np.testing.assert_allclose(look_at, [0.05, 2.95, 1.0], atol=1e-9)  # the box turned upright: z from 0 to 2
    with pytest.raises(UnknownNameError, match="unknown tag 'missing'"):
        viewer.set_pose("missing", (0.0, 0.0, 0.0))
    with pytest.raises(ShapeError, match=r"pose must be \(7,\)"):
        viewer.set_pose("body", (0.0, 0.0))
    with pytest.raises(ArgumentError, match="unit norm"):
        viewer.add(PointsLayer("q", tall), pose=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    assert [layer.name for layer in viewer.layers] == ["p"]  # refused before anything was built


def test_a_pose_of_non_finite_or_non_numeric_values_is_refused_and_a_near_unit_quaternion_is_normalised() -> None:
    """R1's probe: a NaN or inf translation and ``None`` inside were accepted (the camera turned NaN), a string or a
    ragged pose raised numpy's builtin error, and a norm of 1.0009 reached viser as it was."""
    server = FakeServer()
    viewer = Viewer(server=server, ground=False)
    refused = [[np.nan, 0, 0, 0, 0, 0, 1], [np.inf, 0, 0], [0, 0, None], "abc", [0, 0, [0]]]
    for pose in refused:
        with pytest.raises(ArgumentError, match="pose must be"):
            viewer.add(PointsLayer("q", np.zeros((1, 3))), pose=pose)
    assert viewer.layers == []
    given = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0009])
    viewer.add(PointsLayer("q", np.zeros((1, 3))), pose=given)
    np.testing.assert_allclose(server.scene.nodes["/q"].wxyz, [1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(viewer.pose("q"), [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    assert given[6] == 1.0009  # the caller's array is left as it was
    with pytest.raises(ArgumentError, match="unit norm"):
        viewer.set_pose("q", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0011])


def test_timeline_clamps_loops_and_drives_the_gui_slider() -> None:
    timeline = Timeline(5, fps=100.0)
    seen: list[int] = []
    timeline.on_step(seen.append)
    server = FakeServer()
    gui = server.gui
    timeline.build_gui(Sidebar(server, threading.RLock()))
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
    assert gui.control("Play").value is False


def test_timeline_controls_run_under_the_viewer_lock_without_waiting_for_playback() -> None:
    server = FakeServer()
    viewer = Viewer(server=server)
    viewer.add(_skeleton(1000))
    play, speed = server.gui.control("Play"), server.gui.control("Speed")
    assert [c.name for c in server.gui.controls if c.kind != "GuiFolderHandle"][:4] == ["Play", "Step", "Speed", "Loop"]
    play.value = True
    play.fire()  # under the lock, as every sidebar callback
    time.sleep(0.1)
    speed.value = "2\u00d7"
    speed.fire()
    assert viewer.timeline.playing and viewer.timeline.speed == 2.0  # no restart: the thread reads it
    with viewer.lock:  # the playback thread waits for this lock: unticking Play must not wait for the thread
        play.value = False
        play.fire()
    assert not viewer.timeline.playing
    viewer.timeline.pause()  # outside the lock it waits for the thread to end


def test_play_as_a_button_reads_play_or_pause() -> None:
    server = FakeServer()
    viewer = Viewer(server=server)
    viewer.timeline.play_button = True
    viewer.add(_skeleton(1000))
    play = server.gui.control("Play")
    assert play.kind == "GuiButtonHandle" and play.label == "Play"
    with viewer.lock:  # under the lock, as every sidebar callback: pausing must not wait for the thread
        play.fire()
        assert viewer.timeline.playing and play.label == "Pause"
        play.fire()
    assert not viewer.timeline.playing and play.label == "Play"
    viewer.timeline.play()  # started from code, the button follows
    assert play.label == "Pause"
    viewer.timeline.pause()
    assert play.label == "Play"


def test_a_sidebar_button_may_change_the_fps_while_playing() -> None:
    viewer = Viewer(server=FakeServer(), ground=False)
    viewer.add(PointsLayer("cloud", np.zeros((300, 10, 3))))

    def on_click() -> None:
        time.sleep(0.1)  # work under the lock: the playback thread comes to wait for it
        viewer.set_fps(60.0)  # pauses playback, which waits for the playback thread

    viewer.sidebar.button("Scene", "60 fps", on_click=on_click)
    viewer.timeline.play()
    time.sleep(0.2)
    click = threading.Thread(target=viewer.server.gui.control("60 fps").fire, daemon=True)
    click.start()
    click.join(1.0)
    try:
        assert not click.is_alive(), "the callback waits for the playback thread, which waits for the callback's lock"
        assert viewer.timeline.fps == 60.0 and viewer.timeline.playing
    finally:
        viewer.timeline._halt()  # never join: after a deadlock the thread would never end


class _Probe(Layer):
    """Records ``(thread name, step)`` of every push; each push takes a little time, as a real one does."""

    def __init__(self, nsteps: int) -> None:
        super().__init__("probe", nsteps)
        self.pushed: list[tuple[str, int]] = []

    def build(self, scene: Any, root: str) -> None:
        pass

    def set_step(self, step: int) -> None:
        self.pushed.append((threading.current_thread().name, step))
        time.sleep(1e-4)


def test_save_while_playing_records_each_step_once_and_playback_goes_on(tmp_path: Any) -> None:
    viewer = Viewer(server=FakeServer(), fps=1000.0, ground=False)
    probe = viewer.add(_Probe(200))
    viewer.timeline.play()
    time.sleep(0.05)
    try:
        probe.pushed.clear()
        viewer.save(tmp_path / "clip.viser")
        saved = [step for thread, step in probe.pushed[:200]]
        assert saved == list(range(200)), "the playback thread pushed its steps between the saved ones"
        assert viewer.timeline.playing
    finally:
        viewer.timeline.pause()
    for fps in (0.0, -1.0, float("nan")):
        with pytest.raises(ArgumentError, match="fps must be positive"):
            viewer.save(tmp_path / "clip.viser", fps=fps)


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
    np.testing.assert_array_equal(node.props["batched_scales"], 1.0)
    frames.scales[0] = 2.0  # the array rule: an edit in place reaches the node at the next push
    frames.set_step(1)
    np.testing.assert_array_equal(node.batched_scales, [2.0, 1.0, 1.0])

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


def test_a_layer_draws_its_array_attributes_so_an_edit_in_place_and_a_redraw_update_the_scene() -> None:
    """One rule for every layer: the float32 attribute is what it draws. A C-contiguous float32 input is that
    attribute (no copy of a big cloud); any other input is converted once, so the attribute is what to edit."""
    server = FakeServer()
    viewer = Viewer(server=server, ground=False)
    cloud = np.zeros((3, 4, 3))  # numpy's default float64
    points = viewer.add(PointsLayer("cloud", cloud))
    origins = np.zeros((1, 2, 3), np.float32)
    arrows = viewer.add(ArrowsLayer("forces", origins, vectors=np.ones((1, 2, 3), np.float32), scale=0.5))
    assert points.points is not cloud and arrows.origins is origins
    points.points[:, :, 2] += 0.5
    arrows.vectors[0, 1] = (0.0, 0.0, 4.0)  # unscaled, as given: the scale applies at the push
    viewer.show()
    np.testing.assert_allclose(server.scene.nodes["/cloud/cloud/points"].points[:, 2], 0.5)
    np.testing.assert_allclose(server.scene.nodes["/forces/forces/arrows"].points[1, 1], [0.0, 0.0, 2.0])
    assert np.shares_memory(PointsLayer("static", origins[0]).points, origins)  # a static float32 cloud too


def test_radii_and_a_static_coloured_mesh_follow_the_array_rule_and_no_layer_keeps_a_strided_view() -> None:
    """R1's probe: an edit of ``SpheresLayer.radii`` or of a static coloured ``MeshLayer``'s vertices plus a redraw
    wrote nothing, and ``radii`` and a skin's rest arrays kept a non-contiguous float32 view of the caller's array."""
    server = FakeServer()
    viewer = Viewer(server=server, ground=False)
    balls = viewer.add(SpheresLayer("balls", np.zeros((2, 2, 3)), radii=np.ones(2, np.float32)))
    triangle = np.eye(3, dtype=np.float32)
    mesh = viewer.add(MeshLayer("scan", triangle, faces=[[0, 1, 2]], colors=np.eye(3)))
    scan, mark = server.scene.nodes["/scan/scan/mesh"], len(server.scene.log)
    viewer.show(1)
    viewer.show()  # a forced push sends nothing that did not change
    assert ("/balls/balls/spheres", "batched_scales") not in server.scene.log[mark:] and scan.writes == []
    balls.radii[0] = 3.0
    mesh.vertices[0, 0] = (5.0, 0.0, 0.0)
    viewer.show()
    np.testing.assert_allclose(server.scene.nodes["/balls/balls/spheres"].batched_scales, [3.0, 1.0])
    assert scan.writes == ["glb_data"] and scan.glb_data != scan.props["glb_data"]
    strided = np.zeros((6, 3), np.float32)[::2]
    rest = np.tile(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], np.float32), (2, 2))[:, :7]
    skin = SkinnedMeshLayer(
        "skin", strided, faces=[[0, 1, 2]], weights=np.full((3, 2), 0.5), bind_poses=rest, bone_poses=rest[None]
    )
    assert not np.shares_memory(SpheresLayer("s", strided, radii=strided[:, 0]).radii, strided)
    assert not np.shares_memory(skin.vertices, strided) and not np.shares_memory(skin.bind_poses, rest)


def test_arrows_hide_zero_vectors() -> None:
    scene = FakeServer().scene
    arrows = ArrowsLayer(
        "a", np.zeros((1, 2, 3)), vectors=[[[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]], scale=2.0, head_length=0.1
    )
    arrows.build(scene, "/g/a")
    drawn = scene.nodes["/g/a/arrows"].props  # viser's arrows (tail, tip): the zero vector is left out
    np.testing.assert_allclose(drawn["points"], [[[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]]])
    assert drawn["head_length"] == 0.1 and drawn["colors"].shape == (1, 3)


def test_arrows_are_viser_arrows_of_the_shown_vectors_with_their_per_step_colours() -> None:
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
                shaft_radius=0.02,
                head_radius=0.04,
                head_length=0.1,
            )
        ],
        valid=keep,
    )
    arrows = server.scene.nodes["/f/f/arrows"]
    assert arrows.kind == "ArrowsHandle"
    assert (arrows.props["shaft_radius"], arrows.props["head_radius"], arrows.props["head_length"]) == (0.02, 0.04, 0.1)
    viewer.show(0)  # the zero vector is left out; the short one stays
    np.testing.assert_allclose(arrows.points, [[[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]], [[0.0, 0.0, 0.0], [0.05, 0.0, 0.0]]])
    np.testing.assert_array_equal(arrows.colors, np.zeros((2, 3)))
    viewer.show(1)  # the long arrow is invalid here; the colours follow the arrows shown
    np.testing.assert_allclose(arrows.points, [[[0.0, 0.0, 0.0], [0.05, 0.0, 0.0]]])
    np.testing.assert_array_equal(arrows.colors, [[0, 0, 255]])


def test_arrows_draw_nothing_up_to_min_length() -> None:
    """Zero and float-noise vectors (default ``min_length``), or ones up to a given ``min_length``, are left out: no
    lone head."""
    server = FakeServer()
    viewer = Viewer(server=server)
    lengths = np.array([0.0, 1e-7, 5e-3, 0.5], dtype=np.float32)
    vectors = np.zeros((1, 4, 3), dtype=np.float32)
    vectors[..., 2] = lengths
    for name, min_length in (("default", {}), ("coarse", {"min_length": 0.01})):
        viewer.add([ArrowsLayer(name, np.ones((1, 4, 3)), vectors=vectors, **min_length)])
    viewer.show(0)
    for name, shown in (("default", [5e-3, 0.5]), ("coarse", [0.5])):
        points = server.scene.nodes[f"/{name}/{name}/arrows"].points
        np.testing.assert_allclose(points[:, 1, 2] - points[:, 0, 2], shown, atol=1e-6)  # float32 tips


def test_spheres_and_arrows_take_per_step_colours() -> None:
    server = FakeServer()
    viewer = Viewer(server=server)
    colours = np.zeros((3, 2, 3))
    colours[1, 0] = (1.0, 0.0, 0.0)  # contact 0 turns red at step 1
    viewer.add([SpheresLayer("contacts", np.zeros((3, 2, 3)), radii=np.full(2, 0.05), colors=colours)])
    viewer.add([ArrowsLayer("forces", np.zeros((3, 2, 3)), vectors=np.ones((3, 2, 3)), colors=colours)])
    spheres, arrows = server.scene.nodes["/contacts/contacts/spheres"], server.scene.nodes["/forces/forces/arrows"]
    viewer.show(1)
    np.testing.assert_array_equal(spheres.batched_colors, [[255, 0, 0], [0, 0, 0]])
    np.testing.assert_array_equal(arrows.colors, [[255, 0, 0], [0, 0, 0]])
    viewer.show(2)
    assert not spheres.batched_colors.any()
    static = SpheresLayer("static", np.zeros((3, 2, 3)), radii=np.full(2, 0.05), colors=(0.0, 1.0, 0.0))
    viewer.add([static])
    assert static._colors.shape == (2, 3)  # one colour stays static: never rewritten per step
    with pytest.raises(ShapeError, match="per-step colors"):
        ArrowsLayer("bad", np.zeros((3, 2, 3)), vectors=np.ones((3, 2, 3)), colors=np.zeros((2, 2, 3)))


@pytest.mark.parametrize("colors", [np.zeros((2, 5, 3)), np.zeros((4, 7, 3))], ids=["steps", "points"])
def test_points_refuse_per_step_colours_of_another_shape(colors: np.ndarray) -> None:
    """PointsLayer took per-step colours of any ``(T', N', 3)``, then failed or drew wrong colours at a later step;
    the other layers with per-step colours refused them."""
    with pytest.raises(ShapeError, match="per-step colors"):
        PointsLayer("bad", np.zeros((4, 5, 3)), colors=colors)


def test_plot_panel_lives_in_the_gui_and_marks_the_current_frame() -> None:
    server = FakeServer()
    viewer = Viewer(server=server)
    panel = viewer.add(PlotPanel("tau", np.random.default_rng(0).normal(size=(20, 3)), labels=["a", "b", "c"]))
    plot = next(control for control in server.gui.controls if control.kind == "GuiUplotHandle")
    assert len(plot.props["data"]) == 4 and plot.props["data"][0][-1] == pytest.approx(19 / 30)
    viewer.show(7)
    assert plot.series[1]["points"]["filter"] == (7,) and plot.series[1]["label"] == "a"
    assert panel.nsteps == 20 and viewer.timeline.nsteps == 20
    viewer.remove(panel)
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


def test_theme_shades_run_from_dark_drum_faces_to_pale_bones_with_bodies_in_the_hue() -> None:
    theme = Theme()
    hue = theme.hue(0)
    brightness = {role: sum(theme.shade(hue, role)) for role in ("face", "joint", "body", "stick", "bone")}
    assert brightness["face"] < brightness["joint"] < brightness["body"] < brightness["stick"] < brightness["bone"]
    assert theme.shade(hue, "body") == hue
    with pytest.raises(ArgumentError, match="role must be one of"):
        theme.shade(hue, "hull")  # pyright: ignore[reportArgumentType]
    with pytest.raises(ShapeError, match="hue must be an RGB triple"):  # was the builtin error of an unpack
        theme.shade((1.0, 0.0), "body")  # pyright: ignore[reportArgumentType]


def test_the_theme_leaves_visers_lights_alone_and_the_ground_takes_its_shadow() -> None:
    server = FakeServer()
    viewer = Viewer(server=server)  # the stand-in has no light or environment-map call: one would raise here
    assert not [name for name, node in server.scene.nodes.items() if "Light" in node.kind]
    grid = server.scene.nodes["/ground/ground/grid"].props
    assert grid["infinite_grid"] and grid["shadow_opacity"] == LIGHT.shadow
    viewer.set_theme(DARK)
    assert server.scene.nodes["/ground/ground/grid"].shadow_opacity == DARK.shadow
    GroundLayer(size=4.0).build(server.scene, "/g/square")  # a number keeps a square of that width
    square = server.scene.nodes["/g/square/grid"].props
    assert not square["infinite_grid"] and square["width"] == 4.0


def test_the_sidebar_width_is_the_themes_also_through_set_theme() -> None:
    """Three applications use viser's "large" sidebar; set_theme used to put "medium" back."""
    server = FakeServer()
    viewer = Viewer(server=server, theme=Theme(sidebar_width="large"))
    assert server.gui.theme["control_width"] == "large"
    viewer.set_theme(dataclasses.replace(DARK, sidebar_width="large"))
    assert server.gui.theme["control_width"] == "large" and server.gui.theme["dark_mode"]
    viewer.set_theme(LIGHT)
    assert server.gui.theme["control_width"] == "medium"
    with pytest.raises(ArgumentError, match="sidebar_width must be one of"):
        Theme(sidebar_width="wide")  # type: ignore[arg-type]


def test_ground_plane_follows_the_theme_up_axis_unless_given() -> None:
    server = FakeServer()
    viewer = Viewer(server=server, theme=Theme(up="+y"), ground=False)
    ground, explicit = GroundLayer(), GroundLayer("explicit", plane="xy")
    viewer.add([ground, explicit])

    def planes() -> tuple[str, str]:
        return tuple(server.scene.nodes[f"/{name}/{name}/grid"].props["plane"] for name in ("ground", "explicit"))

    assert planes() == ("xz", "xy") and ground.plane is None  # the argument stays as given
    viewer.set_visible(ground, False)
    viewer.set_theme(Theme(up="+x"))  # the grid is made again in the new plane, still hidden
    assert planes() == ("yz", "xy") and not server.scene.nodes["/ground/ground/grid"].visible
    assert GroundLayer().bounds(0) is None  # the grid never drives the camera's framing


def test_ground_tilts_to_a_normal_through_an_origin_and_fills() -> None:
    scene = FakeServer().scene
    normal = np.array([0.0, 1.0, 1.0]) / np.sqrt(2.0)
    GroundLayer(origin=(1.0, 2.0, 3.0), normal=(0.0, 1.0, 1.0), height=0.5, fill=0.4).build(scene, "/g/floor")
    grid = scene.nodes["/g/floor/grid"].props
    w, x, y, z = grid["wxyz"]
    turned = so3.act(
        torch.tensor([x, y, z, w], dtype=torch.float64), torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    )
    np.testing.assert_allclose(turned.numpy(), normal, atol=1e-6)
    np.testing.assert_allclose(grid["position"], np.array([1.0, 2.0, 3.0]) + 0.5 * normal)
    assert grid["plane"] == "xy" and grid["plane_opacity"] == 0.4
    GroundLayer(plane="xz").build(scene, "/g/flat")
    assert scene.nodes["/g/flat/grid"].props["plane_opacity"] == 0.0  # lines only by default
    with pytest.raises(ArgumentError, match="normal"):
        GroundLayer(normal=(0.0, 0.0, 0.0))


def test_serve_returns_on_ctrl_c_and_stops_playback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ctrl-C ends viser's serve, which returns; a playback thread left running dies inside torch at exit (SIGABRT)."""
    server = FakeServer()
    viewer = Viewer(server=server)
    viewer.add([_skeleton(10)])

    def ctrl_c() -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(server, "sleep_forever", ctrl_c)
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


def test_the_images_section_follows_plots_before_an_applications_sections() -> None:
    server = FakeServer()
    sidebar = Sidebar(server, threading.RLock())
    for name in ("Info", "Tuning", "Images", "Plots", "Layers"):
        sidebar.section(name)
    folders = sorted(
        (control for control in server.gui.controls if control.kind == "GuiFolderHandle"), key=lambda f: f.order
    )
    assert [folder.label for folder in folders] == ["Layers", "Plots", "Images", "Tuning", "Info"]


def test_a_shared_dropdown_offers_only_what_the_layers_still_there_offer() -> None:
    """A Style dropdown shared by a robot with meshes and one without loses "meshes" with the first; the choice it
    showed goes quietly to the first option left: the layers still there keep their state."""
    server = FakeServer()
    sidebar = Sidebar(server, threading.RLock())
    chosen: dict[str, list[str]] = {"meshes": [], "skeleton": []}

    class Robot:
        def __init__(self, name: str, styles: tuple[str, ...]) -> None:
            self.name, self.styles = name, styles

        def build_gui(self, sidebar: Sidebar) -> None:
            on_change = chosen[self.name].append
            sidebar.dropdown("Robot", "Style", options=self.styles, value=self.styles[0], on_change=on_change)

    with_meshes, skeleton = Robot("meshes", ("meshes", "minimal", "medium")), Robot("skeleton", ("minimal", "medium"))
    sidebar.build(with_meshes)  # type: ignore[arg-type]
    sidebar.build(skeleton)  # type: ignore[arg-type]
    style = server.gui.control("Style")
    assert style.options == ("meshes", "minimal", "medium") and style.value == "meshes"
    sidebar.remove(with_meshes)  # type: ignore[arg-type]
    assert style.options == ("minimal", "medium") and style.value == "minimal"
    assert chosen == {"meshes": [], "skeleton": []}
    style.value = "medium"
    style.fire(None)
    assert chosen["skeleton"] == ["medium"]


class _Styled(Layer):
    """A layer with RobotVisual's shared Style dropdown: it takes the styles it has and ignores the others."""

    def __init__(self, name: str, styles: tuple[str, ...], style: str) -> None:
        super().__init__(name)
        self.styles, self.style, self.chosen = styles, style, []

    def build(self, scene: Any, root: str) -> None:
        pass

    def set_step(self, step: int) -> None:
        pass

    def build_gui(self, sidebar: Sidebar) -> None:
        def choose(style: str) -> None:
            if style in self.styles:
                self.style = style
                self.chosen.append(style)

        sidebar.dropdown("Robot", "Style", options=self.styles, value=self.style, on_change=choose)


def test_removing_or_replacing_a_layer_never_restyles_the_others() -> None:
    """R1's probe: a G1 with meshes beside a Panda at "medium"; removing the G1 left the Panda "minimal", and a robot
    built with ``style="medium"`` that replaced the G1 came up "minimal"."""
    server = FakeServer()
    viewer = Viewer(server=server, ground=False)
    styles = ("minimal", "medium")
    g1 = viewer.add(_Styled("g1", ("meshes", *styles), "meshes"))
    panda = viewer.add(_Styled("panda", styles, "medium"))
    style = server.gui.control("Style")
    viewer.remove(g1)
    assert panda.style == "medium" and panda.chosen == []
    assert "meshes" not in style.options and style.value in style.options
    g1 = viewer.add(_Styled("g1", ("meshes", *styles), "meshes"))
    style.value = "meshes"  # a choice in the browser: the robots that have it follow
    style.fire(None)
    new = _Styled("panda2", styles, "medium")
    viewer.replace(g1, new)
    assert new.style == "medium" and new.chosen == [] and panda.style == "medium"
    assert "meshes" not in style.options and style.value in style.options


def test_layers_section_has_a_checkbox_per_tag_once_there_are_two_and_follows_set_visible() -> None:
    server = FakeServer()
    viewer = Viewer(server=server)  # the ground's tag is never listed

    def boxes() -> dict[str, object]:
        return {
            c.name: c
            for c in server.gui.controls
            if c.kind == "GuiCheckboxHandle" and c.name not in ("Play", "Loop") and not c.removed
        }

    skel = _skeleton(10, "skel")
    viewer.add([skel])
    assert boxes() == {} and "Layers" not in [c.name for c in server.gui.controls]  # one tag: nothing to toggle
    arm, hand = _skeleton(10, "arm"), PointsLayer("hand", np.zeros((1, 3)))
    viewer.add([arm, hand], tags="robot")
    viewer.add(PlotPanel("torque", np.zeros((10, 2))))  # a chart in the sidebar has nothing to hide in the scene
    assert list(boxes()) == ["Skel", "Robot"]
    assert [control.name for control in server.gui.controls].count("Layers") == 1
    viewer.set_visible(arm, False)
    assert arm.visible is False and boxes()["Robot"].value is True  # the hand still shows
    viewer.set_visible(hand, False)
    assert boxes()["Robot"].value is False
    boxes()["Skel"].value = False
    boxes()["Skel"].fire()
    assert skel.visible is False

    viewer.show(4)
    boxes()["Robot"].value = True
    boxes()["Robot"].fire()
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
    assert server.scene.nodes["/balls/balls/spheres"].props["cast_shadow"]  # opaque: it casts as any solid does
    mesh.set_opacity(0.4)
    spheres.set_opacity(0.4)
    node = server.scene.nodes["/mesh/mesh/mesh"]
    assert node.opacity == 0.4 and mesh.opacity == 0.4
    assert server.scene.nodes["/balls/balls/spheres"].opacity == 0.4
    for faded in (node, server.scene.nodes["/balls/balls/spheres"]):  # the shadow map would draw a glass solid
        assert not faded.cast_shadow and not faded.receive_shadow
    mesh.set_opacity(1.0)
    assert node.opacity is None and node.cast_shadow and node.receive_shadow  # viser's opaque
    mesh.set_opacity(0.6)
    viewer.remove(mesh)
    viewer.add(mesh)  # built again with the opacity it keeps, and without shadows
    props = server.scene.nodes["/mesh/mesh/mesh"].props
    assert (props["opacity"], props["cast_shadow"], props["receive_shadow"]) == (0.6, False, False)
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
    """Whether every point lies inside a 16:9 view of vertical field of view ``fov``, upright along +z."""
    sight = (look_at - position) / np.linalg.norm(look_at - position)
    right = np.cross(sight, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    rays = points - position
    depth = rays @ sight
    tall, wide = np.abs(rays @ np.cross(right, sight)) / depth, np.abs(rays @ right) / depth
    return bool((depth > 0).all() and (tall <= np.tan(fov / 2)).all() and (wide <= np.tan(fov / 2) * 16 / 9).all())


def test_aim_frames_the_box_of_the_visible_layers_in_new_and_connected_browsers() -> None:
    server = FakeServer()
    server.clients[0] = FakeClient()
    viewer = Viewer(server=server)  # the ground has no bounds: never framed
    points = np.zeros((3, 2, 3))
    points[:, 1, 2] = 1.0
    points[2, :, 0] = 5.0  # the last step moves away
    viewer.add(PointsLayer("pts", points), tags="g", pose=(0.0, 2.0, 0.0))
    hidden = viewer.add(PointsLayer("far", np.full((1, 3), 50.0)))
    viewer.set_visible(hidden, False)
    fov = np.radians(30.0)
    position, look_at = viewer.aim(steps=[0], fov=fov)
    for camera in (server.initial_camera, server.clients[0].camera):
        assert camera.fov == fov
        np.testing.assert_allclose(camera.position, position)
        np.testing.assert_allclose(camera.look_at, look_at)
        assert camera.writes.index("position") < camera.writes.index("look_at")
    np.testing.assert_allclose(look_at, [0.0, 2.0, 0.5])  # the box of step 0, moved by the tag's pose
    # That box spans 80 % of a 16:9 view: its corners reach the bound on one screen axis and stay inside the other.
    view = (look_at - position) / np.linalg.norm(look_at - position)
    right = np.cross(view, viewer.theme.up_vector)
    right /= np.linalg.norm(right)
    corners = np.array([[x, y, z] for x in (0.0, 0.0) for y in (2.0, 2.0) for z in (0.0, 1.0)]) - position
    depth = corners @ view
    tall = np.abs(corners @ np.cross(right, view)) / depth / np.tan(fov / 2)
    wide = np.abs(corners @ right) / depth / (np.tan(fov / 2) * 16 / 9)
    np.testing.assert_allclose(max(tall.max(), wide.max()), 0.8)
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


def test_adding_a_layer_reads_the_bounds_of_the_others_only_after_they_were_pushed_again() -> None:
    """The automatic aim of ``add`` keeps each layer's boxes, so building a scene of N robots is not quadratic, and
    reads them again once a step reached the layer (``show``, ``redraw``, ``update``), the way its arrays change."""
    server = FakeServer()
    viewer = Viewer(server=server, ground=False)
    first = viewer.add(_Box("first", 40))
    assert len(first.read) == 32  # one read per sampled step
    first.read.clear()
    second = viewer.add(_Box("second", 40), pose=(0.0, 3.0, 0.0))
    assert first.read == [] and len(second.read) == 32
    viewer.show()  # a push: the arrays may have changed in place
    viewer.add(_Box("third", 1))
    assert len(first.read) == 32 and len(second.read) == 64
    viewer.set_pose("second", (0.0, -3.0, 0.0))  # poses and visibility are applied after the reads
    viewer.set_visible("third", False)
    second.read.clear()
    viewer.add(_Box("fourth", 1))
    assert second.read == []
    remembered = np.asarray(server.initial_camera.position), np.asarray(server.initial_camera.look_at)
    np.testing.assert_allclose(remembered, viewer.aim())  # the view of a fresh read


def test_a_redraw_while_hidden_drops_the_boxes_a_layer_shown_by_its_own_switch_is_framed_by() -> None:
    """R1's probe: ``layer.set_visible(False)``, an edit, ``redraw()`` and ``layer.set_visible(True)`` left the automatic
    aim on the old boxes, 13.9 m from a fresh fit."""
    server = FakeServer()
    viewer = Viewer(server=server, ground=False)
    cloud = viewer.add(PointsLayer("cloud", np.zeros((1, 2, 3), np.float32)))
    cloud.set_visible(False)
    cloud.points[0, 1] = (0.0, 0.0, 10.0)
    cloud.redraw()
    cloud.set_visible(True)
    viewer.add(PointsLayer("other", np.zeros((1, 1, 3), np.float32)))
    remembered = np.asarray(server.initial_camera.position), np.asarray(server.initial_camera.look_at)
    np.testing.assert_allclose(remembered, viewer.aim())  # the view of a fresh read


def test_a_step_change_playback_or_a_longer_timeline_keeps_the_boxes_the_automatic_aim_read() -> None:
    """A plain step change carries no edit: building a scene while it plays or while its clips grow the timeline
    reads each layer's samples once."""
    server = FakeServer()
    viewer = Viewer(server=server, ground=False)
    first = viewer.add(_Box("first", 40))
    first.read.clear()
    viewer.show(5)
    second = viewer.add(_Box("second", 60), pose=(0.0, 3.0, 0.0))  # a longer clip: the timeline grows
    viewer.timeline.step = 50  # what playback does
    viewer.add(_Box("third", 90))
    assert first.read == [] and len(second.read) == 32
    remembered = np.asarray(server.initial_camera.position), np.asarray(server.initial_camera.look_at)
    np.testing.assert_allclose(remembered, viewer.aim())  # the view of a fresh read
