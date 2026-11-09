"""Per-layer sample times, the viewer-owned pointer slot, ray picking and the camera-layer move."""

from __future__ import annotations

import numpy as np
import pytest

from better_robot.viewer import ArrowsLayer, CameraLayer, PointsLayer, SpheresLayer
from better_robot.viewer.layer import Layer
from better_robot.viewer.layers.skinned import CameraLayer as SkinnedCameraLayer
from better_robot.viewer.tools import nearest_on_ray
from better_robot.viewer.viewer import Viewer

from better_robot.exceptions import ArgumentError, ShapeError
from better_robot.testing import FakeServer


class _Recorder(Layer):
    """Records every step it is asked to show."""

    def __init__(self, name: str, nsteps: int) -> None:
        super().__init__(name, nsteps)
        self.shown: list[int] = []

    def build(self, scene: object, root: str) -> None:
        pass

    def set_step(self, step: int) -> None:
        self.shown.append(step)


def test_a_10hz_and_a_30hz_layer_share_one_30fps_timeline() -> None:
    viewer = Viewer(server=FakeServer(), fps=30.0)
    slow = _Recorder("slow", 10)
    viewer.add([slow], times=np.arange(10) / 10.0)
    fast = _Recorder("fast", 28)
    viewer.add([fast], times=np.arange(28) / 30.0)
    # The 10 Hz layer's last sample is at 0.9 s, i.e. timeline step 27.
    assert viewer.timeline.nsteps == 28
    expected_slow = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3]
    for step in range(10):
        viewer.show(step)
        assert slow.shown[-1] == expected_slow[step] and fast.shown[-1] == step
    viewer.show(27)
    assert slow.shown[-1] == 9


def test_times_default_to_one_sample_per_step_and_are_validated() -> None:
    viewer = Viewer(server=FakeServer())
    layer = _Recorder("plain", 5)
    viewer.add([layer])
    viewer.add([_Recorder("long", 100)], group="long")
    viewer.show(3)
    assert layer.shown[-1] == 3
    viewer.show(99)
    assert layer.shown[-1] == 4
    with pytest.raises(ShapeError, match="shape"):
        viewer.add([_Recorder("short", 5)], times=np.arange(4) / 10.0)
    with pytest.raises(ArgumentError, match="non-decreasing"):
        viewer.add([_Recorder("back", 3)], times=[0.0, 0.2, 0.1])
    with pytest.raises(ArgumentError, match="finite"):
        viewer.add([_Recorder("inf", 2)], times=[0.0, np.inf])


def test_a_refused_add_changes_nothing() -> None:
    viewer = Viewer(server=FakeServer(), fps=10.0)
    timed = _Recorder("timed", 3)
    viewer.add([timed], times=[0.0, 0.1, 0.2])
    with pytest.raises(ArgumentError, match="already has a layer"):
        viewer.add([_Recorder("timed", 3)], times=[0.0, 0.5, 1.0])
    assert viewer.timeline.nsteps == 3  # a refused add with later times does not stretch the timeline
    first, clash = _Recorder("first", 3), _Recorder("first", 3)
    with pytest.raises(ArgumentError, match="already has a layer named 'first'"):
        viewer.add([first, clash], group="pair")  # a refused call adds none of its layers
    assert first not in viewer.layers
    with pytest.raises(ArgumentError, match=r"an iterable of layers; pass \[_Recorder"):
        viewer.add(first)


def test_the_same_layer_object_is_never_added_twice() -> None:
    viewer = Viewer(server=FakeServer())
    layer, other = _Recorder("a", 3), _Recorder("b", 3)
    viewer.add([layer])
    viewer.add([other], group="g")
    with pytest.raises(ArgumentError, match="'a' is already in this viewer"):
        viewer.add([layer], group="other")
    with pytest.raises(ArgumentError, match="'c' is already in this viewer"):
        twice = _Recorder("c", 3)
        viewer.add([twice, twice], group="pair")
    with pytest.raises(ArgumentError, match="'b' is already in this viewer; replace takes a new layer object"):
        viewer.replace(layer, other)
    with pytest.raises(ArgumentError, match="rebuild redraws a layer in place"):
        viewer.replace(layer, layer)
    assert [(entry.name, id(entry)) for entry in viewer.layers] == [("a", id(layer)), ("b", id(other))]


def test_on_pointer_holds_one_scene_pointer_callback() -> None:
    server = FakeServer()
    viewer = Viewer(server=server)
    clicks: list[object] = []
    viewer.on_pointer("click", clicks.append)
    assert server.scene.pointers == [("click", clicks.append)]
    with pytest.raises(ArgumentError, match="already holds a 'click' callback; call clear_pointer"):
        viewer.on_pointer("rect-select", clicks.append)  # never replaces the held callback silently
    assert server.scene.pointers == [("click", clicks.append)]
    viewer.clear_pointer()
    viewer.on_pointer("rect-select", clicks.append)
    [(kind, callback)] = server.scene.pointers
    assert kind == "rect-select"
    callback("event")
    assert clicks == ["event"]
    viewer.clear_pointer()
    assert server.scene.pointers == []
    viewer.clear_pointer()
    with pytest.raises(ArgumentError, match="kind"):
        viewer.on_pointer("hover", clicks.append)


def test_nearest_on_ray_picks_the_closest_point_in_front_of_the_origin() -> None:
    points = np.array([[0.0, 0.0, 2.0], [0.0, 0.05, 1.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.5]])
    origin, direction = np.zeros(3), np.array([0.0, 0.0, 3.0])
    assert nearest_on_ray(points, origin, direction=direction, radius=0.1) == 1
    assert nearest_on_ray(points, origin, direction=direction, radius=0.01) == 0
    assert nearest_on_ray(points, origin, direction=-direction, radius=0.01) == 2
    assert nearest_on_ray(points[[3]], origin, direction=direction, radius=0.1) is None


def test_camera_layer_lives_in_its_own_module_and_is_re_exported() -> None:
    assert CameraLayer.__module__ == "better_robot.viewer.layers.camera"
    assert SkinnedCameraLayer is CameraLayer


def test_valid_hides_a_layer_on_invalid_samples_without_touching_its_switch() -> None:
    server = FakeServer()
    viewer = Viewer(server=server)
    valid = np.array([True, True, False, False, True])
    layer = PointsLayer("person", np.zeros((5, 4, 3)))
    viewer.add([layer], valid=valid)
    frame, cloud = server.scene.nodes["/scene/person"], server.scene.nodes["/scene/person/points"]
    viewer.show(2)
    assert frame.visible is False and cloud.visible and layer.visible  # the layer's frame hides the sample
    viewer.show(3)
    assert frame.writes.count("visible") == 1  # still invalid: no second write
    viewer.show(4)
    assert frame.visible is True
    viewer.set_visible(layer, False)
    assert cloud.visible is False and not layer.visible
    viewer.show(2)
    viewer.set_visible(layer, True)  # switched on again on an invalid sample: its frame keeps it hidden
    assert frame.visible is False and cloud.visible and layer.visible
    valid[2] = True  # kept by reference: edit in place and re-push
    viewer.show()
    assert frame.visible is True
    with pytest.raises(ShapeError, match="valid must have shape"):
        viewer.add([PointsLayer("bad", np.zeros((5, 4, 3)))], valid=np.ones((5, 4), bool))  # points: whole layer only


def test_valid_per_element_hides_single_spheres_and_arrows() -> None:
    server = FakeServer()
    viewer = Viewer(server=server)
    keep = np.array([[True, False], [False, True]])
    viewer.add([SpheresLayer("contacts", np.ones((2, 2, 3)), radii=np.array([0.1, 0.2]))], valid=keep)
    viewer.add([ArrowsLayer("forces", np.zeros((2, 2, 3)), vectors=np.ones((2, 2, 3)))], valid=keep)
    viewer.show(1)
    np.testing.assert_allclose(server.scene.nodes["/scene/contacts/spheres"].batched_scales, [0.0, 0.2])
    np.testing.assert_allclose(server.scene.nodes["/scene/forces/heads"].batched_scales[0], 0.0)
    shafts = server.scene.nodes["/scene/forces/shafts"].points
    np.testing.assert_allclose(shafts[0, 1], shafts[0, 0])  # zero-length shaft
    assert len(viewer._points([1])) == 4  # camera fit: one sphere (2 corners) + one arrow (origin, tip)


def test_remove_group_shrinks_the_timeline_to_the_remaining_layers() -> None:
    server = FakeServer()
    viewer = Viewer(server=server)
    viewer.add([_Recorder("long", 10)], group="clip_a")
    short = _Recorder("short", 4)
    viewer.add([short], group="clip_b")
    viewer.show(8)
    viewer.remove_group("clip_a")
    assert viewer.timeline.nsteps == 4 and viewer.timeline.step == 3
    assert server.gui.control("Step").max == 3 and short.shown[-1] == 3
    viewer.remove_group("clip_b")
    assert viewer.timeline.nsteps == 1 and viewer.timeline.step == 0
    viewer.add([_Recorder("next", 6)], group="clip_c")
    assert viewer.timeline.nsteps == 6


def test_set_fps_changes_the_rate_and_remaps_timed_layers() -> None:
    viewer = Viewer(server=FakeServer(), fps=30.0)
    clip = _Recorder("clip", 12)
    viewer.add([clip])
    video = _Recorder("video", 10)
    viewer.add([video], times=np.arange(10) / 10.0)
    assert viewer.timeline.nsteps == 28
    viewer.show(6)
    viewer.set_fps(10.0)  # step = frame of a 10 fps clip
    assert viewer.timeline.fps == 10.0 and viewer.timeline.nsteps == 12 and viewer.timeline.step == 6
    assert video.shown[-1] == 6 and clip.shown[-1] == 6
    viewer.timeline.play()
    viewer.set_fps(60.0)
    assert viewer.timeline.playing  # playback continues at the new rate
    viewer.timeline.pause()
    with pytest.raises(ArgumentError, match="fps must be positive"):
        viewer.set_fps(0.0)
