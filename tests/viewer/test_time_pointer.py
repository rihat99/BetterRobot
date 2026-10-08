"""Per-layer sample times, the viewer-owned pointer slot, ray picking and the camera-layer move."""

from __future__ import annotations

import numpy as np
import pytest

from better_robot.viewer import CameraLayer
from better_robot.viewer.layer import Layer
from better_robot.viewer.layers.skinned import CameraLayer as SkinnedCameraLayer
from better_robot.viewer.tools import nearest_on_ray
from better_robot.viewer.viewer import Viewer

from .fake_server import FakeServer


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
    slow = viewer.add(_Recorder("slow", 10), times=np.arange(10) / 10.0)
    fast = viewer.add(_Recorder("fast", 28), times=np.arange(28) / 30.0)
    # The 10 Hz layer's last sample is at 0.9 s, i.e. timeline step 27.
    assert viewer.timeline.nsteps == 28
    expected_slow = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3]
    for step in range(10):
        assert viewer._index(slow, step) == expected_slow[step]
        assert viewer._index(fast, step) == step
    assert viewer._index(slow, 27) == 9
    viewer.show(4)
    assert slow.shown[-1] == 1 and fast.shown[-1] == 4


def test_times_default_to_one_sample_per_step_and_are_validated() -> None:
    viewer = Viewer(server=FakeServer())
    layer = viewer.add(_Recorder("plain", 5))
    assert layer.times is None and viewer._index(layer, 3) == 3 and viewer._index(layer, 99) == 4
    with pytest.raises(ValueError, match="shape"):
        viewer.add(_Recorder("short", 5), times=np.arange(4) / 10.0)
    with pytest.raises(ValueError, match="non-decreasing"):
        viewer.add(_Recorder("back", 3), times=[0.0, 0.2, 0.1])
    with pytest.raises(ValueError, match="finite"):
        viewer.add(_Recorder("inf", 2), times=[0.0, np.inf])


def test_times_are_set_per_add_and_not_on_a_refused_add() -> None:
    viewer = Viewer(server=FakeServer())
    timed = viewer.add(_Recorder("timed", 3), times=[0.0, 0.1, 0.2])
    with pytest.raises(ValueError, match="already has a layer"):
        viewer.add(_Recorder("timed", 3), times=[0.0, 0.5, 1.0])
    np.testing.assert_array_equal(timed.times, [0.0, 0.1, 0.2])
    refused = _Recorder("timed", 3)
    with pytest.raises(ValueError, match="already has a layer"):
        viewer.add(refused, times=[0.0, 0.5, 1.0])
    assert refused.times is None
    assert viewer.add(timed, group="other").times is None


def test_on_pointer_owns_the_single_scene_pointer_slot() -> None:
    server = FakeServer()
    viewer = Viewer(server=server)
    clicks: list[object] = []
    viewer.on_pointer("click", clicks.append)
    # A second registration replaces the first instead of tripping viser's single slot.
    viewer.on_pointer("rect-select", clicks.append)
    kind, callback = server.scene.pointer
    assert kind == "rect-select"
    callback("event")
    assert clicks == ["event"]
    viewer.clear_pointer()
    assert server.scene.pointer is None
    viewer.clear_pointer()
    with pytest.raises(ValueError, match="kind"):
        viewer.on_pointer("hover", clicks.append)


def test_nearest_on_ray_picks_the_closest_point_in_front_of_the_origin() -> None:
    points = np.array([[0.0, 0.0, 2.0], [0.0, 0.05, 1.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.5]])
    origin, direction = np.zeros(3), np.array([0.0, 0.0, 3.0])
    assert nearest_on_ray(points, origin, direction, 0.1) == 1
    assert nearest_on_ray(points, origin, direction, 0.01) == 0
    assert nearest_on_ray(points, origin, -direction, 0.01) == 2
    assert nearest_on_ray(points[[3]], origin, direction, 0.1) is None


def test_camera_layer_lives_in_its_own_module_and_is_re_exported() -> None:
    assert CameraLayer.__module__ == "better_robot.viewer.layers.camera"
    assert SkinnedCameraLayer is CameraLayer
