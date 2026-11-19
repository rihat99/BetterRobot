"""Gizmo pose round trip and picking math."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from better_robot.viewer import PointsLayer
from better_robot.viewer.tools import (
    CameraFrustum,
    PoseGizmo,
    nearest_on_ray,
    project,
    ray_hit,
    vertices_in_rect,
    vertices_in_sphere,
)
from better_robot.viewer.viewer import Viewer

from better_robot.exceptions import ShapeError, UnknownNameError
from better_robot.testing import FakeServer


def test_gizmo_reports_poses_in_xyzw_order_and_fires_callbacks() -> None:
    server = FakeServer()
    viewer = Viewer(server=server)
    seen: list[np.ndarray] = []
    gizmo = PoseGizmo(viewer, "target", pose=[1.0, 2.0, 3.0, 0.0, 0.0, 1.0, 0.0], on_change=seen.append)
    handle = server.scene.nodes["/tools/target"]
    np.testing.assert_allclose(handle.props["wxyz"], [0.0, 0.0, 0.0, 1.0])
    handle.position = np.array([4.0, 5.0, 6.0])
    handle.wxyz = np.array([1.0, 0.0, 0.0, 0.0])
    handle.fire()
    assert gizmo.wait(timeout=5.0)
    np.testing.assert_allclose(seen[-1], [4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 1.0])
    gizmo.pose = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    np.testing.assert_allclose(handle.wxyz, [0.0, 1.0, 0.0, 0.0])
    with pytest.raises(ShapeError, match=r"pose must be \(7,\)"):  # R1: the setter took any shape
        gizmo.pose = [1.0, 2.0, 3.0]
    np.testing.assert_allclose(handle.position, [0.0, 0.0, 0.0])


def test_one_drag_reaches_the_gizmo_once() -> None:
    server = FakeServer()
    seen: list[np.ndarray] = []
    gizmo = PoseGizmo(Viewer(server=server), "once", pose=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], on_change=seen.append)
    handle = server.scene.nodes["/tools/once"]
    handle.position, handle.wxyz = np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])
    assert len(handle.callbacks) == 1  # viser's on_update already fires on the drag's end phase
    handle.fire()
    assert gizmo.wait(timeout=5.0) and len(seen) == 1
    gizmo.remove()


def test_a_failing_gizmo_callback_stops_the_worker_and_raises_on_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    reported: list[BaseException] = []
    monkeypatch.setattr(threading, "excepthook", lambda hook: reported.append(hook.exc_value))

    def broken(pose: np.ndarray) -> None:
        raise ArithmeticError("IK diverged")

    server = FakeServer()
    gizmo = PoseGizmo(Viewer(server=server), "target", pose=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], on_change=broken)
    handle = server.scene.nodes["/tools/target"]
    handle.position, handle.wxyz = np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])
    handle.fire()
    for _ in range(2):  # the gizmo stays stopped: every wait raises
        with pytest.raises(ArithmeticError, match="IK diverged"):
            gizmo.wait(timeout=5.0)
    assert [type(error) for error in reported] == [ArithmeticError]  # printed by the default hook, not swallowed


def test_remove_finishes_the_running_callback_drops_the_pending_pose_and_releases_waiters() -> None:
    server = FakeServer()
    running, release, calls = threading.Event(), threading.Event(), []

    def slow(pose: np.ndarray) -> None:
        running.set()
        release.wait(5.0)
        calls.append(pose)

    gizmo = PoseGizmo(Viewer(server=server), "removed", pose=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], on_change=slow)
    handle = server.scene.nodes["/tools/removed"]
    handle.position, handle.wxyz = np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])
    handle.fire()
    assert running.wait(5.0)
    handle.fire()  # pending behind the running callback
    waiter = threading.Thread(target=gizmo.wait, daemon=True)  # no timeout: returns once the gizmo is removed
    waiter.start()
    threading.Timer(0.1, release.set).start()
    gizmo.remove()
    assert len(calls) == 1  # the running callback finished before remove returned; the pending pose was dropped
    waiter.join(5.0)
    assert not waiter.is_alive()
    handle.fire()  # a late drag reaches no one
    assert gizmo.wait(timeout=5.0) and len(calls) == 1
    assert not any(thread.name == "gizmo:removed" for thread in threading.enumerate())


def test_gizmo_drops_stale_poses_and_calls_back_off_the_caller_thread() -> None:
    server = FakeServer()
    viewer = Viewer(server=server)
    seen: list[tuple[np.ndarray, threading.Thread]] = []

    def slow(pose: np.ndarray) -> None:
        time.sleep(0.05)
        seen.append((pose, threading.current_thread()))

    gizmo = PoseGizmo(viewer, "target", pose=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], on_change=slow)
    handle = server.scene.nodes["/tools/target"]
    handle.wxyz = np.array([1.0, 0.0, 0.0, 0.0])
    for step in range(50):
        handle.position = np.array([float(step), 0.0, 0.0])
        handle.fire()
    assert gizmo.wait(timeout=5.0)
    assert 1 <= len(seen) <= 4
    np.testing.assert_allclose(seen[-1][0][:3], [49.0, 0.0, 0.0])
    assert all(thread is not threading.main_thread() for _, thread in seen)
    gizmo.remove()


def test_ray_hit_sphere_and_rect_selection() -> None:
    vertices = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [-5.0, 5.0, 5.0]])
    faces = np.array([[0, 1, 2]])
    hit = ray_hit(np.array([0.2, 0.2, -1.0]), np.array([0.0, 0.0, 1.0]), vertices=vertices, faces=faces)
    assert hit is not None
    np.testing.assert_allclose(hit[0], [0.2, 0.2, 1.0])
    assert hit[1] == 0
    assert ray_hit(np.array([0.2, 0.2, 2.0]), np.array([0.0, 0.0, 1.0]), vertices=vertices, faces=faces) is None
    assert vertices_in_sphere(vertices, np.zeros(3), radius=1.5).tolist() == [0, 1, 2]

    # Camera at the origin looking down +z (identity orientation): the triangle sits in front, the far
    # vertex off-centre; a marquee around the image centre selects the first three vertices only.
    camera = CameraFrustum(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]), fov=np.pi / 2, aspect=1.0)
    screen, depth = project(vertices, camera)
    np.testing.assert_allclose(screen[0], [0.5, 0.5])
    np.testing.assert_allclose(screen[1], [1.0, 0.5])
    assert (depth > 0).all()
    assert vertices_in_rect(
        vertices, camera, rect_min=np.array([0.4, 0.4]), rect_max=np.array([1.0, 1.0])
    ).tolist() == [0, 1, 2]


def test_a_camera_frustum_is_a_pose_in_the_library_order_also_from_a_browser_camera() -> None:
    """viser's camera is scalar-first ``wxyz``; the frustum holds ``[x, y, z, qx, qy, qz, qw]`` like every pose."""
    half = np.sqrt(0.5)
    browser = SimpleNamespace(position=(1.0, 2.0, 3.0), wxyz=(half, 0.0, half, 0.0), fov=1.0, aspect=1.5)  # +z to +x
    camera = CameraFrustum.from_camera(browser)
    np.testing.assert_allclose(camera.pose, [1.0, 2.0, 3.0, 0.0, half, 0.0, half])
    screen, depth = project(np.array([[3.0, 2.0, 3.0]]), camera)  # two metres straight ahead
    np.testing.assert_allclose(screen, [[0.5, 0.5]], atol=1e-12)
    np.testing.assert_allclose(depth, [2.0])
    assert repr(camera) == "CameraFrustum(pose=[1.0, 2.0, 3.0, 0.0, 0.707, 0.0, 0.707], fov=1.000, aspect=1.500)"
    with pytest.raises(ShapeError, match=r"pose must be \(7,\)"):
        CameraFrustum(np.zeros(4), fov=1.0, aspect=1.0)


def test_picking_on_a_placed_tag_takes_the_viewer_pose_of_that_tag() -> None:
    """R1's probe: a world click ray at vertex 0 as drawn, on a layer whose tag is turned a quarter about +z, picked
    vertex 1 from the layer's own arrays."""
    viewer = Viewer(server=FakeServer(), ground=False)
    turn = [0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)]
    layer = viewer.add(PointsLayer("p", np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])), pose=[0.0, 0.0, 0.0, *turn])
    pose = viewer.pose("p")
    np.testing.assert_allclose(pose, [0.0, 0.0, 0.0, *turn])
    drawn = np.array([0.0, 1.0, 0.0])  # where vertex 0 appears in the world
    origin = np.array([0.0, 1.0, 5.0])
    assert nearest_on_ray(layer.points[0], origin, direction=drawn - origin, radius=0.05) == 1  # as R1 found it
    assert nearest_on_ray(layer.points[0], origin, direction=drawn - origin, radius=0.05, pose=pose) == 0
    # A triangle on the plane x = 1 of the layer, drawn on the plane y = 1; a ray along -y meets it only when placed.
    triangle, faces = np.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 0.0, 1.0]]), np.array([[0, 1, 2]])
    start, down = np.array([-0.5, 3.0, 0.2]), np.array([0.0, -1.0, 0.0])
    assert ray_hit(start, down, vertices=triangle, faces=faces) is None
    hit = ray_hit(start, down, vertices=triangle, faces=faces, pose=pose)
    assert hit is not None and hit[1] == 0
    np.testing.assert_allclose(hit[0], [-0.5, 1.0, 0.2], atol=1e-12)  # the point in the world
    assert vertices_in_sphere(layer.points[0], drawn, radius=0.1, pose=pose).tolist() == [0]
    camera = CameraFrustum(np.array([0.0, 1.0, -2.0, 0.0, 0.0, 0.0, 1.0]), fov=np.pi / 4, aspect=1.0)  # looks at +z
    np.testing.assert_allclose(project(layer.points[0], camera, pose=pose)[0][0], [0.5, 0.5], atol=1e-12)
    centre = np.array([0.45, 0.45]), np.array([0.55, 0.55])
    assert vertices_in_rect(layer.points[0], camera, rect_min=centre[0], rect_max=centre[1], pose=pose).tolist() == [0]
    viewer.add(PointsLayer("q", np.zeros((1, 3))))
    np.testing.assert_allclose(viewer.pose("q"), [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])  # never placed: the identity
    with pytest.raises(UnknownNameError):
        viewer.pose("nowhere")
    with pytest.raises(ShapeError, match=r"pose must be \(7,\)"):
        nearest_on_ray(layer.points[0], origin, direction=drawn - origin, radius=0.05, pose=np.zeros(3))
