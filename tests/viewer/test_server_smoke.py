"""One real viser server on loopback: a skeleton + arrows clip builds and every frame pushes fast."""

from __future__ import annotations

import socket
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import better_robot as br
from better_robot import collision
from better_robot.exceptions import ArgumentError

from better_robot.viewer import ArrowsLayer, FramesLayer, PointsLayer
from better_robot.viewer.layers.mesh import SpheresLayer
from better_robot.viewer.layer import Layer
from better_robot.viewer.robot import RobotVisual
from better_robot.viewer.tools import PoseGizmo
from better_robot.viewer.viewer import Viewer
from tests.support.ports import free_port
from tests.support.robots import g1_urdf


def test_fifty_frame_scene_builds_and_pushes_every_frame(tmp_path: Path) -> None:
    viewer = Viewer(port=free_port())
    try:
        rng = np.random.default_rng(0)
        joints = rng.normal(size=(50, 24, 3)).astype(np.float32)
        poses = np.zeros((50, 24, 7), dtype=np.float32)
        poses[..., :3] = joints
        poses[..., 6] = 1.0
        viewer.add([SpheresLayer("skeleton", joints, radii=np.full(24, 0.03))])
        viewer.add([ArrowsLayer("forces", joints[:, :4], vectors=rng.normal(size=(50, 4, 3)) * 0.2)])
        viewer.add([FramesLayer("frames", poses)])
        viewer.add([PointsLayer("cloud", rng.normal(size=(50, 1000, 3)))], tags="scan")
        start = time.perf_counter()
        for step in range(50):
            viewer.show(step)
        per_frame = (time.perf_counter() - start) / 50
        assert viewer.timeline.step == 49
        assert per_frame < 2e-3, f"{per_frame * 1e3:.2f} ms per frame"
        viewer.remove("scan")
        assert len(viewer.layers) == 4
        path = viewer.save(tmp_path / "clip.viser")
        assert path.stat().st_size > 50 * 4 * 24 * 4  # at least the per-frame skeleton arrays
        assert viewer.timeline.step == 49
    finally:
        viewer.server.stop()


def test_url_follows_viser_when_requested_port_is_taken() -> None:
    """viser moves to the next free port when the requested one is busy; ``Viewer.url`` must say where."""
    with socket.socket() as blocker:
        blocker.bind(("127.0.0.1", 0))
        blocker.listen()
        taken = blocker.getsockname()[1]
        viewer = Viewer(port=taken)
        try:
            assert viewer.port == viewer.server.get_port() != taken
            assert viewer.url == f"http://127.0.0.1:{viewer.port}"
        finally:
            viewer.server.stop()


def test_a_server_passed_in_gives_its_own_address() -> None:
    """``Viewer(server=s)`` reports where ``s`` listens, not the default port, which another page may hold."""
    import viser  # noqa: PLC0415 - the optional dependency this test exercises

    server = viser.ViserServer(host="127.0.0.1", port=free_port(), verbose=False)
    try:
        viewer = Viewer(server=server)
        assert (viewer.host, viewer.port) == ("127.0.0.1", server.get_port())
        assert viewer.url == f"http://127.0.0.1:{server.get_port()}"
        for given in (dict(port=8080), dict(host="10.0.0.1"), dict(host="127.0.0.1", port=9999)):  # were dropped
            with pytest.raises(ArgumentError, match="a server passed in listens where it was made"):
                Viewer(server=server, **given)
    finally:
        server.stop()


class _BrokenPush(Layer):
    """Builds a point cloud, then raises on its first push."""

    def build(self, scene: Any, root: str) -> None:
        self._track(scene.add_point_cloud(f"{root}/points", np.zeros((2, 3)), np.zeros((2, 3), np.uint8)))

    def set_step(self, step: int) -> None:
        raise RuntimeError("broken push")


@pytest.mark.slow
def test_a_layer_that_fails_on_a_real_server_leaves_the_scene_as_it_was() -> None:
    viewer = Viewer(port=free_port(), ground=False)
    try:
        kept = PointsLayer("kept", np.random.default_rng(0).normal(size=(5, 4, 3)))
        viewer.add([kept])
        nodes = set(viewer.server.scene._handle_from_node_name)
        with pytest.raises(RuntimeError, match="broken push"):
            viewer.add([PointsLayer("fine", np.zeros((3, 4, 3))), _BrokenPush("bad", 9)], tags="new")
        with pytest.raises(RuntimeError, match="broken push"):
            viewer.replace(kept, _BrokenPush("kept", 9))
        assert viewer.layers == [kept] and viewer.timeline.nsteps == 5
        assert set(viewer.server.scene._handle_from_node_name) == nodes
        viewer.show(3)
        viewer.set_fps(10.0)
        np.testing.assert_array_equal(
            viewer.server.scene._handle_from_node_name["/kept/kept/points"].points, kept.points[3]
        )
    finally:
        viewer.server.stop()


@pytest.mark.slow
def test_a_pose_gizmo_on_a_real_server_uses_no_deprecated_viser_call() -> None:
    viewer = Viewer(port=free_port())
    try:
        gizmo = PoseGizmo(viewer, "target", pose=np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0]))  # warnings are errors
        np.testing.assert_allclose(gizmo.pose, [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0])
        gizmo.remove()
    finally:
        viewer.server.stop()


def test_a_switch_from_code_moves_the_shared_control_and_no_other_robot() -> None:
    """On a real server, where viser runs a control's callbacks for a value set from code as for a browser's."""
    viewer = Viewer(port=free_port())
    try:
        model = br.load(g1_urdf())
        spheres, _ = collision.load_spheres(collision.SPHERE_SETS / "g1.json", model)
        first = viewer.add(RobotVisual(model, name="first", spheres=spheres))
        second = viewer.add(RobotVisual(model, name="second", spheres=spheres), pose=(0.0, 1.0, 0.0))
        ran: list[Any] = []  # joins the three shared controls: every change that runs the callbacks lands here
        sidebar = viewer.sidebar
        style = sidebar.dropdown("Robot", "Style", options=first.styles, value=first.style, on_change=ran.append)
        overlay = sidebar.checkbox("Robot", "Collision", value=False, on_change=ran.append)
        frames = sidebar.checkbox("Robot", "Frames", value=False, on_change=ran.append)
        first.set_style("minimal")
        first.set_collision(True)
        first.set_frames(["pelvis"])
        assert (style.value, overlay.value, frames.value) == ("minimal", True, True)
        assert ran == [] and (first.style, second.style) == ("minimal", "medium")
        first.set_frames(False)
        assert frames.value is False and ran == []
        style.value = "medium"  # as a browser's choice: every robot follows
        assert ran == ["medium"] and first.style == "medium"
    finally:
        viewer.server.stop()
