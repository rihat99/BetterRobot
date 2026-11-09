"""One real viser server on loopback: a skeleton + arrows clip builds and every frame pushes fast."""

from __future__ import annotations

import socket
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from better_robot.viewer.layers import ArrowsLayer, FramesLayer, GroundLayer, PointsLayer, SkeletonLayer
from better_robot.viewer.layer import Layer
from better_robot.viewer.viewer import Viewer


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_fifty_frame_scene_builds_and_pushes_every_frame(tmp_path: Path) -> None:
    viewer = Viewer(port=_free_port())
    try:
        rng = np.random.default_rng(0)
        joints = rng.normal(size=(50, 24, 3)).astype(np.float32)
        parent_ids = np.concatenate([[-1], np.arange(23)])
        poses = np.zeros((50, 24, 7), dtype=np.float32)
        poses[..., :3] = joints
        poses[..., 6] = 1.0
        viewer.add([GroundLayer()])
        viewer.add([SkeletonLayer("skeleton", joints, parent_ids)])
        viewer.add([ArrowsLayer("forces", joints[:, :4], rng.normal(size=(50, 4, 3)) * 0.2)])
        viewer.add([FramesLayer("frames", poses)])
        viewer.add([PointsLayer("cloud", rng.normal(size=(50, 1000, 3)))], group="scan")
        start = time.perf_counter()
        for step in range(50):
            viewer.show(step)
        per_frame = (time.perf_counter() - start) / 50
        assert viewer.timeline.step == 49
        assert per_frame < 2e-3, f"{per_frame * 1e3:.2f} ms per frame"
        viewer.remove_group("scan")
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


class _BrokenPush(Layer):
    """Builds a point cloud, then raises on its first push."""

    def build(self, scene: Any, root: str) -> None:
        self._track(scene.add_point_cloud(f"{root}/points", np.zeros((2, 3)), np.zeros((2, 3), np.uint8)))

    def set_step(self, step: int) -> None:
        raise RuntimeError("broken push")


@pytest.mark.slow
def test_a_layer_that_fails_on_a_real_server_leaves_the_scene_as_it_was() -> None:
    viewer = Viewer(port=_free_port())
    try:
        kept = PointsLayer("kept", np.random.default_rng(0).normal(size=(5, 4, 3)))
        viewer.add([kept])
        nodes = set(viewer.server.scene._handle_from_node_name)
        with pytest.raises(RuntimeError, match="broken push"):
            viewer.add([PointsLayer("fine", np.zeros((3, 4, 3))), _BrokenPush("bad", 9)], group="new")
        with pytest.raises(RuntimeError, match="broken push"):
            viewer.replace(kept, _BrokenPush("kept", 9))
        assert viewer.layers == [kept] and viewer.timeline.nsteps == 5
        assert set(viewer.server.scene._handle_from_node_name) == nodes
        viewer.show(3)
        viewer.set_fps(10.0)
        np.testing.assert_array_equal(
            viewer.server.scene._handle_from_node_name["/scene/kept/points"].points, kept.points[3]
        )
    finally:
        viewer.server.stop()
