"""One real viser server on loopback: a skeleton + arrows clip builds and every frame pushes fast."""

from __future__ import annotations

import socket
import time
from pathlib import Path

import numpy as np
import pytest

from better_robot.viewer.layers import ArrowsLayer, FramesLayer, GroundLayer, PointsLayer, SkeletonLayer
from better_robot.viewer.viewer import Viewer

pytest.importorskip("viser")


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_fifty_frame_scene_builds_and_pushes_every_frame(tmp_path: Path) -> None:
    viewer = Viewer(port=_free_port())
    try:
        rng = np.random.default_rng(0)
        joints = rng.normal(size=(50, 24, 3)).astype(np.float32)
        parents = np.concatenate([[-1], np.arange(23)])
        poses = np.zeros((50, 24, 7), dtype=np.float32)
        poses[..., :3] = joints
        poses[..., 6] = 1.0
        viewer.add(GroundLayer())
        viewer.add(SkeletonLayer("skeleton", joints, parents))
        viewer.add(ArrowsLayer("forces", joints[:, :4], rng.normal(size=(50, 4, 3)) * 0.2))
        viewer.add(FramesLayer("frames", poses))
        viewer.add(PointsLayer("cloud", rng.normal(size=(50, 1000, 3))), group="scan")
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
