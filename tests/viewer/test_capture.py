"""``Viewer.record`` through a headless browser: camera framing, step order and writers with a fake client."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from better_robot.viewer import capture
from better_robot.viewer.layers import GroundLayer, PointsLayer
from better_robot.viewer.viewer import Viewer

from .fake_server import FakeServer


class FakeClient:
    """Stands in for ``HeadlessClient``: each render paints the current step into the red channel."""

    calls: list[tuple] = []

    def __init__(self, viewer: Viewer, *, size: tuple[int, int], gpu: bool) -> None:
        self.viewer = viewer

    def __enter__(self) -> FakeClient:
        return self

    def __exit__(self, *exc: object) -> None:
        FakeClient.calls.append(("exit",))

    def set_camera(self, position, look_at, up) -> None:
        FakeClient.calls.append(("camera", np.asarray(position), np.asarray(look_at), tuple(up)))

    def render(self, height: int, width: int) -> np.ndarray:
        FakeClient.calls.append(("render", self.viewer.timeline.step))
        return np.full((height, width, 3), self.viewer.timeline.step, dtype=np.uint8)


def test_record_frames_the_offset_group_after_the_first_render_and_writes_pngs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(capture, "HeadlessClient", FakeClient)
    FakeClient.calls = []
    viewer = Viewer(server=FakeServer())
    points = np.zeros((3, 2, 3), dtype=np.float32)
    points[:, 1, 2] = 1.0
    viewer.add(PointsLayer("pts", points), group="g", offset=(10.0, 0.0, 0.0))
    viewer.record(tmp_path / "clip.png", size=(8, 6))
    kinds = [call[0] for call in FakeClient.calls]
    assert kinds == ["render", "camera", "render", "render", "render", "exit"]
    _, _, look_at, up = FakeClient.calls[1]
    assert look_at[0] == pytest.approx(10.0, abs=0.1) and up == viewer.theme.up_vector
    frames = [np.asarray(Image.open(tmp_path / f"clip_{k:03d}.png")) for k in range(3)]
    assert [frame.shape for frame in frames] == [(6, 8, 3)] * 3
    assert [int(frame[0, 0, 0]) for frame in frames] == [0, 1, 2]


def test_write_frames_gif_and_rejects_unknown_suffix(tmp_path: Path) -> None:
    frames = [np.full((4, 5, 3), k * 50, dtype=np.uint8) for k in range(3)]
    assert capture.write_frames(frames, tmp_path / "a.gif", fps=10).stat().st_size > 0
    with pytest.raises(ValueError, match="mp4"):
        capture.write_frames(frames, tmp_path / "a.mp4", fps=10)


def test_find_chrome_reports_how_to_install(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BETTER_ROBOT_CHROME", raising=False)
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="BETTER_ROBOT_CHROME"):
        capture.find_chrome()


@pytest.mark.slow
def test_record_screenshots_a_real_scene(tmp_path: Path) -> None:
    try:
        capture.find_chrome()
    except RuntimeError:
        pytest.skip("no Chromium binary on this machine")
    viewer = Viewer(port=int(os.environ.get("BETTER_ROBOT_TEST_PORT", "8094")))
    try:
        viewer.add(PointsLayer("pts", np.zeros((2, 1, 3), dtype=np.float32)))
        viewer.record(tmp_path / "shot.png", steps=[1], size=(64, 48))
    finally:
        viewer.server.stop()
    assert np.asarray(Image.open(tmp_path / "shot_000.png")).shape == (48, 64, 3)


def test_points_apply_group_offset_and_skip_hidden_or_pointless_layers() -> None:
    """``Viewer._points`` (the camera-fit input) is ``Layer.world_points`` shifted by the group's ``offset``."""
    viewer = Viewer(server=FakeServer())
    viewer.add(GroundLayer())
    viewer.add(PointsLayer("p", np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])), group="a", offset=(0.0, 2.0, 0.0))
    hidden = viewer.add(PointsLayer("h", np.full((1, 3), 50.0)), group="b")
    viewer.set_visible(hidden, False)
    np.testing.assert_allclose(viewer._points([0]), [[0.0, 2.0, 0.0], [1.0, 2.0, 0.0]])
