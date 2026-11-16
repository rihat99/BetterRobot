"""``Viewer.record``: argument checks and the GPU mapping without a browser, then real clips and stills in Chromium.

The browser tests drive viser's own client through internals (``record.PROBE``); when a viser upgrade changes them,
``test_the_viser_client_still_has_what_the_recorder_drives`` fails with the viser version in its message.
"""

from __future__ import annotations

import base64
import io
import json
import math
import os
import shutil
import socket
import subprocess
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from better_robot.exceptions import ArgumentError, OptionalDependencyError
from better_robot.testing import FakeServer
from better_robot.viewer import DARK, LIGHT, PointsLayer, SpheresLayer, Theme, Viewer
from better_robot.viewer import record as recorder

# Six steps, six colours whose bright channels tell the step apart even after lighting and compression.
_STEP_COLORS = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=np.float32)
_CAMERA = (np.array([0.0, -3.0, 0.0]), np.zeros(3))  # the ball at the origin fills the frame's centre


class FakePage:
    """Stands in for the browser page: records what ``record`` asks for and returns placeholder bytes."""

    calls: list[dict] = []

    def __init__(self, viewer: Viewer, size: tuple[int, int], camera: tuple, fov: float, gpu: int | None) -> None:
        FakePage.calls.append(dict(size=size, camera=camera, fov=fov, gpu=gpu, playing=viewer.timeline.playing))

    def __enter__(self) -> FakePage:
        return self

    def __exit__(self, *exc: object) -> None:
        pass

    def record(self, steps: list[int], *, codecs: list[str], fps: float, bitrate: float) -> tuple[bytes, str]:
        FakePage.calls[-1].update(steps=steps, codecs=codecs, fps=fps, bitrate=bitrate)
        return b"frames", (codecs or ["png"])[0]


def _viewer(nsteps: int = 3) -> Viewer:
    viewer = Viewer(server=FakeServer())
    viewer.add([PointsLayer("points", np.zeros((nsteps, 2, 3), dtype=np.float32))])
    return viewer


def test_record_refuses_bad_paths_steps_sizes_and_cameras_before_a_browser_starts(tmp_path: Path) -> None:
    viewer = _viewer()
    cases = [
        (dict(path=tmp_path / "a.gif"), "writes .mp4, .webm or .png"),
        (dict(path=tmp_path / "a.webm", steps=[]), "at least one step"),
        (dict(path=tmp_path / "a.webm", steps=[0, 3]), r"steps must lie in \[0, 3\)"),
        (dict(path=tmp_path / "a.png", steps=[0, 1]), "one step"),
        (dict(path=tmp_path / "a.webm", size=(641, 360)), "both even"),
        (dict(path=tmp_path / "a.webm", camera=(np.zeros((2, 3)), np.zeros((2, 3)))), r"\(3, 3\) per step"),
    ]
    for kwargs, message in cases:
        kwargs.setdefault("camera", _CAMERA)
        with pytest.raises(ArgumentError, match=message):
            viewer.record(**kwargs)


def test_mp4_needs_ffmpeg_and_copies_the_pages_h264_stream(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("BETTER_ROBOT_FFMPEG", raising=False)
    monkeypatch.setattr(shutil, "which", lambda name: None)
    monkeypatch.setattr(recorder, "_Page", FakePage)
    FakePage.calls = []
    viewer = _viewer()
    with pytest.raises(OptionalDependencyError, match=r"ffmpeg .*record to \.webm"):
        viewer.record(tmp_path / "clip.mp4", camera=_CAMERA)
    assert viewer.record(tmp_path / "clip", camera=_CAMERA) == tmp_path / "clip.webm"
    assert (tmp_path / "clip.webm").read_bytes() == b"frames"
    monkeypatch.setenv("BETTER_ROBOT_FFMPEG", "/opt/ffmpeg")
    commands = []
    monkeypatch.setattr(subprocess, "run", lambda command, **kwargs: commands.append((command, kwargs["input"])))
    assert viewer.record(tmp_path / "clip", camera=_CAMERA, fps=25.0) == tmp_path / "clip.mp4"
    ((command, data),) = commands
    assert FakePage.calls[-1]["codecs"][0].startswith("avc1") and data == b"frames"
    assert command[command.index("-r") + 1] == "25" and command[command.index("-c") + 1] == "copy"


def test_record_pauses_playback_and_restores_the_step(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(recorder, "_Page", FakePage)
    FakePage.calls = []
    viewer = _viewer(nsteps=5)
    viewer.show(3)
    viewer.timeline.play()
    try:
        viewer.record(tmp_path / "clip.webm", size=(640, 360), camera=_CAMERA, gpu=None)
        viewer.record(tmp_path / "still.png", camera=_CAMERA, fov=math.radians(30.0))
        assert viewer.timeline.playing
    finally:
        viewer.timeline.pause()
    clip, still = FakePage.calls
    assert not clip["playing"] and clip["steps"] == [0, 1, 2, 3, 4] and clip["size"] == (640, 360)
    assert clip["fps"] == viewer.timeline.fps and clip["bitrate"] == pytest.approx(8e6 / 9)
    assert clip["codecs"] == ["vp09.00.51.08", "vp8"]
    assert still["steps"] == [viewer.timeline.step] and still["codecs"] == [] and still["fov"] == math.radians(30.0)


def test_gpu_index_maps_to_its_pci_bus_and_the_browser_is_found_on_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(recorder, "glob", lambda pattern: ["VkLayer_MESA_device_select.json"])
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    # The process list: this test's process (a child of its parent, the stand-in browser) on GPU 4 only.
    listing = f"GPU 00000000:25:00.0\n    Process ID : 1\nGPU 00000000:A5:00.0\n    Process ID : {os.getpid()}\n"
    answers = {
        "4": SimpleNamespace(returncode=0, stdout="00000000:A5:00.0\n", stderr=""),
        "PIDS": SimpleNamespace(returncode=0, stdout=listing, stderr=""),
    }
    missing = SimpleNamespace(returncode=6, stdout="No devices were found", stderr="")
    monkeypatch.setattr(subprocess, "run", lambda command, **kwargs: answers.get(command[-1], missing))
    assert recorder._dri_prime(4) == "pci-0000_a5_00_0"
    with pytest.raises(ArgumentError, match="gpu=9 is not reachable: No devices were found; pass gpu=None"):
        recorder._dri_prime(9)
    assert recorder._gpus_of(os.getppid()) == {"pci-0000_a5_00_0"}


def test_a_missing_chromium_reports_how_to_install(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BETTER_ROBOT_CHROME", raising=False)
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(OptionalDependencyError, match="BETTER_ROBOT_CHROME"):
        recorder._chrome()


# Plays a WebM in the page and reads the centre and corner pixels of each frame; the file is the data, not our writer.
_DECODE = r"""
(async (data, frames, fps) => {
  const bytes = Uint8Array.from(atob(data), (c) => c.charCodeAt(0));
  const video = document.createElement('video');
  video.muted = true;
  video.src = URL.createObjectURL(new Blob([bytes], { type: 'video/webm' }));
  await new Promise((ok, fail) => { video.onloadeddata = ok; video.onerror = () => fail(new Error('the browser cannot play the file')); });
  const canvas = Object.assign(document.createElement('canvas'), { width: video.videoWidth, height: video.videoHeight });
  const context = canvas.getContext('2d', { willReadFrequently: true });
  const pixel = (x, y) => Array.from(context.getImageData(x, y, 1, 1).data.slice(0, 3));
  const out = { width: video.videoWidth, height: video.videoHeight, duration: video.duration, centre: [], corner: [] };
  for (let k = 0; k < frames; k++) {
    video.currentTime = (k + 0.5) / fps;
    await new Promise((ok) => { video.onseeked = ok; });
    context.drawImage(video, 0, 0);
    out.centre.push(pixel(canvas.width / 2, canvas.height / 2));
    out.corner.push(pixel(2, 2));
  }
  return out;
})
"""


def _ball_viewer(theme: Theme) -> Viewer:
    """A real viewer whose ball at the origin takes the colour of the step (a per-step property, not a pose)."""
    with socket.socket() as sock:  # a free port
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    viewer = Viewer(port=port, theme=theme)
    colors = _STEP_COLORS[:, None, :]  # (T, 1, 3)
    viewer.add([SpheresLayer("ball", np.zeros((6, 1, 3)), radii=np.array([0.6]), colors=colors, opacity=1.0)])
    return viewer


def _step_of(rgb: list[int]) -> int:
    """The step whose colour has the bright channels of ``rgb``."""
    bright = tuple(bool(value > 0.5 * max(rgb)) for value in rgb)
    return [tuple(bool(c) for c in color) for color in _STEP_COLORS].index(bright)


def _page_color(theme: Theme) -> tuple[int, int, int]:
    """What the browser shows behind the scene: viser's page colour (white, or Mantine's dark 9)."""
    return (255, 255, 255) if not theme.dark else (20, 21, 23)


@pytest.mark.slow
def test_the_viser_client_still_has_what_the_recorder_drives_and_a_clip_shows_each_step_once() -> None:
    viewer = _ball_viewer(DARK)
    try:
        with recorder._Page(viewer, (320, 180), _CAMERA, math.radians(40.0), None) as page:
            missing = page.eval(recorder.PROBE)
            clip, _ = page.record(list(range(6)), codecs=list(recorder._CODECS), fps=10.0, bitrate=1e6)
            decoded = page.eval(f"{_DECODE}({json.dumps(base64.b64encode(clip).decode())}, 6, 10)")
    finally:
        viewer.server.stop()
    assert missing == [], f"viser {version('viser')} changed the client internals Viewer.record drives: {missing}"
    assert (decoded["width"], decoded["height"]) == (320, 180)
    assert round(decoded["duration"] * 10) == 6
    assert [_step_of(rgb) for rgb in decoded["centre"]] == list(range(6))
    np.testing.assert_allclose(decoded["corner"], [_page_color(DARK)] * 6, atol=6)


@pytest.mark.slow
def test_a_still_has_the_exact_size_the_step_and_the_light_page_colour(tmp_path: Path) -> None:
    viewer = _ball_viewer(LIGHT)
    try:
        viewer.show(4)
        path = viewer.record(tmp_path / "still.png", size=(321, 181), camera=_CAMERA)
    finally:
        viewer.server.stop()
    image = np.asarray(Image.open(io.BytesIO(path.read_bytes())).convert("RGB"))
    assert image.shape == (181, 321, 3)
    assert _step_of(image[90, 160].tolist()) == 4
    np.testing.assert_array_equal(image[2, 2], _page_color(LIGHT))
