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
import threading
import time
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from better_robot.exceptions import ArgumentError, OptionalDependencyError
from better_robot.testing import FakeServer
from better_robot.testing.fake_server import FakeClient
from better_robot.viewer import (
    DARK,
    LIGHT,
    ArrowsLayer,
    LinesLayer,
    MeshLayer,
    PointsLayer,
    SpheresLayer,
    Theme,
    Viewer,
)
from better_robot.viewer import record as recorder
from better_robot.viewer.theme import rgb_uint8

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


def test_record_refuses_bad_paths_steps_sizes_and_cameras_before_a_browser_starts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(recorder, "_Page", FakePage)
    FakePage.calls = []
    viewer = _viewer()
    cases = [
        (dict(path=tmp_path / "a.gif"), "writes .mp4, .webm or .png"),
        (dict(path=tmp_path / "a.webm", steps=[]), "at least one step"),
        (dict(path=tmp_path / "a.webm", steps=[0, 3]), r"steps must lie in \[0, 3\)"),
        (dict(path=tmp_path / "a.webm", steps=[0, 1.9]), r"steps must be integers, got \[0, 1.9\]"),
        (dict(path=tmp_path / "a.png", steps=[0, 1]), "one step"),
        (dict(path=tmp_path / "a.webm", size=(641, 360)), "both even"),
        (dict(path=tmp_path / "a.webm", camera=(np.zeros((2, 3)), np.zeros((2, 3)))), r"\(3, 3\) per step"),
        (
            dict(path=tmp_path / "a.webm", camera=(np.zeros(3), np.zeros(3), np.zeros(3))),
            r"a pair \(position, look_at\), got 3 items",
        ),
        (dict(path=tmp_path / "a.webm", fps=0.0), r"fps must be positive \(None: the timeline's\), got 0.0"),
        (dict(path=tmp_path / "a.webm", fps=-5.0), "fps must be positive"),
        (dict(path=tmp_path / "a.webm", bitrate=-1.0), r"bitrate must be positive \(None: .*\), got -1.0"),
        (dict(path=tmp_path / "a.webm", fov=0.0), r"fov must lie in \(0, pi\) radians, got 0.0"),
        (dict(path=tmp_path / "a.png", fov=math.pi), r"fov must lie in \(0, pi\)"),
    ]
    for kwargs, message in cases:
        kwargs.setdefault("camera", _CAMERA)
        with pytest.raises(ArgumentError, match=message):
            viewer.record(**kwargs)
    with pytest.raises(ArgumentError, match=r"fov must lie in \(0, pi\)"):
        viewer.aim(fov=0.0)
    assert FakePage.calls == []


def test_record_without_a_camera_leaves_the_live_cameras_and_auto_aim_alone(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(recorder, "_Page", FakePage)
    FakePage.calls = []
    viewer = _viewer()
    viewer.add(SpheresLayer("ball", np.zeros((3, 1, 3)), radii=np.array([0.5])))
    framed = np.asarray(viewer.server.initial_camera.position)  # where add aimed the cameras: the view of aim()
    viewer.server.clients[0] = FakeClient()
    user = ((4.0, 5.0, 6.0), (1.0, 1.0, 1.0))  # where a user turned the browser's camera
    cameras = (viewer.server.initial_camera, viewer.server.clients[0].camera)
    for camera in cameras:
        camera.position, camera.look_at = user
    viewer.record(tmp_path / "clip.webm")
    for camera in cameras:
        assert (camera.position, camera.look_at) == user
    np.testing.assert_allclose(FakePage.calls[-1]["camera"][0], framed)
    viewer.add(SpheresLayer("far", np.full((1, 1, 3), 5.0), radii=np.array([0.5])))
    moved = np.asarray(viewer.server.clients[0].camera.position)
    np.testing.assert_allclose(moved, viewer.aim()[0])  # add still aims the cameras: record did not call aim


def test_a_failing_ffmpeg_keeps_the_encoded_video_and_names_it(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(recorder, "_Page", FakePage)
    monkeypatch.setenv("BETTER_ROBOT_FFMPEG", "/opt/ffmpeg")
    failed = SimpleNamespace(returncode=1, stderr=b"pipe:0: Invalid data found when processing input\n")
    monkeypatch.setattr(subprocess, "run", lambda command, **kwargs: failed)
    with pytest.raises(OptionalDependencyError, match=r"exited with code 1: pipe:0: Invalid data.*clip\.h264"):
        _viewer().record(tmp_path / "clip.mp4", camera=_CAMERA)
    assert (tmp_path / "clip.h264").read_bytes() == b"frames"


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
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **kwargs: commands.append((command, kwargs["input"])) or SimpleNamespace(returncode=0),
    )
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


def test_a_sidebar_button_records_while_playing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(recorder, "_Page", FakePage)
    FakePage.calls = []
    viewer = _viewer(nsteps=300)

    def on_click() -> None:
        time.sleep(0.1)  # work under the lock: the playback thread comes to wait for it
        viewer.record(tmp_path / "clip.webm", camera=_CAMERA)  # pauses playback, which waits for the thread

    viewer.sidebar.button("Scene", "Record", on_click=on_click)
    viewer.timeline.play()
    time.sleep(0.2)
    click = threading.Thread(target=viewer.server.gui.control("Record").fire, daemon=True)
    click.start()
    click.join(1.0)
    try:
        assert not click.is_alive(), "the callback waits for the playback thread, which waits for the callback's lock"
        assert not FakePage.calls[-1]["playing"] and viewer.timeline.playing
    finally:
        viewer.timeline._halt()  # never join: after a deadlock the thread would never end


def test_the_page_reaches_a_server_bound_to_every_interface_through_loopback() -> None:
    """R1's probe: a server on IPv6 loopback gave ``url == "http://::1:8125"``, which no browser opens."""
    cases = {
        "0.0.0.0": ("http://0.0.0.0:8123", "http://127.0.0.1:8123"),
        "localhost": ("http://localhost:8123", "http://localhost:8123"),
        "::1": ("http://[::1]:8123", "http://[::1]:8123"),
        "::": ("http://[::]:8123", "http://[::1]:8123"),
    }
    for host, (url, origin) in cases.items():
        viewer = Viewer(server=FakeServer(port=8123))
        viewer.host = host  # where a real server would listen; the stand-in is always on loopback
        page = recorder._Page(viewer, (320, 180), _CAMERA, 0.7, None)
        page.__exit__()  # removes the browser profile; no browser started
        assert (viewer.url, page.origin) == (url, origin)


@pytest.mark.slow
def test_a_page_that_does_not_load_names_its_address_not_the_viser_version(tmp_path: Path) -> None:
    """The unbracketed IPv6 address made Chromium show an error page, and record blamed the client internals of the
    viser version after a minute."""
    with socket.socket() as closed:  # a port nothing listens on
        closed.bind(("127.0.0.1", 0))
        port = closed.getsockname()[1]
    viewer = Viewer(server=FakeServer(port=port), ground=False)
    viewer.add(PointsLayer("p", np.zeros((1, 3))))
    started = time.monotonic()
    with pytest.raises(ArgumentError, match=rf"could not load the viewer's page from http://127.0.0.1:{port}"):
        viewer.record(tmp_path / "still.png", size=(64, 64), camera=_CAMERA)
    assert time.monotonic() - started < 30.0


@pytest.mark.slow
def test_a_page_between_two_documents_is_polled_again(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """While Chromium moves to its error page DevTools refuses to evaluate (no execution context); on a loaded host
    the full tier met that refusal instead of the address. The wait polls again and still names the address."""
    send, refused = recorder._Page._send, []

    def refuse_once(page: object, method: str, **params: object) -> dict:
        if method == "Runtime.evaluate" and not refused:
            refused.append(method)
            raise ArgumentError("Runtime.evaluate: Cannot find default execution context")
        return send(page, method, **params)

    monkeypatch.setattr(recorder._Page, "_send", refuse_once)
    with socket.socket() as closed:  # a port nothing listens on
        closed.bind(("127.0.0.1", 0))
        port = closed.getsockname()[1]
    viewer = Viewer(server=FakeServer(port=port), ground=False)
    viewer.add(PointsLayer("p", np.zeros((1, 3))))
    with pytest.raises(ArgumentError, match=rf"could not load the viewer's page from http://127.0.0.1:{port}"):
        viewer.record(tmp_path / "still.png", size=(64, 64), camera=_CAMERA)
    assert refused


@pytest.mark.slow
def test_a_page_devtools_keeps_refusing_names_devtools_answer_not_the_viser_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A DevTools that refuses every evaluation (a dead session) was reported after a minute as a viser version
    without the client internals record drives (R2 m2): at the deadline the error is DevTools' last answer."""
    send = recorder._Page._send

    def refuse(page: object, method: str, **params: object) -> dict:
        if method == "Runtime.evaluate":
            raise ArgumentError("Runtime.evaluate: Session with given id not found.")
        return send(page, method, **params)

    clock = iter(range(1_000_000))  # a second per reading: the wait's 60 s pass at once
    monkeypatch.setattr(recorder._Page, "_send", refuse)
    monkeypatch.setattr(recorder, "time", SimpleNamespace(monotonic=lambda: float(next(clock)), sleep=lambda _: None))
    with socket.socket() as closed:  # a port nothing listens on
        closed.bind(("127.0.0.1", 0))
        port = closed.getsockname()[1]
    viewer = Viewer(server=FakeServer(port=port), ground=False)
    viewer.add(PointsLayer("p", np.zeros((1, 3))))
    with pytest.raises(ArgumentError, match="evaluated for 60 s: Runtime.evaluate: Session with given id") as error:
        viewer.record(tmp_path / "still.png", size=(64, 64), camera=_CAMERA)
    assert "viser" not in str(error.value)


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
    """The step whose colour has the bright channels of ``rgb``; the theme's lights lift the dark ones."""
    bright = tuple(bool(value > (min(rgb) + max(rgb)) / 2) for value in rgb)
    return [tuple(bool(c) for c in color) for color in _STEP_COLORS].index(bright)


def _backdrop_top(theme: Theme) -> np.ndarray:
    """What a frame's top rows show behind the scene: the top colour of the theme's backdrop gradient."""
    return rgb_uint8(theme.backdrop[0])


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
    np.testing.assert_allclose(decoded["corner"], [_backdrop_top(DARK)] * 6, atol=6)


@pytest.mark.slow
def test_a_still_has_the_exact_size_the_step_and_the_light_backdrop(tmp_path: Path) -> None:
    viewer = _ball_viewer(LIGHT)
    try:
        viewer.show(4)
        path = viewer.record(tmp_path / "still.png", size=(321, 181), camera=_CAMERA)
    finally:
        viewer.server.stop()
    image = np.asarray(Image.open(io.BytesIO(path.read_bytes())).convert("RGB"))
    assert image.shape == (181, 321, 3)
    assert _step_of(image[90, 160].tolist()) == 4
    np.testing.assert_array_equal(image[2, 2], _backdrop_top(LIGHT))


@pytest.mark.slow
def test_every_primitive_draws_the_srgb_colour_it_was_given(tmp_path: Path) -> None:
    """One patch per viser call that carries a colour, all given the same colour, seen along ``y`` from 40 m.

    viser draws a plain mesh's ``color`` as sRGB: it is the reference under the same light for the calls that read
    linear values (batched meshes, arrows, glTF vertex colours; 6 levels). The unlit ones are held to the colour
    itself: a point within 8 levels, a line within 12, because the page's filmic tone mapping bends it. A colour sent
    without ``theme.linear_uint8`` is off by about 100 levels here.
    """
    import trimesh  # noqa: PLC0415 - viser's own dependency

    color = (0.2, 0.45, 0.85)
    with socket.socket() as sock:  # a free port
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    viewer = Viewer(port=port, ground=False)
    quad = np.array([[-0.3, 0, -0.3], [0.3, 0, -0.3], [0.3, 0, 0.3], [-0.3, 0, 0.3]], dtype=np.float32)
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    ball = trimesh.creation.icosphere(subdivisions=4, radius=0.3)
    names = ["quad", "glb", "ball", "sphere", "arrow", "line", "point"]
    at = {name: np.array([0.8 * k, 0.0, 0.0], dtype=np.float32) for k, name in enumerate(names)}
    viewer.add(
        [
            MeshLayer("quad", quad + at["quad"], faces=faces, color=color),
            MeshLayer("glb", quad + at["glb"], faces=faces, colors=np.tile(color, (4, 1))),
            MeshLayer("ball", ball.vertices.astype(np.float32) + at["ball"], faces=ball.faces, color=color),
            SpheresLayer("sphere", at["sphere"][None], radii=np.array([0.3]), colors=color, opacity=1.0),
            ArrowsLayer(
                "arrow",
                at["arrow"][None] - (0.0, 0.0, 0.4),
                vectors=np.array([[0.0, 0.0, 0.8]]),
                colors=color,
                shaft_radius=0.2,
                head_radius=0.3,
                head_length=0.2,
            ),
            LinesLayer(
                "line", np.stack([at["line"] - (0.3, 0, 0), at["line"] + (0.3, 0, 0)])[None], colors=color, width=40.0
            ),
            PointsLayer("point", at["point"][None], colors=color, point_size=0.5, point_shape="square"),
        ]
    )
    width, height, fov = 800, 250, math.radians(2.6)
    try:
        path = viewer.record(
            tmp_path / "patches.png", size=(width, height), camera=((2.4, -40.0, 0.0), (2.4, 0.0, 0.0)), fov=fov
        )
    finally:
        viewer.server.stop()
    image = np.asarray(Image.open(path).convert("RGB")).astype(int)
    per_metre = height / (2.0 * 40.0 * math.tan(fov / 2.0))
    drawn = {}
    for name, place in at.items():
        column, row = round(width / 2 + (place[0] - 2.4) * per_metre), height // 2
        drawn[name] = np.median(image[row - 2 : row + 3, column - 2 : column + 3].reshape(-1, 3), axis=0)
    given = rgb_uint8(color)
    np.testing.assert_allclose(drawn["glb"], drawn["quad"], atol=6)
    np.testing.assert_allclose(drawn["sphere"], drawn["ball"], atol=6)
    np.testing.assert_allclose(drawn["arrow"], drawn["ball"], atol=6)
    np.testing.assert_allclose(drawn["point"], given, atol=8)
    np.testing.assert_allclose(drawn["line"], given, atol=12)
