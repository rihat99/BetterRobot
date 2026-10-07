"""Screenshots of the live viser scene through a headless Chromium, and writers for PNG / GIF / WebM clips.

The browser is found via ``$BETTER_ROBOT_CHROME``, else ``chromium``, ``google-chrome`` or ``chrome-headless-shell``
on ``PATH``. A machine without ``libasound.so.2`` (common on headless servers) makes Chromium exit at start; put a
copy of the library on ``LD_LIBRARY_PATH`` (e.g. extracted from the ``libasound2`` package) before recording.
``scripts/chrome_headless.sh`` in the repository is such a wrapper for development.
"""

from __future__ import annotations

import io
import math
import os
import shutil
import subprocess
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

_CHROME_NAMES = ("chromium", "google-chrome", "chrome-headless-shell")
FOV = math.radians(50.0)  # vertical field of view `fit_camera` assumes and `HeadlessClient.set_camera` sets


def find_chrome() -> str:
    """Path of a Chromium binary: ``$BETTER_ROBOT_CHROME``, else the first of ``chromium``/``google-chrome``/``chrome-headless-shell`` on ``PATH``."""
    path = os.environ.get("BETTER_ROBOT_CHROME") or next(filter(None, map(shutil.which, _CHROME_NAMES)), None)
    if path is None:
        raise RuntimeError(
            "Viewer.record needs a Chromium binary: install chromium (or google-chrome / chrome-headless-shell, e.g. "
            "`npx playwright install chromium-headless-shell`) and put it on PATH or set BETTER_ROBOT_CHROME"
        )
    return path


def find_ffmpeg() -> str:
    """Path of ``ffmpeg``: ``$FFMPEG``, else ``ffmpeg`` on ``PATH``."""
    path = os.environ.get("FFMPEG") or shutil.which("ffmpeg")
    if path is None:
        raise RuntimeError("writing .webm needs ffmpeg: install it, or set FFMPEG to its path; or record .png/.gif")
    return path


def fit_camera(
    points: np.ndarray,
    *,
    up: Sequence[float] = (0.0, 0.0, 1.0),
    azimuth: float = math.radians(35.0),
    elevation: float = math.radians(25.0),
    fov: float = FOV,
    aspect: float = 16.0 / 9.0,
    fill: float = 0.7,
) -> tuple[np.ndarray, np.ndarray]:
    """``(position, look_at)`` of a 3/4 view where ``points (N, 3)`` span ``fill`` of the frame height (or width).

    ``up`` is the world up axis, ``fov`` the vertical field of view and ``aspect`` width / height. The projected
    bounding box is fitted, not the bounding sphere, so tall thin subjects are not framed as if they were wide.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    up_axis = np.asarray(up, dtype=np.float64) / np.linalg.norm(up)
    side = np.cross(np.array([0.0, 0.0, 1.0]) if abs(up_axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0]), up_axis)
    side /= np.linalg.norm(side)
    forward = np.cross(up_axis, side)
    direction = (
        math.cos(elevation) * math.cos(azimuth) * side
        + math.cos(elevation) * math.sin(azimuth) * forward
        + math.sin(elevation) * up_axis
    )
    right = np.cross(-direction, up_axis)
    right /= np.linalg.norm(right)
    screen_up = np.cross(right, -direction)
    # Start centred on the box of the points seen along the view direction; back off until every point projects
    # inside the fill rectangle (depth >= |x| / tan_x and |y| / tan_y); then re-centre on the perspective box.
    x, y = points @ right, points @ screen_up
    look_at = (x.min() + x.max()) / 2 * right + (y.min() + y.max()) / 2 * screen_up
    look_at = look_at + float((points @ direction).mean()) * direction
    tan_y = fill * math.tan(fov / 2.0)
    tan_x = tan_y * aspect
    for _ in range(8):
        offset = points - look_at
        depth_needed = np.maximum(np.abs(offset @ screen_up) / tan_y, np.abs(offset @ right) / tan_x)
        distance = max(float((offset @ direction + depth_needed).max()), 1e-3)
        depth = distance - offset @ direction
        screen_x, screen_y = offset @ right / depth, offset @ screen_up / depth
        look_at = look_at + distance * (
            (screen_x.max() + screen_x.min()) / 2 * right + (screen_y.max() + screen_y.min()) / 2 * screen_up
        )
    return look_at + distance * direction, look_at


class HeadlessClient:
    """A headless Chromium connected to a ``Viewer``; ``render`` screenshots what that browser shows.

    Use as a context manager: entering launches the browser and waits for it to connect, exiting kills it.
    ``gpu=True`` asks Chromium for EGL/GPU rendering; the default is software (SwiftShader).
    """

    def __init__(
        self, viewer: Any, *, size: tuple[int, int] = (960, 540), gpu: bool = False, timeout: float = 30.0
    ) -> None:
        self.viewer, self.size, self.gpu, self.timeout = viewer, size, gpu, timeout
        self.client: Any = None
        self._process: subprocess.Popen | None = None

    def __enter__(self) -> HeadlessClient:
        width, height = self.size
        flags = (
            ["--use-angle=gl-egl", "--enable-gpu", "--ignore-gpu-blocklist"]
            if self.gpu
            else ["--use-angle=swiftshader", "--enable-unsafe-swiftshader"]
        )
        command = [
            find_chrome(),
            "--headless",
            "--no-sandbox",
            f"--window-size={width},{height}",
            "--remote-debugging-port=0",
            *flags,
            self.viewer.url,
        ]
        known = set(self.viewer.server.get_clients())
        self._process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.monotonic() + self.timeout
        while self.client is None:
            new = [client for key, client in self.viewer.server.get_clients().items() if key not in known]
            if new:
                self.client = new[0]
            elif self._process.poll() is not None or time.monotonic() > deadline:
                self.__exit__(None, None, None)
                raise RuntimeError(
                    f"headless Chromium did not connect to {self.viewer.url} within {self.timeout:.0f} s "
                    f"(exit code {self._process.returncode}); if it exits at once, see better_robot.viewer.capture"
                )
            else:
                time.sleep(0.05)
        return self

    def __exit__(self, *exc: object) -> None:
        if self._process is not None and self._process.poll() is None:
            self._process.kill()
            self._process.wait()

    def render(self, height: int, width: int) -> np.ndarray:
        """Screenshot ``(H, W, 3)`` uint8 of the scene as the browser shows it now."""
        image = np.asarray(self.client.get_render(height, width, transport_format="png"))
        if image.shape[-1] == 4:  # the canvas is transparent where nothing is drawn: composite over white
            alpha = image[..., 3:].astype(np.float32) / 255.0
            image = (image[..., :3] * alpha + 255.0 * (1.0 - alpha)).round().astype(np.uint8)
        return image

    def set_camera(self, position: Sequence[float], look_at: Sequence[float], up: Sequence[float]) -> None:
        """Place the browser's camera; call it after the scene has arrived (first ``render``)."""
        with self.client.atomic():
            self.client.camera.fov = FOV
            self.client.camera.up_direction = tuple(map(float, up))
            self.client.camera.position = tuple(map(float, position))
            self.client.camera.look_at = tuple(map(float, look_at))


def write_frames(frames: Sequence[np.ndarray], path: str | Path, *, fps: float) -> Path:
    """Write ``(H, W, 3)`` uint8 frames to ``*.webm`` (ffmpeg, VP8), ``*.gif`` (Pillow), or PNGs.

    PNG output: ``name.png`` writes ``name_000.png, name_001.png, ...``; a path without a suffix is a directory
    that receives ``frame_000.png, ...``. Returns ``path``.
    """
    from PIL import Image  # noqa: PLC0415 - optional at the boundary

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".gif":
        images = [Image.fromarray(frame) for frame in frames]
        images[0].save(path, save_all=True, append_images=images[1:], duration=round(1000 / fps), loop=0)
    elif suffix == ".webm":
        jpegs = []
        for frame in frames:  # JPEG through image2pipe: also works with minimal builds (e.g. Playwright's ffmpeg)
            buffer = io.BytesIO()
            Image.fromarray(frame).save(buffer, format="JPEG", quality=95)
            jpegs.append(buffer.getvalue())
        subprocess.run(
            [
                find_ffmpeg(),
                "-y",
                "-loglevel",
                "error",
                "-f",
                "image2pipe",
                "-c:v",
                "mjpeg",
                "-framerate",
                str(fps),
                "-i",
                "pipe:0",
                "-c:v",
                "libvpx",
                "-b:v",
                "0",
                "-crf",
                "30",
                "-deadline",
                "good",
                "-cpu-used",
                "2",
                "-auto-alt-ref",
                "0",
                "-an",
                str(path),
            ],
            input=b"".join(jpegs),
            check=True,
        )
    elif suffix in ("", ".png"):
        folder, stem = (path, "frame") if suffix == "" else (path.parent, path.stem)
        folder.mkdir(parents=True, exist_ok=True)
        for k, frame in enumerate(frames):
            Image.fromarray(frame).save(folder / f"{stem}_{k:03d}.png")
    else:
        raise ValueError(f"record writes .webm, .gif, .png or a directory, not {path.name!r}")
    return path
