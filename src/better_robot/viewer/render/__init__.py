"""Offline rendering of layers without a browser: PNG steps, GIF, or MP4 when ffmpeg is on PATH.

The same ``(T, ...)`` arrays the interactive viewer pushes are turned into world triangles by
``Layer.geometry(step)`` and drawn by the torch rasterizer in ``rasterizer.py``.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..layer import Layer
from .rasterizer import Camera, rasterize

__all__ = ["Camera", "frame_camera", "record", "render_frame", "render_frames"]


def render_frame(
    layers: Sequence[Layer],
    step: int,
    camera: Camera,
    *,
    size: tuple[int, int] = (960, 540),
    background: tuple[float, float, float] = (1.0, 1.0, 1.0),
    device: torch.device | str | None = None,
) -> np.ndarray:
    """One ``(H, W, 3)`` uint8 image of ``step`` (clamped per layer) from every layer's geometry."""
    vertices, faces, colors, offset = [], [], [], 0
    for layer in layers:
        for geometry in layer.geometry(min(step, layer.nsteps - 1)):
            vertices.append(np.asarray(geometry.vertices, dtype=np.float32))
            faces.append(np.asarray(geometry.faces, dtype=np.int64) + offset)
            colors.append(np.broadcast_to(np.asarray(geometry.colors, dtype=np.float32), (len(geometry.faces), 3)))
            offset += len(geometry.vertices)
    if not faces:
        image = torch.tensor(background).expand(size[1], size[0], 3)
    else:
        image = rasterize(
            torch.from_numpy(np.concatenate(vertices)).to(device),
            torch.from_numpy(np.concatenate(faces)).to(device),
            torch.from_numpy(np.ascontiguousarray(np.concatenate(colors))).to(device),
            camera,
            size,
            background=background,
        )
    return (image.clamp(0.0, 1.0).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)


def frame_camera(layers: Sequence[Layer], steps: Iterable[int] = (0,), **options: Any) -> Camera:
    """A 3/4 camera fitting the geometry of ``layers`` over ``steps`` (layers with ``fits_camera = False`` excluded).

    ``options`` go to ``Camera.framing`` (``azimuth``, ``elevation``, ``fov``, ``margin``, ``up``).
    """
    points = [
        geometry.vertices
        for step in steps
        for layer in layers
        if layer.fits_camera
        for geometry in layer.geometry(min(step, layer.nsteps - 1))
    ]
    if not points:
        return Camera((3.0, 2.0, 2.0))
    return Camera.framing(np.concatenate(points), **options)


def render_frames(
    layers: Sequence[Layer],
    steps: Iterable[int] | None = None,
    *,
    camera: Camera | None = None,
    size: tuple[int, int] = (960, 540),
    background: tuple[float, float, float] = (1.0, 1.0, 1.0),
    device: torch.device | str | None = None,
    up: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> np.ndarray:
    """Images ``(N, H, W, 3)`` uint8 for ``steps`` (default: every step of the longest layer).

    Without ``camera`` a 3/4 view, upright along ``up``, is fitted to the geometry sampled over the clip.
    """
    if steps is None:
        steps = range(max(layer.nsteps for layer in layers))
    steps = list(steps)
    camera = camera or frame_camera(layers, steps[:: max(1, len(steps) // 8)], up=up)
    return np.stack([render_frame(layers, k, camera, size=size, background=background, device=device) for k in steps])


def record(
    layers: Sequence[Layer],
    path: str | Path,
    *,
    fps: float = 30.0,
    steps: Iterable[int] | None = None,
    camera: Camera | None = None,
    size: tuple[int, int] = (960, 540),
    background: tuple[float, float, float] = (1.0, 1.0, 1.0),
    device: torch.device | str | None = None,
    up: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> Path:
    """Render ``layers`` over ``steps`` and write them to ``path``, chosen by its suffix.

    ``.gif``; ``.mp4`` (needs ``ffmpeg`` on PATH); or ``.png``, one file per step named ``<stem>_0000.png``
    beside ``path``. ``steps`` defaults to every step of the longest layer; without ``camera`` a 3/4 view,
    upright along ``up``, is fitted to the clip. Returns the written path (the directory for PNG sequences).
    """
    from PIL import Image  # noqa: PLC0415 - optional dependency (viser ships it)

    path = Path(path)
    images = render_frames(layers, steps, camera=camera, size=size, background=background, device=device, up=up)
    pil = [Image.fromarray(image) for image in images]
    if path.suffix == ".gif":
        pil[0].save(path, save_all=True, append_images=pil[1:], duration=int(1000 / fps), loop=0)
    elif path.suffix == ".mp4":
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("MP4 recording needs ffmpeg on PATH; use .gif or .png instead")
        with tempfile.TemporaryDirectory() as folder:
            for k, image in enumerate(pil):
                image.save(Path(folder) / f"frame_{k:05d}.png")
            subprocess.run(
                [ffmpeg, "-y", "-loglevel", "error", "-framerate", str(fps), "-i", f"{folder}/frame_%05d.png",
                 "-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", str(path)],
                check=True,
            )  # fmt: skip
    elif path.suffix == ".png":
        path.parent.mkdir(parents=True, exist_ok=True)
        for k, image in enumerate(pil):
            image.save(path.with_name(f"{path.stem}_{k:04d}.png"))
        return path.parent
    else:
        raise ValueError(f"unsupported recording format {path.suffix!r}; use .gif, .mp4 or .png")
    return path
