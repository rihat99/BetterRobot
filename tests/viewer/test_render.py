"""Offline rasterizer: coverage, occlusion, and recording through the layer geometry hooks."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from better_robot.viewer import ArrowsLayer, GroundLayer, SkeletonLayer
from better_robot.viewer.render import Camera, frame_camera, record, render_frame, render_frames
from better_robot.viewer.render.rasterizer import rasterize


def _camera_on_z() -> Camera:
    # Looking from z = -2 towards the origin with +y as up: screen x = world x, screen y = -world y.
    return Camera(position=(0.0, 0.0, -2.0), target=(0.0, 0.0, 0.0), up=(0.0, 1.0, 0.0), fov=math.radians(90.0))


def test_triangle_coverage_and_depth_ordering() -> None:
    # fov 90° at distance 2: the image spans world y in [-2, 2]; a 64 px high image → 16 px per metre.
    vertices = torch.tensor(
        [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0],  # far quad (z = 0)
         [-0.5, -0.5, -1.0], [0.5, -0.5, -1.0], [0.5, 0.5, -1.0], [-0.5, 0.5, -1.0]],  # near quad (z = -1)
    )  # fmt: skip
    faces = torch.tensor([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]])
    colors = torch.tensor([[1.0, 0.0, 0.0]] * 2 + [[0.0, 0.0, 1.0]] * 2)
    image = rasterize(vertices, faces, colors, _camera_on_z(), (64, 64), background=(0.0, 0.0, 0.0))
    assert image.shape == (64, 64, 3)
    red, blue = image[..., 0] > 0.3, image[..., 2] > 0.3
    # The far quad covers [-1, 1]² → 32 px square; the near one is 1 m away, spans [-0.5, 0.5]² at z = -1
    # which projects to the same 32 px square and must hide the red one completely.
    assert not red.any() and blue.sum() == 32 * 32
    assert blue[16:48, 16:48].all() and not blue[:16].any() and not blue[48:].any()


def test_layers_render_through_geometry_and_record_writes_files(tmp_path: Path) -> None:
    joints = np.array([[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], [[0.0, 0.0, 0.0], [1.0, 0.0, 1.0]]], dtype=np.float32)
    layers = [
        GroundLayer(size=4.0),
        SkeletonLayer("s", joints, [-1, 0], radius=0.1),
        ArrowsLayer("a", joints[:, :1], joints[:, 1:] * 0.5),
    ]
    camera = frame_camera(layers)
    image = render_frame(layers, 0, camera, size=(160, 90))
    assert image.shape == (90, 160, 3) and image.dtype == np.uint8
    assert (image.reshape(-1, 3).min(axis=0) < 200).any(), "something must be drawn"
    frames = render_frames(layers, camera=camera, size=(80, 45))
    assert frames.shape == (2, 45, 80, 3) and not np.array_equal(frames[0], frames[1])
    assert record(layers, tmp_path / "clip.gif", fps=5, size=(80, 45)).stat().st_size > 0
    assert record(layers, tmp_path / "frames" / "clip.png", size=(80, 45)) == tmp_path / "frames"
    assert sorted(p.name for p in (tmp_path / "frames").iterdir()) == ["clip_0000.png", "clip_0001.png"]
