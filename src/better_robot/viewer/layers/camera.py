"""``CameraLayer``: a pinhole camera's frustum from its intrinsics, with optional per-step images."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, final, override

import numpy as np

from ...exceptions import ShapeError
from ..layer import Bounds, Layer, as_frames, box
from ..theme import LIGHT, RGB, Theme, linear_uint8


@final
class CameraLayer(Layer):
    """A camera at ``poses (T, 7)`` (camera-to-world, OpenCV: ``+z`` forward, ``+x`` right, ``+y`` down) drawn as the
    edges of its frustum, cut ``scale`` metres in front of the camera, with an optional image on that cut.

    ``intrinsics`` is the ``(3, 3)`` matrix ``K`` (``fx, fy`` focal lengths and ``cx, cy`` principal point, in pixels)
    of an image of ``size`` ``(W, H)`` pixels, so an off-centre principal point draws a skewed frustum. ``image`` is
    ``(H, W, 3)`` uint8, ``(T, H, W, 3)``, or a callable ``step -> image`` for lazy loading. ``color`` ``None`` takes
    the theme's ``text`` colour.
    """

    def __init__(
        self,
        name: str,
        poses: np.ndarray,
        *,
        intrinsics: np.ndarray,
        size: tuple[int, int],
        scale: float = 0.3,
        color: RGB | None = None,
        image: np.ndarray | Callable[[int], np.ndarray] | None = None,
    ) -> None:
        self.poses = as_frames(poses, 1, name="poses", width=7)
        super().__init__(name, self.poses.shape[0])
        matrix = np.asarray(intrinsics, dtype=np.float64)
        if matrix.shape != (3, 3):
            raise ShapeError(f"intrinsics must be the (3, 3) matrix K, got shape {matrix.shape}")
        self.intrinsics, self.size, self.scale, self.color = matrix, size, scale, color
        width, height = size
        # The image's corners (top left, top right, bottom right, bottom left) on the cut, in the camera frame.
        pixels = np.array([[0.0, 0.0, 1.0], [width, 0.0, 1.0], [width, height, 1.0], [0.0, height, 1.0]])
        self._corners = (np.linalg.solve(matrix, pixels.T).T * scale).astype(np.float32)  # (4, 3) at z = scale
        self._image = image
        self._ink = LIGHT.text
        self._frame: Any = None
        self._lines: Any = None
        self._picture: Any = None

    def image_at(self, step: int) -> np.ndarray | None:
        """Image shown on the frustum at ``step``: ``(H, W, 3)``, or ``None`` when the layer has no image."""
        if self._image is None:
            return None
        if callable(self._image):
            return self._image(step)
        return self._image if self._image.ndim == 3 else self._image[step]

    @override
    def apply_theme(self, theme: Theme) -> None:
        self._ink = theme.text
        if self.color is None and self._lines is not None:
            self._lines.colors = linear_uint8(self._ink)

    @override
    def build(self, scene: Any, root: str) -> None:
        self._frame = self._track(scene.add_frame(f"{root}/camera", show_axes=False))
        apex = np.zeros((4, 3), dtype=np.float32)
        ring = np.roll(self._corners, -1, axis=0)
        edges = np.concatenate([np.stack([apex, self._corners], 1), np.stack([self._corners, ring], 1)])  # (8, 2, 3)
        self._lines = self._track(
            scene.add_line_segments(
                f"{root}/camera/edges",
                edges,
                linear_uint8(self.color or self._ink),
                thickness=2.0,
                thickness_units="screen",
            )
        )
        image = self.image_at(0)
        if image is not None:
            (left, top, depth), (right, bottom, _) = self._corners[0], self._corners[2]
            self._picture = self._track(
                scene.add_image(
                    f"{root}/camera/image",
                    image,
                    float(right - left),
                    float(bottom - top),
                    position=((left + right) / 2.0, (top + bottom) / 2.0, depth),
                    cast_shadow=False,
                    receive_shadow=False,
                )
            )

    @override
    def bounds(self, step: int) -> Bounds | None:
        position, reach = self.poses[step, :3], np.linalg.norm(self._corners, axis=1).max()  # a ball around the frustum
        return box(np.stack([position - reach, position + reach]))

    @override
    def set_step(self, step: int) -> None:
        self._frame.position = self.poses[step, :3]
        self._frame.wxyz = self.poses[step, [6, 3, 4, 5]]  # viser's scalar-first quaternion
        if self._image is not None and (callable(self._image) or self._image.ndim == 4):
            self._picture.image = self.image_at(step)
