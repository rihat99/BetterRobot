"""``ArrowsLayer``: forces, contacts or axes as line shafts with batched cone heads."""

from __future__ import annotations

from typing import Any

import numpy as np

from .._math import RGB, as_frames, quat_z_to, rgb_uint8, unit_cone
from ..layer import Layer


class ArrowsLayer(Layer):
    """Arrows from ``origins (T, N, 3)`` along ``vectors (T, N, 3) * scale``.

    Zero vectors draw nothing (their head scale is 0), so N stays fixed across frames; use
    them for inactive contacts. ``colors`` is one RGB or ``(N, 3)``.
    """

    def __init__(
        self,
        name: str,
        origins: np.ndarray,
        vectors: np.ndarray,
        *,
        scale: float = 1.0,
        colors: np.ndarray | RGB = (0.90, 0.25, 0.20),
        head_size: float = 0.03,
        line_width: float = 3.0,
    ) -> None:
        self.origins = as_frames(origins, 2)
        self.vectors = as_frames(vectors, 2) * np.float32(scale)
        if self.origins.shape != self.vectors.shape:
            raise ValueError("origins and vectors must have the same shape")
        super().__init__(name, self.origins.shape[0])
        self.head_size = head_size
        self.line_width = line_width
        n_arrows = self.origins.shape[1]
        self._colors = rgb_uint8(colors, (n_arrows,))
        self._shafts: Any = None
        self._heads: Any = None

    def build(self, scene: Any, root: str) -> None:
        self._shafts = self._track(
            scene.add_line_segments(
                f"{root}/shafts",
                self._segments(0),
                colors=np.repeat(self._colors[:, None], 2, axis=1),
                line_width=self.line_width,
            )
        )
        vertices, faces = unit_cone()
        positions, wxyzs, scales = self._head_transforms(0)
        self._heads = self._track(
            scene.add_batched_meshes_simple(
                f"{root}/heads",
                vertices,
                faces,
                batched_wxyzs=wxyzs,
                batched_positions=positions,
                batched_scales=scales,
                batched_colors=self._colors,
                lod="off",
                cast_shadow=False,
            )
        )

    def set_frame(self, frame: int) -> None:
        self._shafts.points = self._segments(frame)
        positions, wxyzs, scales = self._head_transforms(frame)
        self._heads.batched_positions = positions
        self._heads.batched_wxyzs = wxyzs
        self._heads.batched_scales = scales

    def _segments(self, frame: int) -> np.ndarray:
        tips = self.origins[frame] + self.vectors[frame]
        return np.stack([self.origins[frame], tips], axis=1)

    def _head_transforms(self, frame: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        vector = self.vectors[frame]
        length = np.linalg.norm(vector, axis=-1)
        head = np.minimum(self.head_size, length)
        unit = vector / np.maximum(length, 1e-9)[:, None]
        base = self.origins[frame] + vector - unit * head[:, None]
        scales = np.stack([head * 0.5, head * 0.5, head], axis=-1)
        return base, quat_z_to(vector), scales
