"""``ArrowsLayer``: forces, contacts or axes as line shafts with batched cone heads."""

from __future__ import annotations

from typing import Any

import numpy as np

from .._math import RGB, as_frames, quat_z_to, rgb_uint8, unit_cone
from ..layer import Layer


class ArrowsLayer(Layer):
    """Arrows from ``origins (T, N, 3)`` along ``vectors (T, N, 3) * scale``.

    Zero vectors draw nothing (their head scale is 0), so N stays fixed across frames; use
    them for inactive contacts (a ``(T, N)`` ``valid`` in ``Viewer.add`` does the same).
    ``colors`` is one RGB, ``(N, 3)``, or ``(T, N, 3)`` per step.
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
        n_arrows = self._elements = self.origins.shape[1]
        colors = np.asarray(colors, dtype=np.float32)
        self._colors = rgb_uint8(colors, (n_arrows,)) if colors.ndim <= 2 else rgb_uint8(colors)
        if self._colors.ndim == 3 and self._colors.shape != self.origins.shape:
            raise ValueError(f"per-step colors must be (T, N, 3) = {self.origins.shape}, got {self._colors.shape}")
        self._shafts: Any = None
        self._heads: Any = None

    def build(self, scene: Any, root: str) -> None:
        colors = self._colors if self._colors.ndim == 2 else self._colors[0]
        self._shafts = self._track(
            scene.add_line_segments(
                f"{root}/shafts",
                self._segments(0),
                colors=np.repeat(colors[:, None], 2, axis=1),
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
                batched_colors=colors,
                lod="off",
                cast_shadow=False,
            )
        )

    def set_step(self, step: int) -> None:
        self._shafts.points = self._segments(step)
        positions, wxyzs, scales = self._head_transforms(step)
        self._heads.batched_positions = positions
        self._heads.batched_wxyzs = wxyzs
        self._heads.batched_scales = scales
        if self._colors.ndim == 3:
            self._shafts.colors = np.repeat(self._colors[step][:, None], 2, axis=1)
            self._heads.batched_colors = self._colors[step]

    def world_points(self, step: int) -> np.ndarray | None:
        """World positions ``(N, 3)`` drawn at ``step``, for camera fitting."""
        origins, tips = self.origins[step], self.origins[step] + self.vectors[step]
        if self.valid is not None and self.valid.ndim == 2:
            origins, tips = origins[self.valid[step]], tips[self.valid[step]]
        return np.concatenate([origins, tips])

    def _vectors(self, step: int) -> np.ndarray:
        """Vectors ``(N, 3)`` drawn at ``step``: zero where a ``(T, N)`` ``valid`` marks the arrow invalid."""
        if self.valid is None or self.valid.ndim == 1:
            return self.vectors[step]
        return self.vectors[step] * self.valid[step][:, None]

    def _segments(self, step: int) -> np.ndarray:
        tips = self.origins[step] + self._vectors(step)
        return np.stack([self.origins[step], tips], axis=1)

    def _head_transforms(self, step: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        vector = self._vectors(step)
        length = np.linalg.norm(vector, axis=-1)
        head = np.minimum(self.head_size, length)
        unit = vector / np.maximum(length, 1e-9)[:, None]
        base = self.origins[step] + vector - unit * head[:, None]
        scales = np.stack([head * 0.5, head * 0.5, head], axis=-1)
        return base, quat_z_to(vector), scales
