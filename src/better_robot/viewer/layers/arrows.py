"""``ArrowsLayer``: forces, contacts or axes as line or solid shafts with batched cone heads."""

from __future__ import annotations

from typing import Any

import numpy as np

from .._math import RGB, as_frames, quat_z_to, rgb_uint8, unit_cone, unit_cylinder
from ..layer import Layer


class ArrowsLayer(Layer):
    """Arrows from ``origins (T, N, 3)`` along ``vectors (T, N, 3) * scale``.

    Zero vectors draw nothing (their head scale is 0), so N stays fixed across frames; use
    them for inactive contacts (a ``(T, N)`` ``valid`` in ``Viewer.add`` does the same).
    ``colors`` is one RGB, ``(N, 3)``, or ``(T, N, 3)`` per step.

    By default a shaft is a screen-space line ``line_width`` pixels wide and the cone head shrinks on
    vectors shorter than ``head_size``. With ``shaft_radius`` (metres) the arrow is solid: a cylinder
    shaft and a cone head of fixed length ``head_size`` and radius ``head_radius`` (default
    ``head_size / 2``), the shaft filling the rest of the vector; a vector shorter than the head draws
    the head alone.
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
        shaft_radius: float | None = None,
        head_radius: float | None = None,
    ) -> None:
        self.origins = as_frames(origins, 2)
        self.vectors = as_frames(vectors, 2) * np.float32(scale)
        if self.origins.shape != self.vectors.shape:
            raise ValueError("origins and vectors must have the same shape")
        super().__init__(name, self.origins.shape[0])
        self.head_size = head_size
        self.line_width = line_width
        self.shaft_radius = shaft_radius
        self.head_radius = head_size * 0.5 if head_radius is None else head_radius
        n_arrows = self._elements = self.origins.shape[1]
        colors = np.asarray(colors, dtype=np.float32)
        self._colors = rgb_uint8(colors, (n_arrows,)) if colors.ndim <= 2 else rgb_uint8(colors)
        if self._colors.ndim == 3 and self._colors.shape != self.origins.shape:
            raise ValueError(f"per-step colors must be (T, N, 3) = {self.origins.shape}, got {self._colors.shape}")
        self._shafts: Any = None
        self._heads: Any = None

    def build(self, scene: Any, root: str) -> None:
        colors = self._colors if self._colors.ndim == 2 else self._colors[0]
        if self.shaft_radius is None:
            self._shafts = self._track(
                scene.add_line_segments(
                    f"{root}/shafts",
                    self._segments(0),
                    colors=np.repeat(colors[:, None], 2, axis=1),
                    line_width=self.line_width,
                )
            )
        else:
            vertices, faces = unit_cylinder()
            positions, wxyzs, scales = self._shaft_transforms(0)
            self._shafts = self._track(
                scene.add_batched_meshes_simple(
                    f"{root}/shafts",
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
        if self.shaft_radius is None:
            self._shafts.points = self._segments(step)
        else:
            positions, wxyzs, scales = self._shaft_transforms(step)
            self._shafts.batched_positions = positions
            self._shafts.batched_wxyzs = wxyzs
            self._shafts.batched_scales = scales
        positions, wxyzs, scales = self._head_transforms(step)
        self._heads.batched_positions = positions
        self._heads.batched_wxyzs = wxyzs
        self._heads.batched_scales = scales
        if self._colors.ndim == 3:
            if self.shaft_radius is None:
                self._shafts.colors = np.repeat(self._colors[step][:, None], 2, axis=1)
            else:
                self._shafts.batched_colors = self._colors[step]
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
        unit = vector / np.maximum(length, 1e-9)[:, None]
        if self.shaft_radius is None:
            head = np.minimum(self.head_size, length)
            radius = head * 0.5
        else:  # fixed size, hidden only on zero vectors
            head = np.where(length > 0.0, self.head_size, 0.0).astype(np.float32)
            radius = np.where(length > 0.0, self.head_radius, 0.0).astype(np.float32)
        base = self.origins[step] + vector - unit * head[:, None]
        scales = np.stack([radius, radius, head], axis=-1)
        return base, quat_z_to(vector), scales

    def _shaft_transforms(self, step: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Solid shafts: unit cylinders from each origin to its head's base, hidden when the head covers the vector."""
        vector = self._vectors(step)
        length = np.linalg.norm(vector, axis=-1)
        shaft = np.maximum(length - self.head_size, 0.0)
        radius = np.where(shaft > 0.0, self.shaft_radius, 0.0).astype(np.float32)
        unit = vector / np.maximum(length, 1e-9)[:, None]
        centre = self.origins[step] + unit * (shaft * 0.5)[:, None]
        return centre, quat_z_to(vector), np.stack([radius, radius, shaft], axis=-1).astype(np.float32)
