"""``ArrowsLayer``: forces, contacts or axes as viser's solid arrows."""

from __future__ import annotations

from typing import Any, final, override

import numpy as np

from ...exceptions import ShapeError
from ..layer import Bounds, Layer, as_frames, box
from ..theme import RGB, rgb_uint8


@final
class ArrowsLayer(Layer):
    """Arrows from ``origins (T, N, 3)`` along ``vectors (T, N, 3) * scale``, drawn by viser's ``add_arrows``.

    A cylinder shaft of ``shaft_radius`` and a cone head of ``head_radius`` and ``head_length`` (metres, the same for
    every arrow); an arrow shorter than ``head_length`` is all head. An arrow is drawn only when its scaled length
    exceeds ``min_length`` and a ``(T, N)`` ``valid`` of ``Viewer.add`` keeps it: the others are left out of the step
    (use zero vectors for inactive contacts). ``colors`` is one RGB, ``(N, 3)``, or ``(T, N, 3)`` per step.
    """

    def __init__(
        self,
        name: str,
        origins: np.ndarray,
        *,
        vectors: np.ndarray,
        scale: float = 1.0,
        colors: np.ndarray | RGB = (0.90, 0.25, 0.20),
        shaft_radius: float = 0.006,
        head_radius: float = 0.016,
        head_length: float = 0.04,
        min_length: float = 1e-6,
    ) -> None:
        self.origins = as_frames(origins, 2, name="origins", width=3)
        self.vectors = as_frames(vectors, 2, name="vectors", width=3) * np.float32(scale)
        if self.origins.shape != self.vectors.shape:
            raise ShapeError(
                f"origins and vectors must have the same shape, got {self.origins.shape} and {self.vectors.shape}"
            )
        super().__init__(name, self.origins.shape[0])
        self.shaft_radius, self.head_radius, self.head_length = shaft_radius, head_radius, head_length
        self.min_length = min_length
        n_arrows = self._elements = self.origins.shape[1]
        colors = np.asarray(colors, dtype=np.float32)
        self._colors = rgb_uint8(colors, (n_arrows,)) if colors.ndim <= 2 else rgb_uint8(colors)
        if self._colors.ndim == 3 and self._colors.shape != self.origins.shape:
            raise ShapeError(f"per-step colors must be (T, N, 3) = {self.origins.shape}, got {self._colors.shape}")
        self._arrows: Any = None

    @override
    def build(self, scene: Any, root: str) -> None:
        points, colors = self._shown(0)
        self._arrows = self._track(
            scene.add_arrows(
                f"{root}/arrows",
                points,
                colors,
                shaft_radius=self.shaft_radius,
                head_radius=self.head_radius,
                head_length=self.head_length,
            )
        )

    @override
    def set_step(self, step: int) -> None:
        # The count of arrows shown changes from step to step, so the colours go with the points.
        self._arrows.points, self._arrows.colors = self._shown(step)

    @override
    def bounds(self, step: int) -> Bounds | None:
        origins, tips = self.origins[step], self.origins[step] + self.vectors[step]
        if self.valid is not None and self.valid.ndim == 2:
            origins, tips = origins[self.valid[step]], tips[self.valid[step]]
        return box(np.concatenate([origins, tips]))

    def _shown(self, step: int) -> tuple[np.ndarray, np.ndarray]:
        """The arrows drawn at ``step``: ``points (M, 2, 3)`` (tail, tip) and ``colors (M, 3)`` uint8, without the
        ones not longer than ``min_length`` and the ones a ``(T, N)`` ``valid`` marks invalid."""
        origins, vectors = self.origins[step], self.vectors[step]
        shown = np.linalg.norm(vectors, axis=-1) > self.min_length
        if self.valid is not None and self.valid.ndim == 2:
            shown &= self.valid[step]
        colors = self._colors if self._colors.ndim == 2 else self._colors[step]
        return np.stack([origins[shown], origins[shown] + vectors[shown]], axis=1), colors[shown]
