"""``LinesLayer``: line segments, static or per step (trajectories, skeleton wires, a frustum's edges)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, final, override

import numpy as np

from ...exceptions import ShapeError
from ..layer import Bounds, Layer, as_frames, box
from ..theme import LIGHT, RGB, Theme, rgb_uint8


@final
class LinesLayer(Layer):
    """Segments ``points (T, N, 2, 3)`` or static ``(N, 2, 3)`` (start, end) drawn ``width`` pixels wide.

    ``colors`` is one RGB, ``(N, 3)`` one per segment, or ``(T, N, 3)`` per step; ``None`` draws them in the theme's
    ``text`` colour, dark on the light theme and light on the dark one.
    """

    def __init__(
        self,
        name: str,
        points: np.ndarray,
        *,
        colors: np.ndarray | RGB | None = None,
        width: float = 2.0,
        tags: str | Sequence[str] = (),
    ) -> None:
        self.points = as_frames(points, 3, name="points")
        if self.points.shape[2:] != (2, 3):
            raise ShapeError(f"points must be (N, 2, 3) or (T, N, 2, 3) segments, got {self.points.shape}")
        super().__init__(name, self.points.shape[0], tags=tags)
        n_segments = self.points.shape[1]
        self._colors: np.ndarray | None = None  # uint8 (N, 3) or (T, N, 3); None follows the theme
        if colors is not None:
            given = np.asarray(colors, dtype=np.float32)
            self._colors = rgb_uint8(given, (n_segments,)) if given.ndim <= 2 else rgb_uint8(given)
            if self._colors.ndim == 3 and self._colors.shape != (*self.points.shape[:2], 3):
                raise ShapeError(
                    f"per-step colors must be (T, N, 3) = {self.points.shape[:2]} + (3,), got {given.shape}"
                )
        self.width = width
        self._ink = rgb_uint8(LIGHT.text, (n_segments,))  # the theme's text colour, set by ``apply_theme``
        self._lines: Any = None

    @override
    def apply_theme(self, theme: Theme) -> None:
        self._ink = rgb_uint8(theme.text, (self.points.shape[1],))
        if self._colors is None and self._lines is not None:
            self._lines.colors = self._segment_colors(0)

    @override
    def build(self, scene: Any, root: str) -> None:
        self._lines = self._track(
            scene.add_line_segments(f"{root}/lines", self.points[0], self._segment_colors(0), line_width=self.width)
        )

    @override
    def bounds(self, step: int) -> Bounds | None:
        return box(self.points[step])

    @override
    def set_step(self, step: int) -> None:
        self._lines.points = self.points[step]
        if self._colors is not None and self._colors.ndim == 3:
            self._lines.colors = self._segment_colors(step)

    def _segment_colors(self, step: int) -> np.ndarray:
        """Colours ``(N, 2, 3)`` uint8 of both ends of each segment at ``step``, as viser takes them."""
        colors = self._ink if self._colors is None else self._colors if self._colors.ndim == 2 else self._colors[step]
        return np.repeat(colors[:, None], 2, axis=1)
