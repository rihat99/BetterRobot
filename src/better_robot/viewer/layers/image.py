"""``ImagePanel``: an image in the sidebar that follows the current step, e.g. the input video."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, final

import numpy as np

from ...exceptions import ArgumentError, ShapeError
from ..layer import Layer


@final
class ImagePanel(Layer):
    """Sidebar image per step: ``images`` ``(T, H, W, 3)`` uint8, one ``(H, W, 3)``, or a callable ``step -> image``.

    A callable is evaluated only when its step is shown (decoding video frames on demand) and needs ``nsteps``.
    Lives in the GUI like `PlotPanel`; ``Viewer.add(..., times=...)`` syncs a video recorded at another rate.
    ``format`` is the transport encoding, ``"jpeg"`` or ``"png"``.
    """

    def __init__(
        self,
        name: str,
        images: np.ndarray | Callable[[int], np.ndarray],
        *,
        nsteps: int | None = None,
        format: str = "jpeg",
    ) -> None:
        if callable(images):
            if nsteps is None:
                raise ArgumentError("a callable `images` needs `nsteps`: pass ImagePanel(name, images, nsteps=T)")
        else:
            images = np.asarray(images)
            images = images[None] if images.ndim == 3 else images
            if images.ndim != 4:
                raise ShapeError(f"images must be (H, W, 3) or (T, H, W, 3), got shape {images.shape}")
            nsteps = len(images)
        super().__init__(name, nsteps)
        self._images = images
        self.format = format
        self._image: Any = None

    def image_at(self, step: int) -> np.ndarray:
        """Image ``(H, W, 3)`` shown at ``step``."""
        return self._images(step) if callable(self._images) else self._images[step]

    def build(self, scene: Any, root: str) -> None:
        pass

    def build_gui(self, gui: Any) -> None:
        self._image = self._track(gui.add_image(self.image_at(0), label=self.name, format=self.format))

    def set_step(self, step: int) -> None:
        self._image.image = self.image_at(step)
