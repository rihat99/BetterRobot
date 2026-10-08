"""``ImagePanel``: an image in the sidebar that follows the current step, e.g. the input video."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from ..layer import Layer


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
                raise ValueError("a callable `images` needs `nsteps`")
        else:
            images = np.asarray(images)
            images = images[None] if images.ndim == 3 else images
            if images.ndim != 4:
                raise ValueError(f"images must be (H, W, 3) or (T, H, W, 3), got shape {images.shape}")
            nsteps = images.shape[0]
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
        self._image = gui.add_image(self.image_at(0), label=self.name, format=self.format)

    def set_step(self, step: int) -> None:
        self._image.image = self.image_at(step)

    def set_visible(self, visible: bool) -> None:  # noqa: FBT001 - signature of Layer.set_visible
        self._image.visible = visible

    def dispose(self) -> None:
        self._image.remove()
