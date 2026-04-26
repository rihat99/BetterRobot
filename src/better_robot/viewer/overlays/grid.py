"""``overlays/grid.py`` — ground plane grid overlay.

Uses the backend's built-in grid (viser ``add_grid``) when available,
with ``infinite_grid=True`` for a clean look.

See ``docs/concepts/viewer.md §5``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..render_modes.base import RenderContext

if TYPE_CHECKING:
    from ...data_model.data import Data
    from ...data_model.model import Model


class GridOverlay:
    """Infinite ground grid at z=0.

    Always available.
    """

    name = "Grid"
    description = "Ground plane grid"

    def __init__(self) -> None:
        self._ctx: RenderContext | None = None
        self._node_name: str | None = None

    @classmethod
    def is_available(cls, model: "Model", data: "Data") -> bool:
        return True

    def attach(self, context: RenderContext, model: "Model", data: "Data") -> None:
        self._ctx = context
        backend = context.backend
        name = f"{context.namespace}/grid"

        if hasattr(backend, "add_grid"):
            backend.add_grid(name, plane="xy", infinite_grid=True)
        else:
            # Fallback for backends without add_grid
            backend.add_frame(name, axes_length=0.05)

        self._node_name = name

    def update(self, data: "Data") -> None:
        pass  # grid is static

    def set_visible(self, visible: bool) -> None:
        if self._ctx is None or self._node_name is None:
            return
        self._ctx.backend.set_visible(self._node_name, visible)

    def detach(self) -> None:
        if self._ctx is None or self._node_name is None:
            return
        self._ctx.backend.remove(self._node_name)
        self._node_name = None
        self._ctx = None
