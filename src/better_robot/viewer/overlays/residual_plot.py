"""overlays/residual_plot.py — stub.

See docs/design/12_VIEWER.md §5.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..render_modes.base import RenderContext

if TYPE_CHECKING:
    from ...data_model.data import Data
    from ...data_model.model import Model


class ResidualPlotOverlay:
    """Stub — not yet implemented."""

    name = "residual_plot"
    description = "See docs/design/12_VIEWER.md §5"

    @classmethod
    def is_available(cls, model: "Model", data: "Data") -> bool:
        return True

    def attach(self, context: RenderContext, model: "Model", data: "Data") -> None:
        raise NotImplementedError("see docs/design/12_VIEWER.md §5")

    def update(self, data: "Data") -> None:
        raise NotImplementedError("see docs/design/12_VIEWER.md §5")

    def set_visible(self, visible: bool) -> None:
        raise NotImplementedError("see docs/design/12_VIEWER.md §5")

    def detach(self) -> None:
        raise NotImplementedError("see docs/design/12_VIEWER.md §5")
