"""``overlays/path_trace.py`` — path trace overlay placeholder.

Rendering the swept world-frame path of a named frame across a
trajectory is future work — see ``docs/12_VIEWER.md §10.6``. The class
is kept here as a named target so future work can fill in the body
without restructuring the surrounding code.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..render_modes.base import RenderContext

if TYPE_CHECKING:
    from ...data_model.data import Data
    from ...data_model.model import Model


_FUTURE_MSG = (
    "PathTraceOverlay is future work — see docs/12_VIEWER.md §10.6."
)


class PathTraceOverlay:
    """Placeholder for the future frame-path overlay — see §10.6."""

    name = "path_trace"
    description = "World-frame path of a named frame (future work §10.6)"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(_FUTURE_MSG)

    @classmethod
    def is_available(cls, model: "Model", data: "Data") -> bool:
        return False

    def attach(self, context: RenderContext, model: "Model", data: "Data") -> None:
        raise NotImplementedError(_FUTURE_MSG)

    def update(self, data: "Data") -> None:
        raise NotImplementedError(_FUTURE_MSG)

    def set_visible(self, visible: bool) -> None:
        raise NotImplementedError(_FUTURE_MSG)

    def detach(self) -> None:
        raise NotImplementedError(_FUTURE_MSG)
