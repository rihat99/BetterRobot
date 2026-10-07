"""``PlotPanel``: time series in the sidebar with a marker at the current frame."""

from __future__ import annotations

from typing import Any

import numpy as np

from .._math import as_frames
from ..layer import Layer

_PALETTE = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
)


class PlotPanel(Layer):
    """Series ``(T, K)`` against time ``t = frame / fps`` (uPlot); a dot per series marks the frame.

    Lives in the GUI, not the 3D scene: ``build`` does nothing and ``build_gui`` creates the chart.
    """

    def __init__(self, name: str, series: np.ndarray, *, labels: list[str] | None = None, fps: float = 30.0) -> None:
        self.series = as_frames(series, 1)
        super().__init__(name, self.series.shape[0])
        if labels is not None and len(labels) != self.series.shape[1]:
            raise ValueError("one label per series")
        self.labels = labels or [f"{name} {k}" for k in range(self.series.shape[1])]
        self.time = np.arange(self.series.shape[0], dtype=np.float64) / fps
        self._plot: Any = None
        self._visible = True

    def build(self, scene: Any, root: str) -> None:
        pass

    def build_gui(self, gui: Any) -> None:
        data = (self.time, *(self.series[:, k].astype(np.float64) for k in range(self.series.shape[1])))
        self._plot = gui.add_uplot(data, self._series_config(0), aspect=1.6, legend={"show": False})

    def set_frame(self, frame: int) -> None:
        self._plot.series = self._series_config(frame)

    def set_visible(self, visible: bool) -> None:
        self._plot.visible = visible

    def dispose(self) -> None:
        self._plot.remove()

    def _series_config(self, frame: int) -> tuple[dict[str, Any], ...]:
        marker = {"show": True, "size": 8, "filter": (frame,)}
        return (
            {"label": "t"},
            *(
                {"label": label, "stroke": _PALETTE[k % len(_PALETTE)], "width": 1.5, "points": marker}
                for k, label in enumerate(self.labels)
            ),
        )
