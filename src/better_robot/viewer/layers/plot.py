"""``PlotPanel``: time series in the sidebar with a marker at the current step."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING, final, override

import numpy as np

from ...exceptions import ShapeError
from ..layer import Layer, as_frames

if TYPE_CHECKING:
    from ..sidebar import Sidebar

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


@final
class PlotPanel(Layer):
    """Series ``(T, K)`` against time ``t = step / fps`` (uPlot); a dot per series marks the step.

    Lives in the GUI, not the 3D scene: ``build`` does nothing and ``build_gui`` creates the chart, which holds
    ``series`` as it was then; a push moves only the marker.
    """

    def __init__(self, name: str, series: np.ndarray, *, labels: list[str] | None = None, fps: float = 30.0) -> None:
        self.series = as_frames(series, 1)
        super().__init__(name, self.series.shape[0])
        if labels is not None and len(labels) != self.series.shape[1]:
            raise ShapeError(f"labels must have one entry per series ({self.series.shape[1]}), got {len(labels)}")
        self.labels = labels or [f"{name} {k}" for k in range(self.series.shape[1])]
        self.time = np.arange(self.series.shape[0], dtype=np.float64) / fps
        self._plot: Any = None

    @override
    def build(self, scene: Any, root: str) -> None:
        pass

    @override
    def build_gui(self, sidebar: Sidebar) -> None:
        data = (self.time, *(self.series[:, k].astype(np.float64) for k in range(self.series.shape[1])))
        with sidebar.section("Plots"):
            plot = sidebar.gui.add_uplot(
                # seconds on the x axis, not a date: uPlot reads x as a Unix time unless told otherwise
                data,
                self._series_config(0),
                scales={"x": {"time": False}},
                aspect=1.6,
                legend={"show": False},
            )
        self._plot = self._track(plot)

    @override
    def set_step(self, step: int) -> None:
        self._plot.series = self._series_config(step)

    def _series_config(self, step: int) -> tuple[dict[str, Any], ...]:
        marker = {"show": True, "size": 8, "filter": (step,)}
        return (
            {"label": "t"},
            *(
                {"label": label, "stroke": _PALETTE[k % len(_PALETTE)], "width": 1.5, "points": marker}
                for k, label in enumerate(self.labels)
            ),
        )
