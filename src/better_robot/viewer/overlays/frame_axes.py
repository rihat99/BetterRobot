"""``overlays/frame_axes.py`` — coordinate triad overlays on named frames.

See ``docs/design/12_VIEWER.md §5``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch

from ..render_modes.base import RenderContext

if TYPE_CHECKING:
    from ...data_model.data import Data
    from ...data_model.model import Model


class FrameAxesOverlay:
    """Draw a coordinate triad on each selected (or all) model frames.

    ``frame_names=None`` means show all frames in the model.
    """

    name = "Frame axes"
    description = "Coordinate triads on named frames"

    def __init__(
        self,
        *,
        frame_names: Sequence[str] | None = None,
        axes_length: float = 0.04,
        visible: bool = False,
    ) -> None:
        self._requested_names = list(frame_names) if frame_names is not None else None
        self._axes_length = axes_length
        self._initial_visible = visible
        self._ctx: RenderContext | None = None
        self._model: Model | None = None
        self._frame_ids: list[int] = []

    @classmethod
    def is_available(cls, model: "Model", data: "Data") -> bool:
        return True

    def attach(self, context: RenderContext, model: "Model", data: "Data") -> None:
        self._ctx = context
        self._model = model
        backend = context.backend
        ns = context.namespace

        # Resolve which frames to draw
        if self._requested_names is None:
            self._frame_ids = list(range(model.nframes))
        else:
            self._frame_ids = [model.frame_id(n) for n in self._requested_names]

        for fid in self._frame_ids:
            name = f"{ns}/frame_{fid}"
            backend.add_frame(name, axes_length=self._axes_length)

        # Set initial poses if frame_pose_world is available
        if data.frame_pose_world is not None:
            self._set_poses(data)

        # Start hidden by default
        if not self._initial_visible:
            self.set_visible(False)

    def update(self, data: "Data") -> None:
        if self._ctx is None or data.frame_pose_world is None:
            return
        self._set_poses(data)

    def set_visible(self, visible: bool) -> None:
        if self._ctx is None:
            return
        backend = self._ctx.backend
        ns = self._ctx.namespace
        for fid in self._frame_ids:
            backend.set_visible(f"{ns}/frame_{fid}", visible)

    def detach(self) -> None:
        if self._ctx is None:
            return
        backend = self._ctx.backend
        ns = self._ctx.namespace
        for fid in self._frame_ids:
            backend.remove(f"{ns}/frame_{fid}")
        self._frame_ids = []
        self._ctx = None

    def _set_poses(self, data: "Data") -> None:
        backend = self._ctx.backend
        ns = self._ctx.namespace
        b = self._ctx.batch_index
        frame_pose_world = data.frame_pose_world
        for fid in self._frame_ids:
            name = f"{ns}/frame_{fid}"
            if frame_pose_world.dim() == 2:
                pose = frame_pose_world[fid]
            elif frame_pose_world.dim() == 3:
                pose = frame_pose_world[b, fid]
            else:
                pose = frame_pose_world[b, fid]
            backend.set_transform(name, pose)
