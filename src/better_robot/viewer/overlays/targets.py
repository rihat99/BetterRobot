"""``overlays/targets.py`` — draggable IK-target gizmo overlay.

Lives as a ``RenderMode``/overlay so it composes with the rest of the
scene through the same attach/update/set_visible/detach lifecycle as
any other mode.

On interactive backends (``ViserBackend``) each target becomes a
draggable SE(3) transform control. On non-interactive backends
(``MockBackend``) each target renders as a static frame triad and no
callbacks fire. The ``on_change`` callback is invoked with the
updated ``dict[str, torch.Tensor]`` every time a gizmo is dragged —
the usual pattern is to re-solve IK inside the callback and push the
new configuration via ``Visualizer.update``.

See ``docs/concepts/viewer.md §7.1``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import torch

from ..helpers import quat_wxyz_to_xyzw
from ..render_modes.base import RenderContext

if TYPE_CHECKING:
    from ...data_model.data import Data
    from ...data_model.model import Model


class TargetsOverlay:
    """Draws one SE(3) gizmo per IK target frame.

    Parameters
    ----------
    targets:
        ``{frame_name: pose_7vec}`` mapping. Poses are
        ``[tx, ty, tz, qx, qy, qz, qw]``.
    on_change:
        Called with the updated targets dict whenever a gizmo moves.
        Only fires on interactive backends.
    scale:
        Axes length for the frame triad / gizmo size, in metres.
    """

    name = "IK targets"
    description = "Draggable SE(3) IK target gizmos"

    def __init__(
        self,
        targets: dict[str, torch.Tensor],
        *,
        on_change: Callable[[dict[str, torch.Tensor]], None] | None = None,
        scale: float = 0.15,
    ) -> None:
        self._targets: dict[str, torch.Tensor] = {
            k: v.clone() for k, v in targets.items()
        }
        self._on_change = on_change
        self._scale = scale
        self._ctx: RenderContext | None = None
        self._node_names: list[str] = []
        # frame_name → backend gizmo handle (viser handle or None on
        # non-interactive backends). Populated by attach().
        self._handles: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def targets(self) -> dict[str, torch.Tensor]:
        """Last-known target poses (a copy of the internal cache).

        Prefer :meth:`live_targets` when driving an interactive loop:
        it reads the current gizmo handles directly and does not
        depend on any ``on_update`` callback having fired.
        """
        return dict(self._targets)

    def live_targets(self) -> dict[str, torch.Tensor]:
        """Return the *current* gizmo poses freshly read from the backend.

        On an interactive backend (``ViserBackend``) each entry is
        re-read from the underlying transform-controls handle, whose
        ``.position`` / ``.wxyz`` viser keeps in sync with the browser
        as the user drags. On non-interactive backends (``MockBackend``
        and the future ``OffscreenBackend``) there are no handles, so
        this falls back to :meth:`targets`.

        This is the primary entry point for the interactive-IK loop —
        see ``examples/01_basic_ik.py`` / ``examples/02_g1_ik.py``.
        """
        if not self._handles:
            return self.targets

        import numpy as np
        out: dict[str, torch.Tensor] = {}
        for frame_name, handle in self._handles.items():
            if handle is None:
                out[frame_name] = self._targets[frame_name].clone()
                continue
            pos_np = np.array(handle.position, dtype=np.float32)
            wxyz_np = np.array(handle.wxyz, dtype=np.float32)
            xyzw = quat_wxyz_to_xyzw(torch.from_numpy(wxyz_np))
            pose7 = torch.cat([torch.from_numpy(pos_np), xyzw])
            out[frame_name] = pose7
            # Keep the cached copy in sync so a later ``.targets``
            # read mirrors what the user sees in the browser.
            self._targets[frame_name] = pose7.clone()
        return out

    # ------------------------------------------------------------------
    # RenderMode protocol
    # ------------------------------------------------------------------

    @classmethod
    def is_available(cls, model: "Model", data: "Data") -> bool:
        return True

    def attach(self, context: RenderContext, model: "Model", data: "Data") -> None:
        self._ctx = context
        backend = context.backend
        ns = context.namespace
        self._node_names = []
        self._handles = {}

        for frame_name, pose in self._targets.items():
            safe = frame_name.replace("/", "_").replace(" ", "_")
            node = f"{ns}/{safe}"
            pose7 = pose.float()

            if backend.is_interactive and hasattr(backend, "add_transform_control"):
                # Interactive: the transform control IS the visual —
                # place it directly at the namespace level so its
                # position/wxyz are in world coordinates (not parented
                # under another transformed node).
                handle = backend.add_transform_control(
                    node,
                    pose7,
                    scale=self._scale,
                    on_update=self._make_callback(frame_name),
                )
                self._handles[frame_name] = handle
            else:
                # Non-interactive: static frame triad as visual anchor.
                backend.add_frame(node, axes_length=self._scale)
                backend.set_transform(node, pose7)

            self._node_names.append(node)

    def update(self, data: "Data") -> None:
        # Targets are user-driven, not robot-state-driven — no-op.
        pass

    def set_visible(self, visible: bool) -> None:
        if self._ctx is None:
            return
        backend = self._ctx.backend
        for name in self._node_names:
            backend.set_visible(name, visible)

    def detach(self) -> None:
        if self._ctx is None:
            return
        backend = self._ctx.backend
        for name in self._node_names:
            backend.remove(name)
        self._node_names.clear()
        self._handles.clear()
        self._ctx = None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _make_callback(
        self, frame_name: str
    ) -> Callable[[torch.Tensor], None]:
        """Return a closure that updates the targets dict and fires on_change."""
        def _cb(new_pose: torch.Tensor) -> None:
            self._targets[frame_name] = new_pose.clone()
            if self._on_change is not None:
                self._on_change(dict(self._targets))
        return _cb
