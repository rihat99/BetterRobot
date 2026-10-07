"""``PoseGizmo``: a draggable SE(3) handle that reports poses in BetterRobot order."""

from __future__ import annotations

import threading
import traceback
from collections.abc import Callable
from typing import Any

import numpy as np

from .._math import xyzw_to_wxyz


class PoseGizmo:
    """Transform controls at ``pose`` ``[x, y, z, qx, qy, qz, qw]``; ``on_change(pose)`` fires on drags.

    Callbacks run on one worker thread owned by the gizmo, never on viser's callback pool: a drag
    overwrites a single pending pose and the worker handles only the newest one (latest wins), so a
    slow callback such as an IK solve never piles up behind the mouse. Releasing the handle queues
    a final call with the resting pose. :meth:`wait` blocks until the worker is idle.
    Tools are not layers: they hold the viewer and callbacks and sit outside the step loop.
    ``scale`` is the handle size in metres (viser's default of 1 dwarfs a robot). Extra
    ``options`` go to viser's ``add_transform_controls`` (``disable_sliders=True`` and
    ``disable_axes=True`` make a rotation-only handle).
    """

    def __init__(
        self,
        viewer: Any,
        name: str,
        pose: np.ndarray,
        *,
        on_change: Callable[[np.ndarray], None] | None = None,
        scale: float = 0.15,
        **options: Any,
    ) -> None:
        pose = np.asarray(pose, dtype=np.float32)
        self.handle = viewer.server.scene.add_transform_controls(
            f"/tools/{name}", scale=scale, position=pose[:3], wxyz=xyzw_to_wxyz(pose[3:]), **options
        )
        self._callbacks: list[Callable[[np.ndarray], None]] = [on_change] if on_change else []
        self._condition = threading.Condition()
        self._pending: np.ndarray | None = None
        self._busy = False
        self._closed = False
        threading.Thread(target=self._run, name=f"gizmo:{name}", daemon=True).start()
        self.handle.on_update(lambda _: self._submit())
        self.handle.on_drag_end(lambda _: self._submit())

    @property
    def pose(self) -> np.ndarray:
        """Current pose ``(7,)`` in ``[x, y, z, qx, qy, qz, qw]`` order."""
        wxyz = np.asarray(self.handle.wxyz, dtype=np.float32)
        return np.concatenate([np.asarray(self.handle.position, dtype=np.float32), wxyz[[1, 2, 3, 0]]])

    @pose.setter
    def pose(self, pose: np.ndarray) -> None:
        pose = np.asarray(pose, dtype=np.float32)
        self.handle.position = pose[:3]
        self.handle.wxyz = xyzw_to_wxyz(pose[3:])

    def on_change(self, callback: Callable[[np.ndarray], None]) -> None:
        """Also call ``callback(pose (7,))`` on drags (on the gizmo's worker thread)."""
        self._callbacks.append(callback)

    def wait(self, timeout: float | None = None) -> bool:
        """Block until no pose is pending or being handled; ``False`` if ``timeout`` ran out."""
        with self._condition:
            return self._condition.wait_for(lambda: self._pending is None and not self._busy, timeout)

    def remove(self) -> None:
        """Remove the handle from the scene and stop the worker."""
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        self.handle.remove()

    def _submit(self) -> None:
        with self._condition:
            self._pending = self.pose
            self._condition.notify_all()

    def _run(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._pending is not None or self._closed)
                if self._closed:
                    return
                pose, self._pending, self._busy = self._pending, None, True
            for callback in self._callbacks:
                try:
                    callback(pose)
                except Exception:  # a failing callback must not kill the worker; report and go on
                    traceback.print_exc()
            with self._condition:
                self._busy = False
                self._condition.notify_all()
