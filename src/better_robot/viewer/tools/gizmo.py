"""``PoseGizmo``: a draggable SE(3) handle that reports poses in BetterRobot order."""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any, final

import numpy as np

from .._lie import to_wxyz


@final
class PoseGizmo:
    """Transform controls at ``pose`` ``[x, y, z, qx, qy, qz, qw]``; ``on_change(pose)`` fires on drags.

    Callbacks run on one worker thread owned by the gizmo, never on viser's callback pool: a drag
    overwrites a single pending pose and the worker handles only the newest one (latest wins), so a
    slow callback such as an IK solve never piles up behind the mouse. Releasing the handle (viser's end phase of
    ``on_update``) queues a final call with the resting pose. :meth:`wait` blocks until the worker is idle. A
    callback that raises stops the worker: its traceback goes to ``threading.excepthook`` (printed by default) and
    every later :meth:`wait` raises it. The worker is a daemon thread: it ends with :meth:`remove` and never keeps the interpreter alive.
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
            f"/tools/{name}", scale=scale, position=pose[:3], wxyz=to_wxyz(pose[3:]), **options
        )
        self._callbacks: list[Callable[[np.ndarray], None]] = [on_change] if on_change else []
        self._condition = threading.Condition()
        self._pending: np.ndarray | None = None
        self._busy = False
        self._closed = False
        self._error: BaseException | None = None  # what a callback raised; the worker stopped on it
        self._worker = threading.Thread(target=self._run, name=f"gizmo:{name}", daemon=True)
        self._worker.start()
        self.handle.on_update(lambda _: self._submit())  # also fires on the drag's end phase

    @property
    def pose(self) -> np.ndarray:
        """Current pose ``(7,)`` in ``[x, y, z, qx, qy, qz, qw]`` order."""
        wxyz = np.asarray(self.handle.wxyz, dtype=np.float32)
        return np.concatenate([np.asarray(self.handle.position, dtype=np.float32), wxyz[[1, 2, 3, 0]]])

    @pose.setter
    def pose(self, pose: np.ndarray) -> None:
        pose = np.asarray(pose, dtype=np.float32)
        self.handle.position = pose[:3]
        self.handle.wxyz = to_wxyz(pose[3:])

    def on_change(self, callback: Callable[[np.ndarray], None]) -> None:
        """Also call ``callback(pose (7,))`` on drags (on the gizmo's worker thread)."""
        self._callbacks.append(callback)

    def wait(self, timeout: float | None = None) -> bool:
        """Block until no pose is pending or being handled, or the gizmo is removed; ``False`` if ``timeout`` ran out.

        Raises what a callback raised, on every call once the worker has stopped on it.
        """
        with self._condition:
            idle = self._condition.wait_for(lambda: not self._busy and (self._pending is None or self._closed), timeout)
        if self._error is not None:
            raise self._error
        return idle

    def remove(self) -> None:
        """Remove the handle from the scene, drop a pending pose and stop the worker.

        A callback that is running finishes before ``remove`` returns, unless ``remove`` is called from a callback.
        """
        with self._condition:
            self._closed = True
            self._pending = None
            self._condition.notify_all()
        if threading.current_thread() is not self._worker:
            self._worker.join()
        self.handle.remove()

    def _submit(self) -> None:
        with self._condition:
            if not self._closed:
                self._pending = self.pose
                self._condition.notify_all()

    def _run(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._pending is not None or self._closed)
                if self._closed:
                    return
                pose, self._pending, self._busy = self._pending, None, True
            try:
                for callback in self._callbacks:
                    callback(pose)
            except BaseException as error:
                with self._condition:
                    self._error, self._closed = error, True
                raise
            finally:
                with self._condition:
                    self._busy = False
                    self._condition.notify_all()
