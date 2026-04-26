"""``trajectory_player.py`` — minimal frame-by-frame sequence player.

V1 is deliberately tiny: a ``TrajectoryPlayer`` knows how to push one
frame of a ``(B, T, nq)`` ``Trajectory`` to a ``Scene``, and how to run
a straight loop across all frames at a fixed fps. There is no scrub
bar, speed, loop toggle, ghost/trace overlay, manifold interpolation
between keyframes, or batch-axis picker — each of those is listed as
future work in ``docs/design/12_VIEWER.md §10.3``.

See ``docs/design/12_VIEWER.md §8``.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .scene import Scene
    from ..tasks.trajectory import Trajectory


_FUTURE_MSG = (
    "Transport controls (seek/step/speed/loop/ghost/trace) are future "
    "work — see docs/design/12_VIEWER.md §10.3."
)


class TrajectoryPlayer:
    """Drives a ``Scene`` through a ``Trajectory`` ``(B, T, nq)`` by
    integer frame index.

    V1 API:
        - ``show_frame(k)``  — render frame *k* of batch element 0
        - ``play(fps=30.0)`` — blocking straight-through playback

    That's it. Any richer transport control (play/pause, seek, speed,
    loop, ghost, trace, batch picker, manifold interpolation) is §10.3.
    """

    def __init__(self, scene: "Scene", trajectory: "Trajectory") -> None:
        self._scene = scene
        self._traj = trajectory
        self._model = scene._model
        # Push frame 0 immediately so the scene reflects the trajectory
        # before play() is called.
        self.show_frame(0)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def horizon(self) -> int:
        """Number of frames ``T`` in the trajectory."""
        return int(self._traj.horizon)

    # ------------------------------------------------------------------
    # V1 playback — minimal
    # ------------------------------------------------------------------

    def show_frame(self, k: int) -> None:
        """Push the ``k``-th keyframe (of batch 0) to the scene.

        Out-of-range ``k`` is clamped to ``[0, T-1]``.
        """
        T = self.horizon
        if T == 0:
            return
        k = max(0, min(int(k), T - 1))
        q_k = self._traj.q[0, k]
        self._scene.update_from_q(q_k)

    def play(self, *, fps: float = 30.0) -> None:
        """Blocking straight-through playback from frame 0 to ``T-1``.

        Sleeps ``1 / fps`` seconds between frames. Ctrl-C exits cleanly.
        """
        T = self.horizon
        if T == 0:
            return
        dt = 1.0 / float(fps) if fps > 0 else 0.0
        try:
            for k in range(T):
                self.show_frame(k)
                if dt > 0:
                    time.sleep(dt)
        except KeyboardInterrupt:
            return

    # ------------------------------------------------------------------
    # Future work (§10.3) — kept as stubs so callers get a clear error
    # ------------------------------------------------------------------

    def seek(self, t: float) -> None:
        raise NotImplementedError(_FUTURE_MSG)

    def seek_frame(self, k: int) -> None:
        # Deliberately routes through the V1 path so it keeps working.
        self.show_frame(k)

    def step(self, dt: float) -> None:
        raise NotImplementedError(_FUTURE_MSG)

    def pause(self) -> None:
        raise NotImplementedError(_FUTURE_MSG)

    def set_speed(self, speed: float) -> None:
        raise NotImplementedError(_FUTURE_MSG)

    def set_loop(self, loop: bool) -> None:
        raise NotImplementedError(_FUTURE_MSG)

    def set_ghost(self, every: int | None) -> None:
        raise NotImplementedError(_FUTURE_MSG)

    def set_trace(self, frame_name: str | None) -> None:
        raise NotImplementedError(_FUTURE_MSG)

    def set_batch_index(self, idx: int) -> None:
        raise NotImplementedError(_FUTURE_MSG)
