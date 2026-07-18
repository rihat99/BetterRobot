"""``visualizer.py`` — ``Visualizer``, the top-level interactive facade.

The facade is intentionally thin: one ``ViserBackend``, one ``Scene.default``
per robot, single-pose ``update`` and straight-through trajectory
``add_trajectory``.

``viser`` is imported lazily — only when ``_ensure_server()`` is first
called via ``show()`` / ``update()`` / ``add_trajectory()``. ``import
better_robot.viewer`` works without viser installed.

See ``docs/concepts/viewer.md``.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Callable

import torch

if TYPE_CHECKING:
    from ..data_model.data import Data
    from ..data_model.model import Model
    from ..tasks.ik import IKResult
    from ..tasks.trajectory import Trajectory
    from .overlays.targets import TargetsOverlay
    from .primitive import PrimitiveHandle
    from .scene import Scene
    from .themes import Theme
    from .trajectory_player import TrajectoryPlayer


class Visualizer:
    """Top-level viser-backed visualiser.

    A ``Visualizer`` owns one ``ViserBackend`` and one ``Scene.default``.
    It exposes a minimal surface — ``show``, ``update``,
    ``add_trajectory``, ``add_ik_result`` — and forwards each call to
    the scene.

    Example
    -------
    >>> viewer = Visualizer(model, port=8080)
    >>> viewer.show()                    # opens browser, blocks for lifetime
    >>> viewer.update(q)                 # single-pose update
    >>> viewer.add_trajectory(traj)      # straight-through playback
    """

    def __init__(
        self,
        model: "Model",
        *,
        port: int = 8080,
        theme: "Theme | None" = None,
    ) -> None:
        self._model = model
        self._port = port
        self._theme = theme
        self._backend: Any = None
        self._scene: "Scene | None" = None
        self._player: "TrajectoryPlayer | None" = None
        self._last_q: torch.Tensor = model.q_neutral.clone()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _ensure_server(self) -> None:
        """Lazily construct the ViserBackend and default Scene."""
        if self._backend is None:
            from .renderers.viser_backend import ViserBackend
            from .scene import Scene

            self._backend = ViserBackend(port=self._port)
            self._scene = Scene.default(
                self._model,
                backend=self._backend,
                theme=self._theme,
            )
            self._scene.update_from_q(self._last_q)

    def show(self, *, block: bool = True) -> None:
        """Start the viser server and optionally block until Ctrl-C."""
        self._ensure_server()
        if block:
            try:
                while True:
                    time.sleep(0.05)
            except KeyboardInterrupt:
                pass

    def close(self) -> None:
        """Drop the viser server reference.

        ``viser.ViserServer`` does not expose an explicit close hook in
        every version, so this is a best-effort teardown.
        """
        self._backend = None
        self._scene = None
        self._player = None

    # ------------------------------------------------------------------
    # Direct updates
    # ------------------------------------------------------------------

    @property
    def last_q(self) -> torch.Tensor:
        """The last configuration pushed via :meth:`update`."""
        return self._last_q

    def update(self, q_or_data: "torch.Tensor | Data") -> None:
        """Push a new configuration or ``Data`` to the scene."""
        from ..data_model.data import Data

        self._ensure_server()
        assert self._scene is not None
        if isinstance(q_or_data, Data):
            self._scene.update(q_or_data)
        else:
            self._last_q = q_or_data.clone()
            self._scene.update_from_q(q_or_data)

    # ------------------------------------------------------------------
    # Trajectory playback
    # ------------------------------------------------------------------

    def add_trajectory(
        self,
        trajectory: "Trajectory",
    ) -> "TrajectoryPlayer":
        """Attach a ``Trajectory`` and return a ``TrajectoryPlayer``.

        Callers drive playback via ``player.show_frame(k)`` or
        ``player.play(fps=...)``. No richer transport controls are exposed;
        see ``docs/concepts/viewer.md`` ("Trajectory playback").
        """
        from .trajectory_player import TrajectoryPlayer

        self._ensure_server()
        assert self._scene is not None
        self._player = TrajectoryPlayer(self._scene, trajectory)
        return self._player

    def current_player(self) -> "TrajectoryPlayer | None":
        """Return the attached trajectory player, if any."""
        return self._player

    def show_frame(self, k: int) -> None:
        """Push trajectory frame ``k`` through the public playback surface."""
        if self._player is None:
            raise RuntimeError("add_trajectory() must be called before show_frame()")
        self._player.show_frame(k)

    def joint_primitive(self, joint_id: int) -> "PrimitiveHandle":
        """Return a public handle for styling a joint sphere."""
        self._ensure_server()
        assert self._scene is not None
        return self._scene.joint_primitive(joint_id)

    # ------------------------------------------------------------------
    # IK
    # ------------------------------------------------------------------

    def add_ik_result(self, result: "IKResult") -> None:
        """Draw the IK solution configuration (single pose)."""
        self.update(result.q)

    def add_ik_targets(
        self,
        targets: dict[str, torch.Tensor],
        *,
        on_change: "Callable[[dict[str, torch.Tensor]], None] | None" = None,
        scale: float = 0.15,
    ) -> "TargetsOverlay":
        """Attach a draggable SE(3) gizmo per IK target frame.

        Drops a ``TargetsOverlay`` onto the active scene using the same
        ``Scene.add_mode`` lifecycle as any other render mode. Each
        target renders as a draggable transform control on interactive
        backends (viser). Non-interactive backends instead render a static
        frame triad. On every drag, ``on_change`` fires with the updated
        ``{frame_name: pose_7vec}`` dict.

        Typical interactive-IK loop::

            targets = {frame: initial_pose}

            def on_move(new_targets):
                r = br.solve_ik(model, new_targets,
                                initial_q=viewer.last_q)
                viewer.update(r.q)

            viewer.add_ik_targets(targets, on_change=on_move)
            viewer.show()

        The returned ``TargetsOverlay`` exposes its current targets via
        the ``.targets`` property.
        """
        from .overlays.targets import TargetsOverlay

        self._ensure_server()
        assert self._scene is not None
        overlay = TargetsOverlay(targets, on_change=on_change, scale=scale)
        self._scene.add_mode(overlay)
        return overlay

    def scene(self, name: str | None = None) -> "Scene":
        """Return the active Scene (V1 has exactly one)."""
        if name is not None:
            raise ValueError("Visualizer owns exactly one unnamed scene")
        self._ensure_server()
        assert self._scene is not None
        return self._scene
