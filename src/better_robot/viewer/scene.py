"""``scene.py`` — ``Scene``, render-mode composition for one robot.

A ``Scene`` owns the set of attached render modes, keeps the per-mode
visible flags, and routes every ``update()`` to exactly the visible
modes. It is the extensibility point for future render modes.

V1 intentionally drops the robot_collision argument and the
batch-axis picker — see ``docs/concepts/viewer.md §6`` and §10.9.

See ``docs/concepts/viewer.md §6``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from .themes import DEFAULT_THEME, Theme
from .render_modes.base import RenderContext

if TYPE_CHECKING:
    from ..data_model.data import Data
    from ..data_model.model import Model


class Scene:
    """Holds the active render modes for one ``Model`` on one backend.

    A ``Scene`` is where "toggle modes in the UI" behaviour lives. It
    owns the mode instances, keeps the UI panel in sync, and routes
    every ``update()`` to exactly the visible modes.
    """

    def __init__(
        self,
        model: "Model",
        *,
        backend: Any,
        namespace: str = "/robot",
        theme: Theme | None = None,
    ) -> None:
        self._model = model
        self._backend = backend
        self._namespace = namespace
        self._theme: Theme = theme or DEFAULT_THEME
        self._modes: dict[str, Any] = {}        # name → mode instance
        self._visible: dict[str, bool] = {}     # name → visible flag
        self._available: set[str] = set()       # names for which is_available → True
        self._last_data: "Data | None" = None

    # ------------------------------------------------------------------
    # Class method constructor
    # ------------------------------------------------------------------

    @classmethod
    def default(
        cls,
        model: "Model",
        *,
        backend: Any,
        theme: Theme | None = None,
    ) -> "Scene":
        """Attach the best available primary mode plus GridOverlay and FrameAxesOverlay."""
        from ..kinematics.forward import forward_kinematics
        from .render_modes.urdf_mesh import URDFMeshMode
        from .render_modes.skeleton import SkeletonMode
        from .overlays.grid import GridOverlay
        from .overlays.frame_axes import FrameAxesOverlay

        scene = cls(model, backend=backend, theme=theme)

        # Compute a zero-config Data to check availability
        q0 = model.q_neutral
        data0 = forward_kinematics(model, q0, compute_frames=True)

        if URDFMeshMode.is_available(model, data0):
            scene.add_mode(URDFMeshMode())
        else:
            scene.add_mode(SkeletonMode())

        scene.add_mode(GridOverlay())
        scene.add_mode(FrameAxesOverlay())
        return scene

    # ------------------------------------------------------------------
    # Mode management
    # ------------------------------------------------------------------

    def add_mode(self, mode: Any) -> None:
        """Add a render mode or overlay. Calls ``attach`` if available."""
        from ..kinematics.forward import forward_kinematics

        name = mode.name

        # Check availability — most modes use (model, data); the
        # CollisionMode stub additionally accepts a robot_collision kwarg
        # which is not used in V1.
        try:
            available = mode.is_available(
                self._model, self._last_data or _empty_data(self._model)
            )
        except TypeError:
            available = mode.is_available(
                self._model,
                self._last_data or _empty_data(self._model),
                robot_collision=None,
            )

        self._modes[name] = mode
        self._visible[name] = True

        if available:
            self._available.add(name)
            data = self._last_data
            if data is None:
                q0 = self._model.q_neutral
                data = forward_kinematics(self._model, q0, compute_frames=True)
                self._last_data = data
            ctx = RenderContext(
                backend=self._backend,
                namespace=f"{self._namespace}/{name.replace(' ', '_')}",
                theme=self._theme,
            )
            mode.attach(ctx, self._model, data)

    def remove_mode(self, mode_name: str) -> None:
        mode = self._modes.pop(mode_name, None)
        if mode is not None and mode_name in self._available:
            mode.detach()
        self._available.discard(mode_name)
        self._visible.pop(mode_name, None)

    def available_modes(self) -> list[str]:
        """Names of modes whose ``is_available`` returned True."""
        return list(self._available)

    def set_mode_visible(self, mode_name: str, visible: bool) -> None:
        mode = self._modes.get(mode_name)
        if mode is not None and mode_name in self._available:
            mode.set_visible(visible)
            self._visible[mode_name] = visible

    # ------------------------------------------------------------------
    # State updates
    # ------------------------------------------------------------------

    def update(self, data: "Data") -> None:
        """Push new kinematics data to all attached visible modes."""
        self._last_data = data
        for name, mode in self._modes.items():
            if name in self._available and self._visible.get(name, True):
                mode.update(data)

    def update_from_q(self, q: torch.Tensor) -> None:
        """Run FK (+ frame placements) then push to all modes."""
        from ..kinematics.forward import forward_kinematics
        data = forward_kinematics(self._model, q, compute_frames=True)
        self.update(data)


def _empty_data(model: "Model") -> "Data":
    """Allocate a minimal Data with q=q_neutral (no FK)."""
    from ..data_model.data import Data
    return Data(_model_id=id(model), q=model.q_neutral)
