"""``Layer``: one visual fed by preloaded ``(T, ...)`` arrays, drawn under one scene node."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class Geometry:
    """World-space triangles of one layer at one step, as drawn by the offline renderer.

    ``vertices (V, 3)`` in world coordinates, ``faces (F, 3)`` vertex indices, ``colors`` either one RGB
    ``(3,)`` in ``[0, 1]`` or one per face ``(F, 3)``.
    """

    vertices: np.ndarray
    faces: np.ndarray
    colors: np.ndarray


class Layer(ABC):
    """Base class for everything the viewer draws.

    A layer owns its preloaded arrays and the viser handles it creates in ``build``. The viewer
    calls ``set_step(k)`` to write slice ``k`` into those handles; a static layer has
    ``nsteps == 1`` and is pushed once. Handles registered with ``_track`` get the default
    ``set_visible`` and ``dispose``.

    Subclasses implement ``build`` and ``set_step``; ``build_gui`` is optional. ``fits_camera = False``
    keeps a layer's geometry out of the offline renderer's automatic framing (the ground grid);
    ``listed = False`` keeps it out of the sidebar's Layers folder.
    """

    fits_camera: bool = True
    listed: bool = True

    def __init__(self, name: str, nsteps: int = 1) -> None:
        if nsteps < 1:
            raise ValueError("nsteps must be at least 1")
        self.name = name
        self.nsteps = nsteps
        self.root = ""
        self._handles: list[Any] = []
        self._visible = True

    @property
    def visible(self) -> bool:
        """Whether the layer is shown; change it with ``Viewer.set_visible``."""
        return self._visible

    @abstractmethod
    def build(self, scene: Any, root: str) -> None:
        """Create the scene nodes under ``root`` (a viser scene path such as ``/scene/robot``)."""

    @abstractmethod
    def set_step(self, step: int) -> None:
        """Write step ``step`` (already clamped to ``nsteps - 1``) into the existing handles."""

    def build_gui(self, gui: Any) -> None:
        """Add per-layer controls to an open GUI folder; nothing by default."""

    def apply_theme(self, theme: Any) -> None:
        """Adopt defaults from the viewer's ``Theme`` before ``build``; nothing by default."""

    def geometry(self, step: int) -> list[Geometry]:
        """Triangles of step ``step`` for the offline renderer; empty for layers without 3D geometry."""
        return []

    def set_visible(self, visible: bool) -> None:
        """Show or hide every tracked handle."""
        self._visible = visible
        for handle in self._handles:
            handle.visible = visible

    def dispose(self) -> None:
        """Remove every tracked handle from the scene."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _track(self, handle: Any) -> Any:
        self._handles.append(handle)
        return handle
