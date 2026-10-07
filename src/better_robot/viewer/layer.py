"""``Layer``: one visual fed by preloaded ``(T, ...)`` arrays, drawn under one scene node."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class Geometry:
    """World-space triangles of one layer at one frame, as drawn by the offline renderer.

    ``vertices (V, 3)`` in world coordinates, ``faces (F, 3)`` vertex indices, ``colors`` either one RGB
    ``(3,)`` in ``[0, 1]`` or one per face ``(F, 3)``.
    """

    vertices: np.ndarray
    faces: np.ndarray
    colors: np.ndarray


class Layer(ABC):
    """Base class for everything the viewer draws.

    A layer owns its preloaded arrays and the viser handles it creates in ``build``. The viewer
    calls ``set_frame(k)`` to write slice ``k`` into those handles; a static layer has
    ``num_frames == 1`` and is pushed once. Handles registered with ``_track`` get the default
    ``set_visible`` and ``dispose``.

    Subclasses implement ``build`` and ``set_frame``; ``build_gui`` is optional.
    """

    def __init__(self, name: str, num_frames: int = 1) -> None:
        if num_frames < 1:
            raise ValueError("num_frames must be at least 1")
        self.name = name
        self.num_frames = num_frames
        self.root = ""
        self._handles: list[Any] = []

    @abstractmethod
    def build(self, scene: Any, root: str) -> None:
        """Create the scene nodes under ``root`` (a viser scene path such as ``/scene/robot``)."""

    @abstractmethod
    def set_frame(self, frame: int) -> None:
        """Write frame ``frame`` (already clamped to ``num_frames - 1``) into the existing handles."""

    def build_gui(self, gui: Any) -> None:
        """Add per-layer controls to an open GUI folder; nothing by default."""

    def geometry(self, frame: int) -> list[Geometry]:
        """Triangles of frame ``frame`` for the offline renderer; empty for layers without 3D geometry."""
        return []

    def set_visible(self, visible: bool) -> None:
        """Show or hide every tracked handle."""
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
