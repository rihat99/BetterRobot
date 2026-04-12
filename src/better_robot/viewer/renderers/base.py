"""``renderers/base.py`` — ``RendererBackend`` protocol.

All renderer implementations must satisfy this protocol. Render modes and
overlays speak only to this interface — they never import viser or pyrender
directly.

See ``docs/12_VIEWER.md §9.2``.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import torch


@runtime_checkable
class RendererBackend(Protocol):
    """Protocol for renderer backends (viser, offscreen, etc.)."""

    is_interactive: bool
    supports_gui: bool

    def add_mesh(
        self,
        name: str,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        *,
        rgba: tuple[float, float, float, float] = (0.8, 0.8, 0.8, 1.0),
        parent: Any = None,
    ) -> None: ...

    def add_sphere(
        self,
        name: str,
        *,
        radius: float,
        rgba: tuple[float, float, float, float],
        parent: Any = None,
    ) -> None: ...

    def add_cylinder(
        self,
        name: str,
        *,
        radius: float,
        length: float,
        rgba: tuple[float, float, float, float],
        parent: Any = None,
    ) -> None: ...

    def add_capsule(
        self,
        name: str,
        *,
        radius: float,
        length: float,
        rgba: tuple[float, float, float, float],
        parent: Any = None,
    ) -> None: ...

    def add_frame(self, name: str, *, axes_length: float = 0.1) -> None: ...

    def remove(self, name: str) -> None: ...

    def set_transform(self, name: str, pose: torch.Tensor) -> None: ...

    def set_visible(self, name: str, visible: bool) -> None: ...

    def set_camera(self, camera: Any) -> None: ...

    def capture_frame(self) -> "np.ndarray": ...  # type: ignore[name-defined]  # noqa: F821

    def add_gui_button(self, label: str, callback: Any) -> None: ...

    def add_gui_slider(
        self,
        label: str,
        *,
        min: float,
        max: float,
        step: float,
        value: float,
        callback: Any,
    ) -> None: ...

    def add_gui_checkbox(self, label: str, *, value: bool, callback: Any) -> None: ...
