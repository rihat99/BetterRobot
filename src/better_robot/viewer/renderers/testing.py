"""``renderers/testing.py`` — ``MockBackend`` for unit tests.

A pure-Python implementation of ``RendererBackend`` that records every call
into a log. Render-mode unit tests attach a mode to a ``MockBackend``, call
``update(data)``, and assert on the recorded calls. No viser, no pyrender,
no ffmpeg needed.

See ``docs/design/12_VIEWER.md §13``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class Call:
    """One recorded backend call."""
    method: str
    args: tuple
    kwargs: dict


class MockBackend:
    """Records all ``RendererBackend`` calls for assertion in tests."""

    is_interactive: bool = False
    supports_gui: bool = False

    def __init__(self) -> None:
        self.calls: list[Call] = []
        # Track current state for convenience assertions
        self.transforms: dict[str, torch.Tensor] = {}
        self.visible: dict[str, bool] = {}
        self.nodes: set[str] = set()
        self._camera: Any = None

    def _record(self, method: str, *args: Any, **kwargs: Any) -> None:
        self.calls.append(Call(method=method, args=args, kwargs=kwargs))

    def calls_for(self, method: str) -> list[Call]:
        """Return all calls with the given method name."""
        return [c for c in self.calls if c.method == method]

    def last_transform(self, name: str) -> torch.Tensor | None:
        """Return the last pose set for *name*, or None."""
        return self.transforms.get(name)

    def reset(self) -> None:
        """Clear the call log and state."""
        self.calls.clear()
        self.transforms.clear()
        self.visible.clear()
        self.nodes.clear()

    # ------------------------------------------------------------------
    # Geometry primitives
    # ------------------------------------------------------------------

    def add_mesh(self, name: str, vertices: torch.Tensor, faces: torch.Tensor, *,
                 rgba=(0.8, 0.8, 0.8, 1.0), parent=None) -> None:
        self._record("add_mesh", name, vertices, faces, rgba=rgba, parent=parent)
        self.nodes.add(name)
        self.visible[name] = True

    def add_sphere(self, name: str, *, radius: float, rgba, parent=None) -> None:
        self._record("add_sphere", name, radius=radius, rgba=rgba, parent=parent)
        self.nodes.add(name)
        self.visible[name] = True

    def add_cylinder(self, name: str, *, radius: float, length: float,
                     rgba, parent=None) -> None:
        self._record("add_cylinder", name, radius=radius, length=length,
                     rgba=rgba, parent=parent)
        self.nodes.add(name)
        self.visible[name] = True

    def add_capsule(self, name: str, *, radius: float, length: float,
                    rgba, parent=None) -> None:
        self._record("add_capsule", name, radius=radius, length=length,
                     rgba=rgba, parent=parent)
        self.nodes.add(name)
        self.visible[name] = True

    def add_frame(self, name: str, *, axes_length: float = 0.1) -> None:
        self._record("add_frame", name, axes_length=axes_length)
        self.nodes.add(name)
        self.visible[name] = True

    def add_grid(self, name: str, **kwargs: Any) -> None:
        self._record("add_grid", name, **kwargs)
        self.nodes.add(name)
        self.visible[name] = True

    def add_mesh_trimesh(self, name: str, mesh: Any, *, scale: Any = 1.0) -> None:
        self._record("add_mesh_trimesh", name, mesh=mesh, scale=scale)
        self.nodes.add(name)
        self.visible[name] = True

    def remove(self, name: str) -> None:
        self._record("remove", name)
        self.nodes.discard(name)
        self.transforms.pop(name, None)
        self.visible.pop(name, None)

    def set_transform(self, name: str, pose: torch.Tensor) -> None:
        self._record("set_transform", name, pose)
        self.transforms[name] = pose.detach().clone()

    def set_visible(self, name: str, visible: bool) -> None:
        self._record("set_visible", name, visible)
        self.visible[name] = visible

    def set_camera(self, camera: Any) -> None:
        self._record("set_camera", camera)
        self._camera = camera

    def capture_frame(self) -> "Any":
        """Return a solid-colour (H, W, 3) uint8 array for testing."""
        import numpy as np
        self._record("capture_frame")
        return (128 * torch.ones(64, 64, 3, dtype=torch.uint8)).numpy().astype("uint8")

    # ------------------------------------------------------------------
    def add_transform_control(
        self,
        name: str,
        pose: Any,
        *,
        scale: float = 0.15,
        on_update: Any = None,
    ) -> None:
        self._record("add_transform_control", name, pose=pose, scale=scale)
        self.nodes.add(name)
        return None

    # ------------------------------------------------------------------
    # GUI (no-op — MockBackend ignores GUI calls gracefully)
    # ------------------------------------------------------------------

    def add_gui_button(self, label: str, callback: Any) -> None:
        self._record("add_gui_button", label)

    def add_gui_slider(self, label: str, *, min: float, max: float,
                       step: float, value: float, callback: Any) -> None:
        self._record("add_gui_slider", label, min=min, max=max,
                     step=step, value=value)

    def add_gui_checkbox(self, label: str, *, value: bool, callback: Any) -> None:
        self._record("add_gui_checkbox", label, value=value)
