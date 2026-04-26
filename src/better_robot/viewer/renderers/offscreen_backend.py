"""``renderers/offscreen_backend.py`` — headless backend placeholder.

Offscreen / headless rendering (pyrender + EGL/OSMesa) is deferred out
of V1. The ``OffscreenBackend`` class exists here as a named target so
that future work can drop the implementation in place without changing
the surrounding code — constructing it today raises
``NotImplementedError`` pointing at ``docs/design/12_VIEWER.md §10.2``.

The ``test_only_offscreen_backend_imports_pyrender`` layer rule in
``tests/test_layer_dependencies.py`` reserves ``pyrender`` imports for
this module, so the future body can land without leaking the dependency.

See ``docs/design/12_VIEWER.md §10.2``.
"""

from __future__ import annotations

from typing import Any


_FUTURE_MSG = (
    "OffscreenBackend (headless pyrender-based rendering) is future "
    "work — see docs/design/12_VIEWER.md §10.2. V1 ships only ViserBackend; "
    "the offscreen path lands together with VideoRecorder / "
    "render_trajectory (§10.1)."
)


class OffscreenBackend:
    """Placeholder for the future headless renderer — see §10.2."""

    is_interactive: bool = False
    supports_gui: bool = False

    def __init__(self, *, width: int = 1280, height: int = 720) -> None:
        raise NotImplementedError(_FUTURE_MSG)

    # The remaining RendererBackend methods are listed for IDE
    # completion / type-checker awareness. None of them are reachable
    # because ``__init__`` raises first.

    def add_mesh(self, name: str, vertices: Any, faces: Any, *, rgba: Any = None,
                 parent: Any = None) -> None:
        raise NotImplementedError(_FUTURE_MSG)

    def add_sphere(self, name: str, *, radius: float, rgba: Any,
                   parent: Any = None) -> None:
        raise NotImplementedError(_FUTURE_MSG)

    def add_cylinder(self, name: str, *, radius: float, length: float,
                     rgba: Any, parent: Any = None) -> None:
        raise NotImplementedError(_FUTURE_MSG)

    def add_capsule(self, name: str, *, radius: float, length: float,
                    rgba: Any, parent: Any = None) -> None:
        raise NotImplementedError(_FUTURE_MSG)

    def add_frame(self, name: str, *, axes_length: float = 0.1) -> None:
        raise NotImplementedError(_FUTURE_MSG)

    def remove(self, name: str) -> None:
        raise NotImplementedError(_FUTURE_MSG)

    def set_transform(self, name: str, pose: Any) -> None:
        raise NotImplementedError(_FUTURE_MSG)

    def set_visible(self, name: str, visible: bool) -> None:
        raise NotImplementedError(_FUTURE_MSG)

    def set_camera(self, camera: Any) -> None:
        raise NotImplementedError(_FUTURE_MSG)

    def capture_frame(self) -> Any:
        raise NotImplementedError(_FUTURE_MSG)
