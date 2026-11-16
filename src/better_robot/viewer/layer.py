"""``Layer``: one visual fed by preloaded ``(T, ...)`` arrays, drawn under one scene node."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, TYPE_CHECKING, override

import numpy as np

from ..exceptions import ArgumentError, ShapeError

if TYPE_CHECKING:
    from .sidebar import Sidebar
    from .theme import Theme
    from .viewer import Viewer

Bounds = tuple[np.ndarray, np.ndarray]


def as_frames(
    array: np.ndarray | object, event_ndim: int, *, name: str = "array", width: int | None = None
) -> np.ndarray:
    """Promote an array to ``(T, ...)`` float32: a static array (no leading axis) becomes ``T = 1``.

    ``event_ndim`` is the number of trailing axes of one step, e.g. 2 for joints ``(J, 3)``; ``width``, when given,
    is the size of the last axis. A wrong shape raises ``ShapeError`` naming the argument ``name``.
    """
    out = np.asarray(array, dtype=np.float32)
    if out.ndim == event_ndim:
        out = out[None]
    if out.ndim != event_ndim + 1 or (width is not None and out.shape[-1] != width):
        last = "" if width is None else f" with a last axis of {width}"
        raise ShapeError(f"{name} must have {event_ndim} or {event_ndim + 1} axes{last}, got shape {out.shape}")
    return np.ascontiguousarray(out)


def as_tags(tags: str | Sequence[str]) -> tuple[str, ...]:
    """``tags`` as a tuple of names; a string is one tag."""
    return (tags,) if isinstance(tags, str) else tuple(tags)


def box(points: np.ndarray) -> Bounds | None:
    """World box ``(lo (3,), hi (3,))`` of ``points (..., 3)``; ``None`` when there is no point."""
    flat = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if not len(flat):
        return None
    return flat.min(axis=0), flat.max(axis=0)


def as_triangles(faces: np.ndarray) -> np.ndarray:
    """``faces`` as contiguous uint32 ``(F, 3)`` vertex indices of triangles; another shape raises ``ShapeError``."""
    out = np.ascontiguousarray(np.asarray(faces, dtype=np.uint32))
    if out.ndim != 2 or out.shape[1] != 3:
        raise ShapeError(f"faces must be (F, 3) vertex indices of triangles, got {out.shape}")
    return out


class Layer(ABC):
    """Base class for everything the viewer draws.

    A layer owns its preloaded arrays and the viser handles it creates in ``build``. The viewer
    calls ``set_step(k)`` to write slice ``k`` into those handles; a static layer has
    ``nsteps == 1`` and is pushed once. Handles registered with ``_track`` get the default
    ``set_visible`` and ``dispose``; ``visible`` and ``set_visible`` are the one way to read and switch a layer, and
    an override of ``set_visible`` or ``dispose`` calls the base method.

    ``tags`` name the groups the layer belongs to (default: its name); the first is its scene parent
    ``/<tag>/<name>``, and ``Viewer.remove``, ``set_visible`` and ``set_offset`` address every layer of a tag.
    Subclasses implement ``build`` and ``set_step``; ``build_gui``, ``apply_theme``, ``bounds`` and, for a layer that
    draws surfaces, ``set_opacity`` are optional. ``scene`` is viser's ``SceneApi`` or the stand-in of
    ``testing.FakeServer``, typed ``Any`` because only ``viewer.py`` imports viser.
    """

    #: Per-sample validity, set by ``Viewer.add(valid=...)``: ``(nsteps,)`` bool hides the whole layer on invalid
    #: samples, ``(nsteps, N)`` hides single elements; ``None`` = always shown.
    valid: np.ndarray | None = None
    #: The viewer the layer is in, set by ``Viewer.add`` and cleared by ``Viewer.remove``; ``redraw`` goes through it.
    viewer: Viewer | None = None
    _elements: int | None = None  # set by a layer that hides single elements; read through ``elements``
    #: Opacity in ``(0, 1]`` (1 is opaque) of the surfaces a mesh layer draws; ``set_opacity`` changes it.
    opacity: float = 1.0

    def __init__(self, name: str, nsteps: int = 1, *, tags: str | Sequence[str] = ()) -> None:
        if nsteps < 1:
            raise ArgumentError(f"layer {name!r}: nsteps must be at least 1, got {nsteps}")
        self.name = name
        self.nsteps = nsteps
        self.tags = as_tags(tags) or (name,)
        self._handles: list[Any] = []
        self._visible = True

    @override
    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name!r}, nsteps={self.nsteps}, visible={self.visible})"

    @property
    def visible(self) -> bool:
        """Whether the layer is switched on; ``Viewer.set_visible`` switches it and catches it up to the current step."""
        return self._visible

    @property
    def elements(self) -> int | None:
        """``N`` when the layer draws ``N`` elements that a ``(nsteps, N)`` ``valid`` hides one by one, else ``None``."""
        return self._elements

    @abstractmethod
    def build(self, scene: Any, root: str) -> None:
        """Create the scene nodes under ``root`` (a viser scene path such as ``/panda/robot``)."""

    @abstractmethod
    def set_step(self, step: int) -> None:
        """Write step ``step`` (already clamped to ``nsteps - 1``) into the existing handles."""

    def build_gui(self, sidebar: Sidebar) -> None:
        """Add the layer's controls to the viewer's ``Sidebar``; nothing by default.

        Controls are shared by ``(section, label)``: two layers asking for the same one drive it together, and a
        control goes when the last layer that asked for it is removed.
        """

    def apply_theme(self, theme: Theme) -> None:
        """Adopt the viewer's ``Theme``: called before ``build`` and again on ``Viewer.set_theme``; nothing by default."""

    def bounds(self, step: int) -> Bounds | None:
        """World box ``(lo (3,), hi (3,))`` of what the layer draws at ``step``, which ``Viewer.aim`` frames.

        ``None`` (the default) keeps the layer out of the framing, as for the ground grid.
        """
        return None

    def redraw(self) -> None:
        """Push the layer's current step to the scene again, after its arrays changed (a live update).

        Does nothing before the layer is added to a viewer or while it is hidden.
        """
        if self.viewer is not None:
            self.viewer.redraw(self)

    def set_visible(self, visible: bool) -> None:  # noqa: FBT001 - the switch is the main operand
        """Show or hide every tracked handle."""
        self._visible = visible
        for handle in self._handles:
            handle.visible = visible

    def set_opacity(self, opacity: float) -> None:
        """Draw the layer's surfaces at ``opacity`` in ``(0, 1]`` (1 is opaque), live and after a rebuild.

        The mesh layers implement it; a layer without surfaces (points, frames, lines, the grid, a panel) raises
        ``ArgumentError``.
        """
        raise ArgumentError(f"{type(self).__name__} {self.name!r} draws no surface to fade; set_opacity is for meshes")

    def dispose(self) -> None:
        """Remove every tracked handle from the scene."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _track(self, handle: Any) -> Any:
        self._handles.append(handle)
        return handle

    def _store_opacity(self, opacity: float) -> None:
        """Keep ``opacity`` once it is in ``(0, 1]``: the one check of the constructors and ``set_opacity``."""
        if not 0.0 < opacity <= 1.0:
            raise ArgumentError(f"opacity must be in (0, 1], got {opacity}")
        self.opacity = float(opacity)

    def _fade(self, opacity: float, surfaces: list[Any] | None = None) -> None:
        """``set_opacity`` of a mesh layer: keep ``opacity`` and write it to ``surfaces`` (every tracked handle when
        ``None``)."""
        self._store_opacity(opacity)
        for handle in self._handles if surfaces is None else surfaces:
            handle.opacity = None if self.opacity >= 1.0 else self.opacity
