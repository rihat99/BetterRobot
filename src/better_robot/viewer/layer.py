"""``Layer``: one visual fed by preloaded ``(T, ...)`` arrays, drawn under one scene node."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, TYPE_CHECKING, override

import numpy as np
import torch

from ..exceptions import ArgumentError, ShapeError
from ..lie import so3

if TYPE_CHECKING:
    from .sidebar import Sidebar
    from .theme import Theme
    from .viewer import Viewer

Bounds = tuple[np.ndarray, np.ndarray]


def as_frames(
    array: np.ndarray | object, event_ndim: int, *, name: str = "array", width: int | None = None
) -> np.ndarray:
    """Promote an array to ``(T, ...)`` float32: a static array (no leading axis) becomes ``T = 1``.

    A C-contiguous float32 array comes back without a copy (a view, for static input), so a layer shares the caller's
    memory; any other input is converted once.

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


def from_z(direction: np.ndarray) -> np.ndarray:
    """Shortest-arc ``xyzw`` rotations ``(..., 4)`` taking ``+z`` onto each ``direction (..., 3)`` (``so3.exp``).

    A zero direction gives the identity, one opposite ``+z`` a half turn about ``x``.
    """
    unit = torch.from_numpy(np.asarray(direction, dtype=np.float64))
    norm = torch.linalg.norm(unit, dim=-1, keepdim=True)
    unit = torch.where(norm > 1e-12, unit / norm.clamp_min(1e-12), unit.new_tensor([0.0, 0.0, 1.0]))
    turn = torch.stack([-unit[..., 1], unit[..., 0], torch.zeros_like(unit[..., 0])], dim=-1)  # z x unit
    sine = torch.linalg.norm(turn, dim=-1, keepdim=True)
    axis = torch.where(sine > 1e-9, turn / sine.clamp_min(1e-9), unit.new_tensor([1.0, 0.0, 0.0]))
    return so3.exp(axis * torch.atan2(sine, unit[..., 2:])).numpy()


def as_triangles(faces: np.ndarray, *, vertices: int) -> np.ndarray:
    """``faces`` as contiguous uint32 ``(F, 3)`` indices into ``vertices`` vertices; another shape raises
    ``ShapeError``, an index outside ``[0, vertices)`` ``ArgumentError`` (the browser would draw it from garbage)."""
    index = np.asarray(faces)
    if index.ndim != 2 or index.shape[1] != 3:
        raise ShapeError(f"faces must be (F, 3) vertex indices of triangles, got {index.shape}")
    if index.size and (np.min(index) < 0 or np.max(index) >= vertices):
        raise ArgumentError(
            f"faces must index the {vertices} vertices, from 0 to {vertices - 1}; got {np.min(index)} to {np.max(index)}"
        )
    return np.ascontiguousarray(index, dtype=np.uint32)


class Layer(ABC):
    """Base class for everything the viewer draws.

    A layer owns its preloaded arrays and the viser handles it creates in ``build``. The viewer
    calls ``set_step(k)`` to write slice ``k`` into those handles; a static layer has
    ``nsteps == 1`` and is pushed once. The arrays are the layer's float32 ``(T, ...)`` attributes
    (``PointsLayer.points``, ``FramesLayer.poses``, ...) and ``SpheresLayer.radii``, read at each push: edit one in
    place and call ``redraw`` (or ``Viewer.show()``) to update the scene; a skin's rest arrays and a chart's
    ``series`` are sent once, at ``build``. An input that already is a C-contiguous float32 array is that attribute,
    no copy, which keeps a big cloud single; any other input (a strided view too) is converted once, so edit the
    attribute. Handles registered with ``_track`` get the default
    ``set_visible`` and ``dispose``; ``visible`` and ``set_visible`` are the one way to read and switch a layer, and
    an override of ``set_visible`` or ``dispose`` calls the base method.

    ``tags`` name the groups the layer belongs to (default: its name); the first is its scene parent
    ``/<tag>/<name>``, and ``Viewer.remove``, ``set_visible`` and ``set_pose`` address every layer of a tag.
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
        """Box ``(lo (3,), hi (3,))`` of what the layer draws at ``step``, in the frame of its first tag (the world
        unless ``Viewer.set_pose`` placed the tag), which ``Viewer.aim`` frames.

        A layer of thin, spread-out parts may return one box per part, ``(lo (N, 3), hi (N, 3))``: ``aim`` frames the
        corners of every box, and one box round a robot is twice what the robot covers on screen. ``None`` (the
        default) keeps the layer out of the framing, as for the ground grid. The automatic aim of ``Viewer.add`` reads
        a sample again only after ``redraw``, ``Viewer.show()`` or ``Viewer.set_visible``, the pushes that follow an
        edit of the arrays.
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

    def _surface(self) -> dict[str, Any]:
        """The settings a mesh node is drawn with at the layer's ``opacity``: viser's ``opacity``, and no shadow cast
        or caught by a see-through surface. The shadow map ignores opacity: a glass at 0.2 would throw a solid shadow
        and wear the shadows of what it holds."""
        solid = self.opacity >= 1.0
        return {"opacity": None if solid else self.opacity, "cast_shadow": solid, "receive_shadow": solid}

    def _fade(self, opacity: float, surfaces: list[Any] | None = None) -> None:
        """``set_opacity`` of a mesh layer: keep ``opacity`` and write ``_surface`` to ``surfaces`` (every tracked
        handle when ``None``)."""
        self._store_opacity(opacity)
        for handle in self._handles if surfaces is None else surfaces:
            for key, value in self._surface().items():
                setattr(handle, key, value)
