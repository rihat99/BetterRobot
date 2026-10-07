"""``Viewer``: a viser server, groups of layers, one timeline, and the sidebar."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .layer import Geometry, Layer
from .layers.plot import PlotPanel
from .theme import LIGHT, Theme
from .timeline import Timeline


@dataclass
class _Entry:
    layer: Layer
    step: int = -1  # last step pushed; -1 = never
    checkbox: Any = None  # the layer's row in the sidebar's Layers folder


@dataclass
class _Group:
    node: Any
    entries: dict[str, _Entry] = field(default_factory=dict)
    offset: tuple[float, float, float] | None = None


class _Shifted(Layer):
    """Read-only view of a layer whose group has an ``offset``, for the offline renderer."""

    def __init__(self, layer: Layer, offset: tuple[float, float, float]) -> None:
        super().__init__(layer.name, layer.nsteps)
        self.fits_camera = layer.fits_camera
        self._layer = layer
        self._offset = np.asarray(offset, dtype=np.float32)

    def build(self, scene: Any, root: str) -> None:
        raise TypeError("a shifted view is only for rendering")

    def set_step(self, step: int) -> None:
        raise TypeError("a shifted view is only for rendering")

    def geometry(self, step: int) -> list[Geometry]:
        return [Geometry(g.vertices + self._offset, g.faces, g.colors) for g in self._layer.geometry(step)]


class Viewer:
    """Interactive viewer: ``add`` layers into groups, ``show`` a step, ``play`` the timeline.

    Every layer lives under the scene node ``/<group>/<layer.name>``; a group is the unit for offsets
    and teardown (``remove_group``). ``show(k)`` is the only place steps are pushed:
    visible layers receive ``set_step(min(k, nsteps - 1))`` inside one ``server.atomic()``
    block, and layers that already show that step are skipped.

    ``server`` is the underlying ``viser.ViserServer`` (escape hatch for custom GUI); pass one
    in to reuse a server or a fake in tests. The default host is loopback only.
    """

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 8080,
        fps: float = 30.0,
        theme: Theme | None = None,
        server: Any | None = None,
    ) -> None:
        if server is None:
            import viser  # noqa: PLC0415 - optional dependency, imported at the boundary

            server = viser.ViserServer(host=host, port=port, verbose=False)
        self.server = server
        self.host, self.port = host, port
        self.theme = theme or LIGHT
        self.theme.apply(server)
        self.timeline = Timeline(1, fps)
        self.timeline.on_step(lambda step: self._push_all(step, force=False))
        self._groups: dict[str, _Group] = {}
        self._lock = threading.RLock()  # the playback thread pushes while the main thread adds/removes
        self._timeline_folder: Any = None  # built when the first multi-step layer arrives
        self._layers_folder: Any = None

    @property
    def url(self) -> str:
        """Address to open in a browser, e.g. ``http://127.0.0.1:8080``."""
        return f"http://{self.host}:{self.port}"

    @property
    def layers(self) -> list[Layer]:
        """Every added layer, visible or not, in group order then insertion order."""
        return [entry.layer for group in self._groups.values() for entry in group.entries.values()]

    def add(self, layer: Layer, group: str = "scene", *, offset: tuple[float, float, float] | None = None) -> Layer:
        """Build ``layer`` under ``/<group>/<layer.name>`` and push the current step to it.

        ``offset`` moves the whole group (side-by-side comparisons). Returns the layer.
        """
        with self._lock:
            return self._add(layer, group, offset)

    def _add(self, layer: Layer, group: str, offset: tuple[float, float, float] | None) -> Layer:
        group_state = self._groups.get(group)
        if group_state is None:
            node = self.server.scene.add_frame(f"/{group}", show_axes=False)
            group_state = self._groups[group] = _Group(node)
        if offset is not None:
            group_state.node.position = offset
            group_state.offset = offset
        if layer.name in group_state.entries:
            raise ValueError(f"group {group!r} already has a layer named {layer.name!r}")
        layer.apply_theme(self.theme)
        layer.root = f"/{group}/{layer.name}"
        entry = group_state.entries[layer.name] = _Entry(layer)
        self._build(entry, group)
        self.timeline.extend(layer.nsteps)
        if self.timeline.nsteps > 1 and self._timeline_folder is None:
            self._timeline_folder = self.server.gui.add_folder("Timeline")
            with self._timeline_folder:
                self.timeline.build_gui(self.server.gui)
        self._push(entry, self.timeline.step)
        return layer

    def add_plot(self, name: str, series: Any, *, labels: list[str] | None = None, group: str = "plots") -> PlotPanel:
        """Sidebar chart of ``series (T, K)`` against time with a marker at the current step."""
        return self.add(PlotPanel(name, series, labels=labels, fps=self.timeline.fps), group=group)  # type: ignore[return-value]

    def remove_group(self, group: str) -> None:
        """Dispose every layer of ``group`` and remove its scene node and GUI folder."""
        with self._lock:
            group_state = self._groups.pop(group)
            for entry in group_state.entries.values():
                entry.layer.dispose()
                if entry.checkbox is not None:
                    entry.checkbox.remove()
            group_state.node.remove()

    def set_visible(self, layer: Layer, visible: bool) -> None:
        """Hide a layer (skipped by ``show``) or show it, catching up to the current step."""
        with self._lock:
            entry = self._entry(layer)
            layer._visible = visible  # also for subclasses whose set_visible skips the base class
            layer.set_visible(visible)
            if entry.checkbox is not None and entry.checkbox.value != visible:
                entry.checkbox.value = visible
            if visible:
                self._push(entry, self.timeline.step)

    def show(self, step: int | None = None) -> None:
        """Move the timeline to ``step`` and push it to every visible layer.

        With no argument, re-push the current step to every visible layer even if it already shows it; call this
        after editing a layer's arrays in place.
        """
        if step is None or step == self.timeline.step:
            self._push_all(self.timeline.step, force=True)
        else:
            self.timeline.step = step

    def save(self, path: str | Path, *, fps: float | None = None) -> Path:
        """Record every step into a ``.viser`` file that plays in viser's static client, no Python needed."""
        path = Path(path)
        fps = fps or self.timeline.fps
        serializer = self.server.get_scene_serializer()
        for step in range(self.timeline.nsteps):
            self._push_all(step, force=True)
            serializer.insert_sleep(1.0 / fps)
        path.write_bytes(serializer.serialize())
        self._push_all(self.timeline.step, force=True)
        return path

    def record(self, path: str | Path, *, fps: float | None = None, **options: Any) -> Path:
        """Render the visible layers offline (no browser) to ``.gif``, ``.mp4`` or ``.png`` steps.

        ``options`` go to ``better_robot.viewer.render.record`` (``camera``, ``size``, ``steps``, ``background``);
        without a ``camera`` the view is fitted to the visible layers sampled over the clip, upright along the
        theme's ``up`` axis. Group offsets apply. Returns the written path.
        """
        from .render import record  # noqa: PLC0415 - keeps torch rasterizer out of the import path

        layers = [
            entry.layer if group.offset is None else _Shifted(entry.layer, group.offset)
            for group in self._groups.values()
            for entry in group.entries.values()
            if entry.layer.visible
        ]
        return record(layers, path, fps=fps or self.timeline.fps, up=self.theme.up_vector, **options)

    def rebuild(self, layer: Layer) -> None:
        """Dispose ``layer``'s scene nodes and GUI and build them again, keeping its visibility and the current step.

        Use this after changing what a layer draws structurally (e.g. a new mesh topology), not for per-step data.
        """
        with self._lock:
            entry = self._entry(layer)
            group = next(name for name, state in self._groups.items() if state.entries.get(layer.name) is entry)
            layer.dispose()
            if entry.checkbox is not None:
                entry.checkbox.remove()
                entry.checkbox = None
            self._build(entry, group)
            layer.set_visible(layer.visible)
            if layer.visible:
                self._push(entry, self.timeline.step)

    def play(self, block: bool = True) -> None:
        """Start playback (a no-op for a single-step scene); with ``block`` keep serving until interrupted."""
        self.timeline.play()
        if block:
            self.server.sleep_forever()

    def _build(self, entry: _Entry, group: str) -> None:
        layer = entry.layer
        layer.build(self.server.scene, layer.root)
        if not layer.listed:
            layer.build_gui(self.server.gui)
            return
        if self._layers_folder is None:
            self._layers_folder = self.server.gui.add_folder("Layers")
        label = layer.name if group == "scene" else f"{group}/{layer.name}"
        with self._layers_folder:
            checkbox = entry.checkbox = self.server.gui.add_checkbox(label, initial_value=layer.visible)
            checkbox.on_update(lambda _: self.set_visible(layer, bool(checkbox.value)))
            layer.build_gui(self.server.gui)

    def _entry(self, layer: Layer) -> _Entry:
        for group in self._groups.values():
            for entry in group.entries.values():
                if entry.layer is layer:
                    return entry
        raise KeyError(f"layer {layer.name!r} is not in this viewer")

    def _push(self, entry: _Entry, step: int, force: bool = True) -> None:
        step = min(step, entry.layer.nsteps - 1)
        if force or step != entry.step:
            entry.layer.set_step(step)
            entry.step = step

    def _push_all(self, step: int, force: bool) -> None:
        with self._lock, self.server.atomic():
            for group in self._groups.values():
                for entry in group.entries.values():
                    if entry.layer.visible:
                        self._push(entry, step, force)
