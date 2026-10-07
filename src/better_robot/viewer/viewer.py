"""``Viewer``: a viser server, groups of layers, one timeline, and the sidebar."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .layer import Layer
from .layers.plot import PlotPanel
from .theme import LIGHT, Theme
from .timeline import Timeline


@dataclass
class _Entry:
    layer: Layer
    visible: bool = True
    frame: int = -1  # last frame pushed; -1 = never


@dataclass
class _Group:
    node: Any
    folder: Any
    entries: dict[str, _Entry] = field(default_factory=dict)


class Viewer:
    """Interactive viewer: ``add`` layers into groups, ``show`` a frame, ``play`` the timeline.

    Every layer lives under the scene node ``/<group>/<layer.name>``; a group is the unit for
    toggling and teardown (``remove_group``). ``show(k)`` is the only place frames are pushed:
    visible layers receive ``set_frame(min(k, num_frames - 1))`` inside one ``server.atomic()``
    block, and layers that already show that frame are skipped.

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
        self.timeline.on_frame(lambda frame: self._push_all(frame, force=False))
        self._groups: dict[str, _Group] = {}
        self._timeline_folder = server.gui.add_folder("Timeline")
        with self._timeline_folder:
            self.timeline.build_gui(server.gui)

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def layers(self) -> list[Layer]:
        return [entry.layer for group in self._groups.values() for entry in group.entries.values()]

    def add(self, layer: Layer, group: str = "scene", *, offset: tuple[float, float, float] | None = None) -> Layer:
        """Build ``layer`` under ``/<group>/<layer.name>`` and push the current frame to it.

        ``offset`` moves the whole group (side-by-side comparisons). Returns the layer.
        """
        group_state = self._groups.get(group)
        if group_state is None:
            node = self.server.scene.add_frame(f"/{group}", show_axes=False)
            folder = self.server.gui.add_folder(group)
            group_state = self._groups[group] = _Group(node, folder)
        if offset is not None:
            group_state.node.position = offset
        if layer.name in group_state.entries:
            raise ValueError(f"group {group!r} already has a layer named {layer.name!r}")
        layer.root = f"/{group}/{layer.name}"
        layer.build(self.server.scene, layer.root)
        entry = group_state.entries[layer.name] = _Entry(layer)
        with group_state.folder:
            checkbox = self.server.gui.add_checkbox(layer.name, initial_value=True)
            checkbox.on_update(lambda _: self.set_visible(layer, bool(checkbox.value)))
            layer.build_gui(self.server.gui)
        self.timeline.extend(layer.num_frames)
        self._push(entry, self.timeline.frame)
        return layer

    def add_plot(self, name: str, series: Any, *, labels: list[str] | None = None, group: str = "plots") -> PlotPanel:
        """Sidebar chart of ``series (T, K)`` against time with a marker at the current frame."""
        return self.add(PlotPanel(name, series, labels=labels, fps=self.timeline.fps), group=group)  # type: ignore[return-value]

    def remove_group(self, group: str) -> None:
        """Dispose every layer of ``group`` and remove its scene node and GUI folder."""
        group_state = self._groups.pop(group)
        for entry in group_state.entries.values():
            entry.layer.dispose()
        group_state.node.remove()
        group_state.folder.remove()

    def set_visible(self, layer: Layer, visible: bool) -> None:
        """Hide a layer (skipped by ``show``) or show it, catching up to the current frame."""
        entry = self._entry(layer)
        entry.visible = visible
        layer.set_visible(visible)
        if visible:
            self._push(entry, self.timeline.frame)

    def show(self, frame: int | None = None) -> None:
        """Move the timeline to ``frame``; with no argument re-push the current frame to every
        visible layer (call this after editing a layer's arrays in place)."""
        if frame is None or frame == self.timeline.frame:
            self._push_all(self.timeline.frame, force=True)
        else:
            self.timeline.frame = frame

    def play(self, block: bool = True) -> None:
        """Start playback; with ``block`` keep the process alive until interrupted."""
        self.timeline.play()
        if block:
            self.server.sleep_forever()

    def _entry(self, layer: Layer) -> _Entry:
        for group in self._groups.values():
            for entry in group.entries.values():
                if entry.layer is layer:
                    return entry
        raise KeyError(f"layer {layer.name!r} is not in this viewer")

    def _push(self, entry: _Entry, frame: int, force: bool = True) -> None:
        frame = min(frame, entry.layer.num_frames - 1)
        if force or frame != entry.frame:
            entry.layer.set_frame(frame)
            entry.frame = frame

    def _push_all(self, frame: int, force: bool) -> None:
        with self.server.atomic():
            for group in self._groups.values():
                for entry in group.entries.values():
                    if entry.visible:
                        self._push(entry, frame, force)
