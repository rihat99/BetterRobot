"""``Viewer``: a viser server, groups of layers, one timeline, and the sidebar."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .layer import Layer
from .layers.plot import PlotPanel
from .theme import LIGHT, Theme
from .timeline import Timeline


@dataclass
class _Entry:
    layer: Layer
    step: int = -1  # last step pushed; -1 = never
    checkbox: Any = None  # the layer's row in the sidebar's Layers folder
    shown: bool = True  # nodes shown; False while a visible layer sits on a sample its ``valid`` marks invalid


@dataclass
class _Group:
    node: Any
    entries: dict[str, _Entry] = field(default_factory=dict)
    offset: tuple[float, float, float] | None = None


class Viewer:
    """Interactive viewer: ``add`` layers into groups, ``show`` a step, ``play`` the timeline.

    Every layer lives under the scene node ``/<group>/<layer.name>``; a group is the unit for offsets
    and teardown (``remove_group``). ``show(k)`` is the only place steps are pushed:
    visible layers receive ``set_step(_index(layer, k))`` inside one ``server.atomic()``
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
            # viser moves to the next free port when ``port`` is taken; ``url`` (and record) must follow it.
            port = server.get_port()
        self.server = server
        self.host, self.port = host, port
        self.theme = theme or LIGHT
        self.theme.apply(server)
        self.timeline = Timeline(1, fps)
        self._pointer: str | None = None
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

    def add(
        self,
        layer: Layer,
        group: str = "scene",
        *,
        offset: tuple[float, float, float] | None = None,
        times: np.ndarray | Sequence[float] | None = None,
        valid: np.ndarray | None = None,
    ) -> Layer:
        """Build ``layer`` under ``/<group>/<layer.name>`` and push the current step to it.

        ``offset`` moves the whole group (side-by-side comparisons). ``times`` ``(T,)`` gives the
        non-decreasing sample times of the layer in seconds (``T = layer.nsteps``); timeline step ``k``
        then shows the latest sample at or before ``k / fps``. ``None`` keeps one sample per step.
        ``valid`` ``(T,)`` bool hides the layer on the samples where it is false (a person not tracked in
        that frame); ``(T, N)`` hides single elements of a layer that draws ``N`` (``SpheresLayer``,
        ``ArrowsLayer``). It is kept by reference, like the layer's arrays. Returns the layer.
        """
        if times is not None:
            times = np.asarray(times, dtype=np.float64)
            if times.shape != (layer.nsteps,):
                raise ValueError(f"times must have shape ({layer.nsteps},), got {times.shape}")
            if not np.all(np.isfinite(times)):
                raise ValueError("times must be finite")
            if np.any(np.diff(times) < 0):
                raise ValueError("times must be non-decreasing")
        if valid is not None:
            valid = np.asarray(valid, dtype=bool)
            shapes = [(layer.nsteps,)] + ([(layer.nsteps, layer._elements)] if layer._elements is not None else [])
            if valid.shape not in shapes:
                raise ValueError(f"valid must have shape {' or '.join(map(str, shapes))}, got {valid.shape}")
        with self._lock:
            return self._add(layer, group, offset, times, valid)

    def _add(
        self,
        layer: Layer,
        group: str,
        offset: tuple[float, float, float] | None,
        times: np.ndarray | None = None,
        valid: np.ndarray | None = None,
    ) -> Layer:
        group_state = self._groups.get(group)
        if group_state is None:
            node = self.server.scene.add_frame(f"/{group}", show_axes=False)
            group_state = self._groups[group] = _Group(node)
        if offset is not None:
            group_state.node.position = offset
            group_state.offset = offset
        if layer.name in group_state.entries:
            raise ValueError(f"group {group!r} already has a layer named {layer.name!r}")
        # Assigned only once the add can no longer be refused, and always, so a re-added layer
        # never keeps the times or validity of an earlier add.
        layer.times, layer.valid = times, valid
        layer.apply_theme(self.theme)
        layer.root = f"/{group}/{layer.name}"
        entry = group_state.entries[layer.name] = _Entry(layer)
        self._build(entry, group)
        self.timeline.extend(self._span(layer))
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
        """Dispose every layer of ``group``, remove its scene node and GUI folder, and fit the timeline to the rest.

        The timeline shrinks to the longest remaining layer (one step when none is left), so a clip browser can
        swap one clip's group for the next.
        """
        with self._lock:
            group_state = self._groups.pop(group)
            for entry in group_state.entries.values():
                entry.layer.dispose()
                if entry.checkbox is not None:
                    entry.checkbox.remove()
            group_state.node.remove()
            self._fit_timeline()

    def set_fps(self, fps: float) -> None:
        """Play the timeline at ``fps`` steps per second, such as the frame rate of the clip on screen.

        The current step is kept and playback continues. Layers added with ``times`` are mapped onto the new
        step length and the timeline is fitted to them; a `PlotPanel` keeps the time axis it was built with.
        """
        if not fps > 0.0:
            raise ValueError(f"fps must be positive, got {fps}")
        playing = self.timeline.playing
        self.timeline.pause()  # outside the lock: the playback thread takes it to push steps
        with self._lock:
            self.timeline.fps = float(fps)
            self._fit_timeline()
            self._push_all(self.timeline.step, force=False)
        if playing:
            self.timeline.play()

    def set_visible(self, layer: Layer, visible: bool) -> None:
        """Hide a layer (skipped by ``show``) or show it, catching up to the current step."""
        with self._lock:
            entry = self._entry(layer)
            layer._visible = visible  # also for subclasses whose set_visible skips the base class
            layer.set_visible(visible)
            entry.shown = visible
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

    def record(
        self,
        path: str | Path,
        *,
        steps: Sequence[int] | None = None,
        size: tuple[int, int] = (960, 540),
        fps: float | None = None,
        gpu: bool = False,
        camera: tuple[Sequence[float], Sequence[float]] | None = None,
    ) -> Path:
        """Screenshot the live scene in a headless Chromium for each step and write ``.webm``, ``.gif`` or PNGs.

        ``size`` is ``(width, height)``; ``steps`` defaults to every step. ``camera`` is ``(position, look_at)``;
        without it a 3/4 view, upright along the theme's ``up`` axis, is fitted to the visible layers over the clip.
        ``path`` ending in ``.png`` writes ``<stem>_000.png, ...``; without a suffix it is a directory of frames.
        Needs a Chromium binary (``$BETTER_ROBOT_CHROME`` or on ``PATH``) and, for ``.webm``, ffmpeg (``$FFMPEG``
        or on ``PATH``). Machines without ``libasound.so.2`` need it on ``LD_LIBRARY_PATH``; see
        ``better_robot.viewer.capture``. Returns ``path``.
        """
        from .capture import HeadlessClient, fit_camera, write_frames  # noqa: PLC0415 - browser path is opt-in

        steps = list(range(self.timeline.nsteps)) if steps is None else list(steps)
        if camera is None:
            camera = fit_camera(self._points(steps), up=self.theme.up_vector)
        width, height = size
        frames = []
        with HeadlessClient(self, size=size, gpu=gpu) as client:
            client.render(height, width)  # the first render returns once the scene has reached the browser
            client.set_camera(*camera, up=self.theme.up_vector)
            for step in steps:
                self.show(step)
                frames.append(client.render(height, width))
        return write_frames(frames, path, fps=fps or self.timeline.fps)

    def _points(self, steps: Sequence[int]) -> np.ndarray:
        """``Layer.world_points`` ``(N, 3)`` of the visible layers over ``steps``, group offsets applied."""
        points = []
        for group in self._groups.values():
            offset = np.zeros(3) if group.offset is None else np.asarray(group.offset, dtype=np.float64)
            for entry in group.entries.values():
                if not entry.layer.visible:
                    continue
                for step in steps:
                    index = self._index(entry.layer, step)
                    valid = entry.layer.valid
                    if valid is not None and valid.ndim == 1 and not valid[index]:
                        continue
                    layer_points = entry.layer.world_points(index)
                    if layer_points is not None and len(layer_points):
                        points.append(np.asarray(layer_points, dtype=np.float64).reshape(-1, 3) + offset)
        return np.concatenate(points) if points else np.array([[1.0, 1.0, 1.0], [-1.0, -1.0, 0.0]])

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
            entry.shown = layer.visible
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

    def on_pointer(self, kind: str, callback: Callable[[Any], None]) -> None:
        """Route viser's scene pointer events of ``kind`` (``"click"`` or ``"rect-select"``) to ``callback``.

        viser has one scene-pointer slot; the viewer owns it so tools never fight over it. A new
        registration replaces the previous one.
        """
        if kind not in ("click", "rect-select"):
            raise ValueError(f"kind must be 'click' or 'rect-select', got {kind!r}")
        self.clear_pointer()
        self.server.scene.on_pointer_event(event_type=kind)(callback)
        self._pointer = kind

    def clear_pointer(self) -> None:
        """Release the scene-pointer slot (no-op when nothing is registered)."""
        if self._pointer is not None:
            self.server.scene.remove_pointer_callback()
            self._pointer = None

    def _fit_timeline(self) -> None:
        """Size the timeline to the longest layer, one step when there is none."""
        self.timeline.resize(max((self._span(layer) for layer in self.layers), default=1))

    def _span(self, layer: Layer) -> int:
        """Timeline steps ``layer`` needs: its sample count, or enough steps to reach its last time."""
        if layer.times is None:
            return layer.nsteps
        return int(np.floor(layer.times[-1] * self.timeline.fps + 1e-9)) + 1

    def _index(self, layer: Layer, step: int) -> int:
        """Sample of ``layer`` shown at timeline ``step``: the step itself, or the nearest earlier time."""
        if layer.times is None:
            return min(step, layer.nsteps - 1)
        # The tolerance keeps a sample stamped exactly on a step from rounding to the previous one.
        index = int(np.searchsorted(layer.times, step / self.timeline.fps + 1e-9, side="right")) - 1
        return min(max(index, 0), layer.nsteps - 1)

    def _push(self, entry: _Entry, step: int, force: bool = True) -> None:
        step = self._index(entry.layer, step)
        if force or step != entry.step:
            entry.layer.set_step(step)
            entry.step = step
            valid = entry.layer.valid
            if valid is not None and valid.ndim == 1 and bool(valid[step]) != entry.shown:
                entry.shown = bool(valid[step])
                entry.layer.set_visible(entry.shown)
                entry.layer._visible = True  # hidden for this sample only; the layer's own switch stays on

    def _push_all(self, step: int, force: bool) -> None:
        with self._lock, self.server.atomic():
            for group in self._groups.values():
                for entry in group.entries.values():
                    if entry.layer.visible:
                        self._push(entry, step, force)
