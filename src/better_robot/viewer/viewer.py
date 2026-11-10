"""``Viewer``: a viser server, groups of layers, one timeline, and the sidebar."""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any, Literal, TYPE_CHECKING, final, override

import numpy as np

from ..exceptions import ArgumentError, ShapeError, UnknownNameError
from ._extras import needs_extra
from .layer import Layer
from .layers.plot import PlotPanel
from .theme import LIGHT, Theme
from .timeline import Timeline

if TYPE_CHECKING:
    from viser import SceneClickEvent, SceneRectSelectEvent

    PointerCallback = Callable[[SceneClickEvent], None] | Callable[[SceneRectSelectEvent], None]

#: What the camera fits when no visible layer has points: a 2 x 2 x 1 m box standing on the origin.
_EMPTY_SCENE_POINTS = np.array([[1.0, 1.0, 1.0], [-1.0, -1.0, 0.0]])


@final
@dataclass
class _Entry:
    layer: Layer
    root: str  # the scene path ``/<group>/<layer.name>`` the layer builds under
    node: Any  # the frame at ``root``; hidden on samples the layer's ``valid`` marks invalid
    times: np.ndarray | None = None  # ``add(times=)``: sample times (nsteps,) in seconds; None = one sample per step
    step: int = -1  # last step pushed; -1 = never
    checkbox: Any = None  # the layer's row in the sidebar's Layers folder


@final
@dataclass
class _Group:
    node: Any
    entries: dict[str, _Entry] = field(default_factory=dict)
    offset: tuple[float, float, float] | None = None


@final
class Viewer:
    """Interactive viewer: ``add`` layers into groups, ``show`` a step, ``play`` the timeline.

    Every layer lives under the scene node ``/<group>/<layer.name>``; a group is the unit for offsets
    and teardown (``remove_group``). ``show(k)`` is the only place steps are pushed:
    visible layers receive ``set_step(_index(layer, k))`` inside one ``server.atomic()``
    block, and layers that already show that step are skipped.

    ``server`` is the underlying ``viser.ViserServer`` (escape hatch for custom GUI); pass one
    in to reuse a server or a fake in tests (``testing.FakeServer``, hence typed ``Any``). The default host is
    loopback only.
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
            with needs_extra("viewer"):
                import viser  # noqa: PLC0415 - optional dependency, imported at the boundary

            server = viser.ViserServer(host=host, port=port, verbose=False)
            # viser moves to the next free port when ``port`` is taken; ``url`` (and record) must follow it.
            port = server.get_port()
        self.server = server
        self.host, self.port = host, port
        self.theme = theme or LIGHT
        self.theme.apply(server)
        self.timeline = Timeline(1, fps=fps)
        self._pointer: tuple[str, PointerCallback] | None = None  # (kind, callback) held by on_pointer
        self.timeline.on_step(lambda step: self._push_all(step, force=False))
        self._groups: dict[str, _Group] = {}
        self._lock = threading.RLock()  # the playback thread pushes while the main thread adds/removes
        self._timeline_folder: Any = None  # built when the first multi-step layer arrives
        self._layers_folder: Any = None

    @override
    def __repr__(self) -> str:
        return (
            f"Viewer({self.url}, groups={list(self._groups)}, layers={len(self.layers)}, nsteps={self.timeline.nsteps})"
        )

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
        layers: Iterable[Layer],
        *,
        group: str = "scene",
        offset: tuple[float, float, float] | None = None,
        times: np.ndarray | Sequence[float] | None = None,
        valid: np.ndarray | None = None,
    ) -> None:
        """Build each of ``layers`` under ``/<group>/<layer.name>`` and push the current step to it.

        One layer is ``add([layer])``. ``offset`` moves the whole group (side-by-side comparisons). ``times`` ``(T,)``
        gives the non-decreasing sample times in seconds of every layer of the call (``T = layer.nsteps``); timeline
        step ``k`` then shows the latest sample at or before ``k / fps``. ``None`` keeps one sample per step.
        ``valid`` ``(T,)`` bool hides the layers on the samples where it is false (a person not tracked in that
        frame); ``(T, N)`` hides single elements of a layer that draws ``N`` (``SpheresLayer``, ``ArrowsLayer``). It
        is kept by reference, like the layers' arrays. Nothing is added when any layer is refused, or raises while it
        builds or shows its first step: that error propagates and the viewer is as it was.
        """
        if isinstance(layers, (Layer, str)) or not isinstance(layers, Iterable):
            hint = f"[{type(layers).__name__}(...)]" if isinstance(layers, Layer) else "[layer, ...]"
            raise ArgumentError(f"add takes an iterable of layers; pass {hint}, got {type(layers).__name__}")
        layers = list(layers)
        if times is not None:
            times = np.asarray(times, dtype=np.float64)
            if not np.all(np.isfinite(times)):
                raise ArgumentError("times must be finite")
            if np.any(np.diff(times) < 0):
                raise ArgumentError("times must be non-decreasing")
        if valid is not None:
            valid = np.asarray(valid, dtype=bool)
        self._check_layers(layers, group, times, valid)
        with self._lock:
            group_state = self._groups.get(group)
            new_group = group_state is None
            if group_state is None:
                group_state = _Group(self.server.scene.add_frame(f"/{group}", show_axes=False))
            placed: list[_Entry] = []
            try:
                for layer in layers:
                    # Always assigned, so a re-added layer never keeps the validity of an earlier add.
                    layer.valid = valid
                    placed.append(self._place(layer, group, times))
            except BaseException:
                for entry in placed:
                    self._teardown(entry)
                if new_group:
                    group_state.node.remove()
                raise
            self._groups[group] = group_state
            group_state.entries.update((entry.layer.name, entry) for entry in placed)
            if offset is not None:
                group_state.node.position = offset
                group_state.offset = offset
            for entry in placed:
                self.timeline.extend(self._span(entry))
            self._timeline_gui()

    def _check_layers(
        self, layers: list[Layer], group: str, times: np.ndarray | None, valid: np.ndarray | None
    ) -> None:
        """Refuse ``add``'s layers before anything is built: a layer already present, a name taken in ``group``, or
        ``times`` / ``valid`` of the wrong shape."""
        taken = set(self._groups[group].entries) if group in self._groups else set()
        for index, layer in enumerate(layers):
            if not isinstance(layer, Layer):
                raise ArgumentError(f"add takes Layer objects, got {type(layer).__name__} at index {index}")
            if self._holds(layer) or any(layer is other for other in layers[:index]):
                raise ArgumentError(
                    f"layer {layer.name!r} is already in this viewer; a layer draws in one place, so build a new one"
                )
            if layer.name in taken:
                raise ArgumentError(
                    f"group {group!r} already has a layer named {layer.name!r}; give the new layer another name"
                )
            taken.add(layer.name)
            if times is not None and times.shape != (layer.nsteps,):
                raise ShapeError(f"times must have shape ({layer.nsteps},) for layer {layer.name!r}, got {times.shape}")
            if valid is not None:
                shapes = [(layer.nsteps,)] + ([(layer.nsteps, layer.elements)] if layer.elements is not None else [])
                if valid.shape not in shapes:
                    raise ShapeError(
                        f"valid must have shape {' or '.join(map(str, shapes))} for layer {layer.name!r}, "
                        f"got {valid.shape}"
                    )

    def _place(self, layer: Layer, group: str, times: np.ndarray | None, *, order: float | None = None) -> _Entry:
        """Build ``layer`` under ``/<group>/<layer.name>`` and push the current step to it, or remove what it built and
        re-raise."""
        layer.apply_theme(self.theme)
        root = f"/{group}/{layer.name}"
        entry = _Entry(layer, root, self.server.scene.add_frame(root, show_axes=False), times)
        try:
            self._build(entry, group, order=order)
            self._push(entry, self.timeline.step)
        except BaseException:
            self._teardown(entry)
            raise
        return entry

    def _timeline_gui(self) -> None:
        """Build the Timeline folder once the timeline has more than one step."""
        if self.timeline.nsteps > 1 and self._timeline_folder is None:
            self._timeline_folder = self.server.gui.add_folder("Timeline")
            with self._timeline_folder:
                self.timeline.build_gui(self.server.gui)

    def add_plot(
        self, name: str, series: np.ndarray, *, labels: list[str] | None = None, group: str = "plots"
    ) -> PlotPanel:
        """Sidebar chart of ``series (T, K)`` against time with a marker at the current step."""
        panel = PlotPanel(name, series, labels=labels, fps=self.timeline.fps)
        self.add([panel], group=group)
        return panel

    def remove_group(self, group: str) -> None:
        """Dispose every layer of ``group``, remove its scene node and GUI folder, and fit the timeline to the rest.

        The timeline shrinks to the longest remaining layer (one step when none is left), so a clip browser can
        swap one clip's group for the next.
        """
        with self._lock:
            group_state = self._group(group)
            del self._groups[group]
            for entry in group_state.entries.values():
                self._teardown(entry)
            group_state.node.remove()
            self._fit_timeline()

    def set_offset(self, group: str, offset: tuple[float, float, float]) -> None:
        """Move the whole ``group`` to ``offset``, as ``add(..., offset=)`` does, e.g. from a side-by-side slider."""
        with self._lock:
            group_state = self._group(group)
            group_state.node.position = offset
            group_state.offset = offset

    def set_fps(self, fps: float) -> None:
        """Play the timeline at ``fps`` steps per second, such as the frame rate of the clip on screen.

        The current step is kept and playback continues. Layers added with ``times`` are mapped onto the new
        step length and the timeline is fitted to them; a `PlotPanel` keeps the time axis it was built with.
        """
        if not fps > 0.0:
            raise ArgumentError(f"fps must be positive, got {fps}")
        playing = self.timeline.playing
        self.timeline.pause()  # outside the lock: the playback thread takes it to push steps
        with self._lock:
            self.timeline.fps = float(fps)
            self._fit_timeline()
            self._push_all(self.timeline.step, force=False)
        if playing:
            self.timeline.play()

    def set_visible(self, layer: Layer, visible: bool) -> None:  # noqa: FBT001 - the switch is the main operand
        """Hide a layer (skipped by ``show``) or show it, catching up to the current step."""
        with self._lock:
            entry = self._entry(layer)
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
        if step is not None and not isinstance(step, int | np.integer):
            raise ArgumentError(f"step must be an int, got {step!r}")
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
        camera: tuple[np.ndarray | Sequence[float], np.ndarray | Sequence[float]] | None = None,
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
            points = self._points(steps)
            camera = fit_camera(points if len(points) else _EMPTY_SCENE_POINTS, up=self.theme.up_vector)
        width, height = size
        frames = []
        with HeadlessClient(self, size=size, gpu=gpu) as client:
            client.render(height, width)  # the first render returns once the scene has reached the browser
            client.set_camera(*camera, up=self.theme.up_vector)
            for step in steps:
                self.show(step)
                frames.append(client.render(height, width))
        return write_frames(frames, path, fps=fps or self.timeline.fps)

    def aim(
        self,
        *,
        steps: Sequence[int] | None = None,
        azimuth: float = math.radians(35.0),
        elevation: float = math.radians(25.0),
    ) -> tuple[np.ndarray, np.ndarray]:
        """Point the connected browsers and every new one (viser's initial camera, also "Reset View") at the visible
        layers: ``record``'s 3/4 view, upright along the theme's ``up`` axis and seen from ``azimuth`` and
        ``elevation`` (radians, see ``capture.fit_camera``), fitted to the bounding box of the visible layers'
        ``world_points`` at each of ``steps`` (default: the current step; steps with nothing visible are skipped). The
        browsers take the field of view the fit assumes. Returns ``(position, look_at)``.
        """
        from .capture import FOV, fit_camera  # noqa: PLC0415 - the capture module is the opt-in browser path

        corners = []
        with self._lock:
            for step in [self.timeline.step] if steps is None else steps:
                points = self._points([step])
                if not len(points):  # nothing visible at this step: it does not pull the view to the origin
                    continue
                # A step's box corners stand in for its points: fitting every vertex of a mesh clip takes minutes.
                corners.append(np.stack(np.meshgrid(*zip(points.min(0), points.max(0))), -1).reshape(-1, 3))
        position, look_at = fit_camera(
            np.concatenate(corners) if corners else _EMPTY_SCENE_POINTS,
            up=self.theme.up_vector,
            azimuth=azimuth,
            elevation=elevation,
        )
        for camera in [self.server.initial_camera, *(client.camera for client in self.server.get_clients().values())]:
            camera.fov = FOV
            camera.position = position  # before look_at: a client's position drags its look_at along
            camera.look_at = look_at
        return position, look_at

    def _points(self, steps: Sequence[int]) -> np.ndarray:
        """``Layer.world_points`` ``(N, 3)`` of the visible layers over ``steps``, group offsets applied; ``N = 0`` when
        nothing visible has points."""
        points = []
        for group in self._groups.values():
            offset = np.zeros(3) if group.offset is None else np.asarray(group.offset, dtype=np.float64)
            for entry in group.entries.values():
                if not entry.layer.visible:
                    continue
                for step in steps:
                    index = self._index(entry, step)
                    valid = entry.layer.valid
                    if valid is not None and valid.ndim == 1 and not valid[index]:
                        continue
                    layer_points = entry.layer.world_points(index)
                    if layer_points is not None and len(layer_points):
                        points.append(np.asarray(layer_points, dtype=np.float64).reshape(-1, 3) + offset)
        return np.concatenate(points) if points else np.empty((0, 3))

    def rebuild(self, layer: Layer) -> None:
        """Dispose ``layer``'s scene nodes and GUI and build them again, keeping its visibility and the current step.

        Use this after changing what a layer draws structurally (e.g. a new mesh topology), not for per-step data.
        """
        with self._lock:
            entry = self._entry(layer)
            group = next(name for name, state in self._groups.items() if state.entries.get(layer.name) is entry)
            layer.dispose()
            order = None if entry.checkbox is None else entry.checkbox.order
            if entry.checkbox is not None:
                entry.checkbox.remove()
                entry.checkbox = None
            self._build(entry, group, order=order)
            if layer.visible:
                self._push(entry, self.timeline.step)

    def replace(self, old: Layer, new: Layer) -> None:
        """Put ``new`` where ``old`` is: same group and Layers row, ``old``'s visibility, the current step pushed.

        ``old``'s scene nodes and controls go; ``new`` is built under ``/<group>/<new.name>`` and the timeline is fitted
        to the layers now present. ``new`` arrives without ``times`` or ``valid`` (``add``'s per-layer options); a
        new name must be free in the group, and ``new`` must not be in the viewer yet (``rebuild`` redraws a layer in
        place). Use it to swap the data a layer draws, such as a body switched in a studio. When ``new`` raises while it
        builds or shows its first step, ``old`` is drawn again and the error propagates.
        """
        with self._lock:
            entry = self._entry(old)
            if self._holds(new):
                raise ArgumentError(
                    f"layer {new.name!r} is already in this viewer; replace takes a new layer object "
                    "(rebuild redraws a layer in place)"
                )
            group = next(name for name, state in self._groups.items() if state.entries.get(old.name) is entry)
            entries = self._groups[group].entries
            if new.name != old.name and new.name in entries:
                raise ArgumentError(
                    f"group {group!r} already has a layer named {new.name!r}; give the new layer another name"
                )
            order = None if entry.checkbox is None else entry.checkbox.order
            # viser keys scene nodes by path, so ``new`` cannot be built beside ``old`` under the same name: ``old``
            # goes first and comes back when ``new`` fails.
            self._teardown(entry)
            new.valid = None
            new.set_visible(old.visible)
            try:
                replacement = self._place(new, group, None, order=order)
            except BaseException:
                entries[old.name] = self._place(old, group, entry.times, order=order)
                raise
            self._groups[group].entries = {
                (new.name if name == old.name else name): (replacement if name == old.name else value)
                for name, value in entries.items()
            }
            self._fit_timeline()
            self._timeline_gui()

    def play(self, *, block: bool = True) -> None:
        """Start playback (a no-op for a single-step scene); with ``block`` keep serving until interrupted, then pause."""
        self.timeline.play()
        if block:
            try:
                self.server.sleep_forever()
            finally:
                # A playback thread still running at interpreter exit dies inside torch and aborts the process.
                self.timeline.pause()

    def _build(self, entry: _Entry, group: str, *, order: float | None = None) -> None:
        """Build ``entry``'s layer and, for a listed layer, its Layers checkbox (at ``order`` when it replaces one)."""
        layer = entry.layer
        layer.build(self.server.scene, entry.root)
        if layer.listed:
            if self._layers_folder is None:
                self._layers_folder = self.server.gui.add_folder("Layers")
            label = layer.name if group == "scene" else f"{group}/{layer.name}"
            with self._layers_folder:
                checkbox = entry.checkbox = self.server.gui.add_checkbox(
                    label, initial_value=layer.visible, order=order
                )
                checkbox.on_update(lambda _: self.set_visible(layer, bool(checkbox.value)))
                layer.build_gui(self.server.gui)
        else:
            layer.build_gui(self.server.gui)
        if not layer.visible:
            layer.set_visible(False)  # new handles start shown

    def _group(self, group: str) -> _Group:
        if group not in self._groups:
            raise UnknownNameError("group", group, valid=list(self._groups))
        return self._groups[group]

    def _teardown(self, entry: _Entry) -> None:
        """Remove ``entry``'s scene nodes, frame and checkbox."""
        entry.layer.dispose()
        if entry.checkbox is not None:
            entry.checkbox.remove()
        entry.node.remove()

    def _entry(self, layer: Layer) -> _Entry:
        for group in self._groups.values():
            for entry in group.entries.values():
                if entry.layer is layer:
                    return entry
        raise ArgumentError(f"layer {layer.name!r} is not in this viewer; add it first")

    def _holds(self, layer: Layer) -> bool:
        """Whether ``layer`` (this object, not its name) is in the viewer."""
        return any(entry.layer is layer for group in self._groups.values() for entry in group.entries.values())

    def on_pointer(self, kind: Literal["click", "rect-select"], callback: PointerCallback) -> None:
        """Route scene clicks (``kind="click"``) or box selections (``"rect-select"``) to ``callback``.

        A click arrives as viser's ``SceneClickEvent`` (``ray_origin``, ``ray_direction``, ``screen_pos``), a box as
        its ``SceneRectSelectEvent`` (``screen_min``, ``screen_max``; image coordinates in ``[0, 1]``, top-left
        origin). The viewer holds one pointer callback so tools never fight over the scene: while one is held, a new
        registration raises ``ArgumentError``; a tool that takes the pointer over calls ``clear_pointer`` first.
        """
        if kind not in ("click", "rect-select"):
            raise ArgumentError(f"kind must be 'click' or 'rect-select', got {kind!r}")
        if self._pointer is not None:
            raise ArgumentError(
                f"the viewer already holds a {self._pointer[0]!r} callback; call clear_pointer() before registering "
                "another"
            )
        scene = self.server.scene
        (scene.on_click if kind == "click" else scene.on_rect_select)()(callback)
        self._pointer = (kind, callback)

    def clear_pointer(self) -> None:
        """Release the pointer callback (no-op when nothing is registered)."""
        if self._pointer is not None:
            kind, callback = self._pointer
            scene = self.server.scene
            (scene.remove_click_callback if kind == "click" else scene.remove_rect_select_callback)(callback)
            self._pointer = None

    def _fit_timeline(self) -> None:
        """Size the timeline to the longest layer, one step when there is none."""
        spans = (self._span(entry) for group in self._groups.values() for entry in group.entries.values())
        self.timeline.resize(max(spans, default=1))

    def _span(self, entry: _Entry) -> int:
        """Timeline steps ``entry``'s layer needs: its sample count, or enough steps to reach its last time."""
        if entry.times is None:
            return entry.layer.nsteps
        return int(np.floor(entry.times[-1] * self.timeline.fps + 1e-9)) + 1

    def _index(self, entry: _Entry, step: int) -> int:
        """Sample of ``entry``'s layer shown at timeline ``step``: the step itself, or the nearest earlier time."""
        if entry.times is None:
            return min(step, entry.layer.nsteps - 1)
        # The tolerance keeps a sample stamped exactly on a step from rounding to the previous one.
        index = int(np.searchsorted(entry.times, step / self.timeline.fps + 1e-9, side="right")) - 1
        return min(max(index, 0), entry.layer.nsteps - 1)

    def _push(self, entry: _Entry, step: int, *, force: bool = True) -> None:
        step = self._index(entry, step)
        if force or step != entry.step:
            entry.layer.set_step(step)
            entry.step = step
            valid = entry.layer.valid
            # Hidden on an invalid sample through its frame, so the layer's own switch stays on.
            shown = valid is None or valid.ndim != 1 or bool(valid[step])
            if entry.node.visible != shown:
                entry.node.visible = shown

    def _push_all(self, step: int, *, force: bool) -> None:
        with self._lock, self.server.atomic():
            for group in self._groups.values():
                for entry in group.entries.values():
                    if entry.layer.visible:
                        self._push(entry, step, force=force)
