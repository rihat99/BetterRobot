"""``Viewer``: a viser server, tagged layers, one timeline, and the sidebar."""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any, Literal, TYPE_CHECKING, final, override

import numpy as np

from ..exceptions import ArgumentError, ShapeError, UnknownNameError
from ._extras import needs_extra
from .layer import Layer, as_tags
from .layers.ground import GroundLayer
from .layers.image import ImagePanel
from .layers.plot import PlotPanel
from .sidebar import Sidebar
from .theme import LIGHT, Theme
from .timeline import Timeline

if TYPE_CHECKING:
    from viser import SceneClickEvent, SceneRectSelectEvent

    PointerCallback = Callable[[SceneClickEvent], None] | Callable[[SceneRectSelectEvent], None]

#: What the camera frames when no visible layer has bounds: a 2 x 2 x 1 m box standing on the origin.
_EMPTY_SCENE: tuple[np.ndarray, np.ndarray] = (np.array([-1.0, -1.0, 0.0]), np.array([1.0, 1.0, 1.0]))
_FOV = math.radians(40.0)  # the vertical field of view of the browsers' camera
_AIM_STEPS = 32  # at most this many evenly spaced steps are framed by ``aim``
_AIM_FILL = 0.8  # share of a 16:9 view's height (or width) the framed box spans
_AIM_ASPECT = 16.0 / 9.0
#: The default view of ``aim``: from the front-right of an x-forward robot, 30 degrees off its front, and from above.
_AZIMUTH, _ELEVATION = math.radians(60.0), math.radians(25.0)


def _as_times(times: np.ndarray | Sequence[float]) -> np.ndarray:
    """``add``'s ``times`` as float64, once they are finite and non-decreasing."""
    times = np.asarray(times, dtype=np.float64)
    if not np.all(np.isfinite(times)):
        raise ArgumentError("times must be finite")
    if np.any(np.diff(times) < 0):
        raise ArgumentError("times must be non-decreasing")
    return times


def _label(tag: str) -> str:
    """A tag as its Layers checkbox reads, in sentence case: ``"left_arm"`` -> ``"Left arm"``."""
    return tag[:1].upper() + tag[1:].replace("_", " ")


@final
@dataclass
class _Entry:
    layer: Layer
    root: str  # the scene path ``/<first tag>/<layer.name>`` the layer builds under
    node: Any  # the frame at ``root``; hidden on samples the layer's ``valid`` marks invalid
    times: np.ndarray | None = None  # ``add(times=)``: sample times (nsteps,) in seconds; None = one sample per step
    step: int = -1  # last step pushed; -1 = never


@final
@dataclass
class _Tag:
    node: Any  # the frame ``/<tag>``: parent of the layers whose first tag this is
    offset: tuple[float, float, float] | None = None


@final
class Viewer:
    """Interactive viewer: ``add`` layers under tags, ``show`` a step, ``serve`` the page.

    Every layer lives under the scene node ``/<tag>/<layer.name>`` of its first tag; a tag is the unit for offsets,
    visibility and removal (``set_offset``, ``set_visible``, ``remove``), and the sidebar's Layers section has one
    checkbox per tag once there are two. ``show(k)`` is the only place steps are pushed: visible layers receive
    ``set_step(_index(layer, k))`` inside one ``server.atomic()`` block, and layers that already show that step are
    skipped. ``lock`` is the one lock: adding and removing layers, every push and every sidebar callback hold it.

    ``server`` is the underlying ``viser.ViserServer`` (escape hatch for custom GUI); pass one in to reuse a server
    or a fake in tests (``testing.FakeServer``, hence typed ``Any``). The default host is loopback only. ``ground``
    adds a ``GroundLayer`` under the tag ``ground``. Until ``aim`` is called, the camera is aimed at the scene after
    each ``add``.
    """

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 8080,
        fps: float = 30.0,
        theme: Theme = LIGHT,
        ground: bool = True,
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
        self.theme = theme
        self.theme.apply(server)
        self.lock = threading.RLock()  # the playback thread pushes while the main thread adds/removes
        self.sidebar = Sidebar(server, self.lock)
        self.timeline = Timeline(1, fps=fps, lock=self.lock)
        self._pointer: tuple[str, PointerCallback] | None = None  # (kind, callback) held by on_pointer
        self.timeline.on_step(lambda step: self._push_all(step, force=False))
        self._entries: list[_Entry] = []
        self._tags: dict[str, _Tag] = {}  # every tag that is a layer's first tag, by first appearance
        self._boxes: dict[str, Any] = {}  # the Layers section's checkbox per tag
        self._timeline_built = False
        self._aimed = False  # True once the user called aim: ``add`` stops aiming by itself
        if ground:
            self.add(GroundLayer())

    @override
    def __repr__(self) -> str:
        return f"Viewer({self.url}, tags={self.tags}, layers={len(self.layers)}, nsteps={self.timeline.nsteps})"

    @property
    def url(self) -> str:
        """Address to open in a browser, e.g. ``http://127.0.0.1:8080``."""
        return f"http://{self.host}:{self.port}"

    @property
    def layers(self) -> list[Layer]:
        """Every added layer, visible or not, in the order added."""
        return [entry.layer for entry in self._entries]

    @property
    def tags(self) -> list[str]:
        """Every tag of the added layers, in order of first appearance."""
        return list(dict.fromkeys(tag for entry in self._entries for tag in entry.layer.tags))

    def add[Added: Layer | Iterable[Layer]](
        self,
        layers: Added,
        *,
        tags: str | Sequence[str] | None = None,
        offset: tuple[float, float, float] | None = None,
        times: np.ndarray | Sequence[float] | None = None,
        valid: np.ndarray | None = None,
    ) -> Added:
        """Build one layer or each of an iterable of layers under ``/<first tag>/<layer.name>``, push the current
        step to it, and return ``layers`` as given: ``robot = viewer.add(RobotVisual(...))``.

        ``tags`` replaces the tags of every layer of the call (a string is one tag). ``offset`` moves the layers'
        first tags (side-by-side comparisons). ``times`` ``(T,)`` gives the non-decreasing sample times in seconds of
        every layer of the call (``T = layer.nsteps``); timeline step ``k`` then shows the latest sample at or before
        ``k / fps``. ``None`` keeps one sample per step. ``valid`` ``(T,)`` bool hides the layers on the samples
        where it is false (a person not tracked in that frame); ``(T, N)`` hides single elements of a layer that draws
        ``N`` (``SpheresLayer``, ``ArrowsLayer``). It is kept by reference, like the layers' arrays. Nothing is added
        when any layer is refused, or raises while it builds or shows its first step: that error propagates and the
        viewer is as it was.
        """
        if isinstance(layers, str) or not isinstance(layers, Layer | Iterable):
            raise ArgumentError(f"add takes a layer or an iterable of layers, got {type(layers).__name__}")
        batch = [layers] if isinstance(layers, Layer) else list(layers)
        times = None if times is None else _as_times(times)
        valid = None if valid is None else np.asarray(valid, dtype=bool)
        given = None if tags is None else as_tags(tags) or None
        self._check_layers(batch, given, times, valid)
        with self.lock:
            placed: list[_Entry] = []
            try:
                for layer in batch:
                    if given is not None:
                        layer.tags = given
                    # Always assigned, so a re-added layer never keeps the validity of an earlier add.
                    layer.valid = valid
                    placed.append(self._place(layer, times))
            except BaseException:
                for entry in reversed(placed):
                    self._teardown(entry)
                    self._entries.remove(entry)
                self._prune_tags()
                raise
            if offset is not None:
                for entry in placed:
                    self.set_offset(entry.layer.tags[0], offset)
            for entry in placed:
                self.timeline.extend(self._span(entry))
            self._timeline_gui()
            self._sync_layers_section()
            if not self._aimed:
                self._point_cameras(*self._framing(None, _AZIMUTH, _ELEVATION, _FOV), _FOV)
        return layers

    def _check_layers(
        self, layers: list[Layer], tags: tuple[str, ...] | None, times: np.ndarray | None, valid: np.ndarray | None
    ) -> None:
        """Refuse ``add``'s layers before anything is built: a layer already present, a name taken under its first
        tag, or ``times`` / ``valid`` of the wrong shape."""
        taken = {entry.root for entry in self._entries}
        for index, layer in enumerate(layers):
            if not isinstance(layer, Layer):
                raise ArgumentError(f"add takes Layer objects, got {type(layer).__name__} at index {index}")
            if self._holds(layer) or any(layer is other for other in layers[:index]):
                raise ArgumentError(
                    f"layer {layer.name!r} is already in this viewer; a layer draws in one place, so build a new one"
                )
            tag = (tags or layer.tags)[0]
            if f"/{tag}/{layer.name}" in taken:
                raise ArgumentError(
                    f"tag {tag!r} already has a layer named {layer.name!r}; give the new layer another name"
                )
            taken.add(f"/{tag}/{layer.name}")
            if times is not None and times.shape != (layer.nsteps,):
                raise ShapeError(f"times must have shape ({layer.nsteps},) for layer {layer.name!r}, got {times.shape}")
            if valid is not None:
                shapes = [(layer.nsteps,)] + ([(layer.nsteps, layer.elements)] if layer.elements is not None else [])
                if valid.shape not in shapes:
                    raise ShapeError(
                        f"valid must have shape {' or '.join(map(str, shapes))} for layer {layer.name!r}, "
                        f"got {valid.shape}"
                    )

    def _place(self, layer: Layer, times: np.ndarray | None, *, at: int | None = None) -> _Entry:
        """Build ``layer`` (scene and sidebar) under ``/<first tag>/<layer.name>``, list it (at index ``at``) and push
        the current step to it, or remove what it built and re-raise."""
        tag = layer.tags[0]
        if tag not in self._tags:
            self._tags[tag] = _Tag(self.server.scene.add_frame(f"/{tag}", show_axes=False))
        layer.viewer = self
        layer.apply_theme(self.theme)
        root = f"/{tag}/{layer.name}"
        entry = _Entry(layer, root, self.server.scene.add_frame(root, show_axes=False), times)
        self._entries.insert(len(self._entries) if at is None else at, entry)
        try:
            layer.build(self.server.scene, root)
            self.sidebar.build(layer)
            if not layer.visible:
                layer.set_visible(False)  # new handles start shown
            self._push(entry, self.timeline.step)
        except BaseException:
            self._teardown(entry)
            self._entries.remove(entry)
            self._prune_tags()
            raise
        return entry

    def _timeline_gui(self) -> None:
        """Build the Timeline section once the timeline has more than one step."""
        if self.timeline.nsteps > 1 and not self._timeline_built:
            self._timeline_built = True
            self.timeline.build_gui(self.sidebar)

    def _sync_layers_section(self) -> None:
        """One Layers checkbox per tag while there are two or more, none otherwise; the ground and tags that hold only
        sidebar panels (charts, images) are not listed. A checkbox reads its tag in sentence case."""
        tags = [
            tag
            for tag in self.tags
            if tag != "ground" and not all(isinstance(layer, PlotPanel | ImagePanel) for layer in self._tagged(tag))
        ]
        if len(tags) < 2:
            if self._boxes:
                self._boxes.clear()
                self.sidebar.remove("Layers")
            return
        for tag in [tag for tag in self._boxes if tag not in tags]:
            del self._boxes[tag]
            self.sidebar.remove("Layers", _label(tag))
        for tag in tags:
            if tag not in self._boxes:
                self._boxes[tag] = self.sidebar.checkbox(
                    "Layers",
                    _label(tag),
                    value=any(layer.visible for layer in self._tagged(tag)),
                    on_change=lambda visible, tag=tag: self.set_visible(tag, visible),
                )

    def add_plot(
        self, name: str, series: np.ndarray, *, labels: list[str] | None = None, tags: str | Sequence[str] = "plots"
    ) -> PlotPanel:
        """Sidebar chart of ``series (T, K)`` against time with a marker at the current step."""
        return self.add(PlotPanel(name, series, labels=labels, fps=self.timeline.fps), tags=tags)

    def remove(self, target: Layer | str) -> None:
        """Dispose a layer, or every layer carrying the tag ``target``, with its scene nodes and sidebar controls, and
        fit the timeline to the rest.

        The timeline shrinks to the longest remaining layer (one step when none is left), so a clip browser can
        remove one clip's tag before adding the next.
        """
        with self.lock:
            for layer in self._targets(target):
                entry = self._entry(layer)
                self._teardown(entry)
                self._entries.remove(entry)
            self._prune_tags()
            self._fit_timeline()
            self._sync_layers_section()

    def set_offset(self, tag: str, offset: tuple[float, float, float]) -> None:
        """Move the layers whose first tag is ``tag`` by ``offset``, as ``add(..., offset=)`` does, e.g. from a
        side-by-side slider."""
        with self.lock:
            if tag not in self._tags:
                raise UnknownNameError("tag", tag, valid=list(self._tags))
            state = self._tags[tag]
            state.node.position = offset
            state.offset = offset

    def set_fps(self, fps: float) -> None:
        """Play the timeline at ``fps`` steps per second, such as the frame rate of the clip on screen.

        The current step is kept and playback continues. Layers added with ``times`` are mapped onto the new
        step length and the timeline is fitted to them; a `PlotPanel` keeps the time axis it was built with. A sidebar
        callback may call it.
        """
        if not fps > 0.0:
            raise ArgumentError(f"fps must be positive, got {fps}")
        playing = self.timeline.playing
        self.timeline.pause()
        with self.lock:
            self.timeline.fps = float(fps)
            self._fit_timeline()
            self._push_all(self.timeline.step, force=False)
        if playing:
            self.timeline.play()

    def set_visible(self, target: Layer | str, visible: bool) -> None:  # noqa: FBT001 - the switch is the main operand
        """Hide a layer, or every layer carrying the tag ``target`` (skipped by ``show``), or show it again, caught up
        to the current step; the Layers checkboxes follow."""
        with self.lock:
            for layer in self._targets(target):
                entry = self._entry(layer)
                layer.set_visible(visible)
                if visible:
                    self._push(entry, self.timeline.step)
            for tag, box in self._boxes.items():
                shown = any(layer.visible for layer in self._tagged(tag))
                if box.value != shown:
                    box.value = shown

    def set_theme(self, theme: Theme) -> None:
        """Apply ``theme`` to the page and to every layer (``Layer.apply_theme``), as at construction."""
        with self.lock:
            self.theme = theme
            theme.apply(self.server)
            for entry in self._entries:
                entry.layer.apply_theme(theme)

    def show(self, step: int | None = None) -> None:
        """Move the timeline to ``step`` and push it to every visible layer.

        With no argument, re-push the current step to every visible layer even if it already shows it; call this
        after editing the arrays of several layers in place (``Layer.redraw`` does it for one).
        """
        if step is not None and not isinstance(step, int | np.integer):
            raise ArgumentError(f"step must be an int, got {step!r}")
        if step is None or step == self.timeline.step:
            self._push_all(self.timeline.step, force=True)
        else:
            self.timeline.step = step

    def redraw(self, target: Layer | str) -> None:
        """Push the current step again to a visible layer, or to every visible layer carrying the tag ``target``, after
        its arrays changed; ``Layer.redraw`` calls this."""
        with self.lock, self.server.atomic():
            for layer in self._targets(target):
                if layer.visible:
                    self._push(self._entry(layer), self.timeline.step)

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
        size: tuple[int, int] = (1920, 1080),
        fps: float | None = None,
        steps: Sequence[int] | None = None,
        camera: tuple[np.ndarray | Sequence[float], np.ndarray | Sequence[float]] | None = None,
        fov: float = math.radians(40.0),
        gpu: int | None = None,
        bitrate: float | None = None,
    ) -> Path:
        """Render ``steps`` of the live scene in a headless Chromium into a video or a still; return the path written.

        The suffix of ``path`` picks the output: ``.mp4`` (H.264, needs ``ffmpeg``: ``$BETTER_ROBOT_FFMPEG`` or on
        ``PATH``), ``.webm`` (VP9, needs nothing), ``.png`` (one still); without a suffix ``.mp4`` when ``ffmpeg`` is
        found, else ``.webm``. Frame ``k`` shows ``steps[k]`` (default: every step; for a still, the current step).
        ``size`` is the output's ``(width, height)`` in pixels, both even for a video, up to the renderer's limit
        (3840 x 2160 is fine); ``fps`` defaults to the timeline's. ``camera`` is ``(position, look_at)``, each ``(3,)``
        or ``(len(steps), 3)`` to move per step; ``None`` is the view ``aim(steps=steps, fov=fov)`` would choose, and
        the live browsers' cameras stay where they are. ``fov`` is the vertical field of view in radians, in
        ``(0, pi)``. ``gpu`` is the ``nvidia-smi`` index of the GPU to render on, ``None`` the CPU; an unreachable GPU
        raises. ``bitrate`` in bits per second defaults to 8 Mbit/s scaled by the pixel count over full HD. When
        ffmpeg fails, the encoded video is kept beside ``path`` (``.h264``, or ``.webm`` when the browser had no H.264
        encoder) and the error names it. The frames are what a browser shows of this scene, background included.
        Playback pauses while recording and the step shown before comes back; a sidebar callback may call it. Needs a
        Chromium (``$BETTER_ROBOT_CHROME`` or on ``PATH``); see ``better_robot.viewer.record``.
        """
        from .record import _ffmpeg, _record  # noqa: PLC0415 - the browser path is opt-in

        path = Path(path)
        if path.suffix == "":
            path = path.with_suffix(".mp4" if _ffmpeg(required=False) else ".webm")
        suffix = path.suffix.lower()
        if suffix not in (".mp4", ".webm", ".png"):
            raise ArgumentError(f"record writes .mp4, .webm or .png, not {path.name!r}")
        nsteps = self.timeline.nsteps
        steps = list(([self.timeline.step] if suffix == ".png" else range(nsteps)) if steps is None else steps)
        if not all(isinstance(step, int | np.integer) for step in steps):
            raise ArgumentError(f"steps must be integers, got {steps}")
        steps = [int(step) for step in steps]
        if (len(steps) != 1) if suffix == ".png" else not steps:
            raise ArgumentError(
                f"{path.name} takes {'one step' if suffix == '.png' else 'at least one step'}, got {steps}"
            )
        if not all(0 <= step < nsteps for step in steps):
            raise ArgumentError(f"steps must lie in [0, {nsteps}), got {steps}")
        width, height = size = (int(size[0]), int(size[1]))
        if width < 2 or height < 2 or suffix != ".png" and (width % 2 or height % 2):
            raise ArgumentError(f"size must be (width, height) with both even for a video, got {size}")
        if fps is not None and not fps > 0.0:
            raise ArgumentError(f"fps must be positive (None: the timeline's), got {fps}")
        if bitrate is not None and not bitrate > 0.0:
            raise ArgumentError(
                f"bitrate must be positive (None: 8 Mbit/s scaled by the pixels over full HD), got {bitrate}"
            )
        if not 0.0 < fov < math.pi:
            raise ArgumentError(f"fov must lie in (0, pi) radians, got {fov}")
        if camera is not None and len(camera) != 2:
            raise ArgumentError(f"camera must be a pair (position, look_at), got {len(camera)} items")
        # The view aim() would choose, without moving the live browsers' cameras or ending the aim after each add.
        position, look_at = self._framing(steps, _AZIMUTH, _ELEVATION, fov) if camera is None else camera
        position, look_at = np.asarray(position, dtype=np.float64), np.asarray(look_at, dtype=np.float64)
        if position.shape not in ((3,), (len(steps), 3)) or look_at.shape != position.shape:
            raise ArgumentError(
                f"camera must be (position, look_at), each (3,) or ({len(steps)}, 3) per step, got {position.shape} "
                f"and {look_at.shape}"
            )
        fps = self.timeline.fps if fps is None else float(fps)
        bits = 8e6 * width * height / (1920 * 1080) if bitrate is None else float(bitrate)
        return _record(
            self, path, steps=steps, size=size, camera=(position, look_at), fov=fov, gpu=gpu, fps=fps, bitrate=bits
        )

    def aim(
        self,
        *,
        steps: Sequence[int] | None = None,
        azimuth: float = _AZIMUTH,
        elevation: float = _ELEVATION,
        fov: float = _FOV,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Point the connected browsers and every new one (viser's initial camera, also "Reset View") at the visible
        layers, and stop aiming by itself after each ``add``.

        The view holds the boxes of the visible layers' ``Layer.bounds`` over ``steps`` (default: every step; at most
        32 evenly spaced ones are read; steps with nothing visible are skipped), seen from ``azimuth`` and
        ``elevation`` (radians) about the theme's ``up`` axis: the boxes are centred in the view, at the nearest
        distance where every corner of every box lies inside 80 % of a 16:9 view's height and width, in the vertical
        field of view ``fov``; the browsers take ``fov``. Returns ``(position, look_at)``. Under ``+z`` up, azimuth 0
        looks from ``-y`` (the right side of a robot facing ``+x``, the usual convention) and 90 degrees from ``+x``:
        the default 60 degrees is a front three-quarter view.
        """
        if not 0.0 < fov < math.pi:
            raise ArgumentError(f"fov must lie in (0, pi) radians, got {fov}")
        with self.lock:
            self._aimed = True
            position, look_at = self._framing(steps, azimuth, elevation, fov)
            self._point_cameras(position, look_at, fov)
        return position, look_at

    def _framing(
        self, steps: Sequence[int] | None, azimuth: float, elevation: float, fov: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """``(position, look_at)`` of ``aim``, without moving a camera."""
        steps = list(range(self.timeline.nsteps)) if steps is None else list(steps)
        if len(steps) > _AIM_STEPS:
            steps = [steps[int(index)] for index in np.linspace(0, len(steps) - 1, _AIM_STEPS).round()]
        boxes = [box for step in steps for box in self._boxes_at(step)] or [_EMPTY_SCENE]
        # Every corner of every box, not the corners of one box round them all: seen from above, the far and near
        # corners of that box lie far outside what a thin moving robot covers, and the robot came out half the size.
        ends = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], dtype=bool)  # (8, 3)
        corners = np.concatenate(
            [np.where(ends, hi.reshape(-1, 1, 3), lo.reshape(-1, 1, 3)).reshape(-1, 3) for lo, hi in boxes]
        )
        up = np.asarray(self.theme.up_vector)
        side = np.cross(np.array([0.0, 0.0, 1.0]) if abs(up[2]) < 0.9 else np.array([1.0, 0.0, 0.0]), up)
        side /= np.linalg.norm(side)
        forward = np.cross(up, side)
        direction = math.cos(elevation) * (math.cos(azimuth) * side + math.sin(azimuth) * forward)
        direction = direction + math.sin(elevation) * up
        right = np.cross(up, direction)
        right /= np.linalg.norm(right)
        screen_up = np.cross(direction, right)
        # Centred along the view's own axes (measured from the middle of the corners, which keeps one box exactly on
        # its centre); then the nearest distance at which every corner projects inside the filled part of a 16:9 view.
        axes = np.stack([right, screen_up, direction])
        middle = (corners.min(axis=0) + corners.max(axis=0)) / 2.0
        seen = (corners - middle) @ axes.T
        center = middle + (seen.min(axis=0) + seen.max(axis=0)) / 2.0 @ axes
        offsets = corners - center
        tan_y = _AIM_FILL * math.tan(fov / 2.0)
        reach = np.maximum(np.abs(offsets @ screen_up) / tan_y, np.abs(offsets @ right) / (tan_y * _AIM_ASPECT))
        distance = max(float(np.max(offsets @ direction + reach)), 1e-3)
        return center + distance * direction, center

    def _boxes_at(self, step: int) -> list[tuple[np.ndarray, np.ndarray]]:
        """``Layer.bounds`` of the visible layers at timeline ``step`` (each one box or a stack of boxes), offsets of
        their first tags applied."""
        boxes = []
        for entry in self._entries:
            layer = entry.layer
            index = self._index(entry, step)
            if not layer.visible or (layer.valid is not None and layer.valid.ndim == 1 and not layer.valid[index]):
                continue
            bounds = layer.bounds(index)
            if bounds is not None:
                offset = self._tags[layer.tags[0]].offset
                shift = np.zeros(3) if offset is None else np.asarray(offset, dtype=np.float64)
                boxes.append((np.asarray(bounds[0], np.float64) + shift, np.asarray(bounds[1], np.float64) + shift))
        return boxes

    def _point_cameras(self, position: np.ndarray, look_at: np.ndarray, fov: float) -> None:
        for camera in [self.server.initial_camera, *(client.camera for client in self.server.get_clients().values())]:
            camera.fov = fov
            camera.position = position  # before look_at: a client's position drags its look_at along
            camera.look_at = look_at

    def replace(self, old: Layer, new: Layer) -> None:
        """Put ``new`` where ``old`` is: same tags, place in the list and sidebar rows, ``old``'s visibility, the
        current step pushed, the camera untouched.

        ``old``'s scene nodes and controls go; ``new`` is built under ``/<first tag>/<new.name>`` and the timeline is
        fitted to the layers now present. ``new`` arrives without ``times`` or ``valid`` (``add``'s per-layer
        options); a new name must be free under the first tag, and ``new`` must not be in the viewer yet. Use it to swap
        the clip or body a layer draws. When ``new`` raises while it builds or shows its first step, ``old`` is drawn
        again and the error propagates.
        """
        with self.lock:
            entry = self._entry(old)
            if self._holds(new):
                raise ArgumentError(f"layer {new.name!r} is already in this viewer; replace takes a new layer object")
            root = f"/{old.tags[0]}/{new.name}"
            if new.name != old.name and any(other.root == root for other in self._entries):
                raise ArgumentError(
                    f"tag {old.tags[0]!r} already has a layer named {new.name!r}; give the new layer another name"
                )
            index = self._entries.index(entry)
            # viser keys scene nodes by path, so ``new`` cannot be built beside ``old`` under the same name: ``old``'s
            # scene goes first; its sidebar controls go after ``new`` asked for its own, so shared rows stay in place.
            self._teardown(entry, controls=False)
            self._entries.remove(entry)
            new.tags = old.tags
            new.valid = None
            new.set_visible(old.visible)
            try:
                self._place(new, None, at=index)
            except BaseException:
                self.sidebar.remove(old)
                self._place(old, entry.times, at=index)
                raise
            self.sidebar.remove(old)
            old.viewer = None
            self._fit_timeline()
            self._timeline_gui()
            self._sync_layers_section()

    def serve(self) -> None:
        """Play the timeline (a no-op for a single step) and serve the page until interrupted, then pause."""
        self.timeline.play()
        try:
            self.server.sleep_forever()
        finally:
            # A playback thread still running at interpreter exit dies inside torch and aborts the process.
            self.timeline.pause()

    def _targets(self, target: Layer | str) -> list[Layer]:
        """The layer ``target``, or the layers carrying the tag ``target``; an unknown one raises."""
        if not isinstance(target, str):
            self._entry(target)
            return [target]
        layers = self._tagged(target)
        if not layers:
            raise UnknownNameError("tag", target, valid=self.tags)
        return layers

    def _tagged(self, tag: str) -> list[Layer]:
        return [entry.layer for entry in self._entries if tag in entry.layer.tags]

    def _prune_tags(self) -> None:
        """Remove the frame of every tag that is no layer's first tag any more."""
        firsts = {entry.layer.tags[0] for entry in self._entries}
        for tag in [tag for tag in self._tags if tag not in firsts]:
            self._tags.pop(tag).node.remove()

    def _teardown(self, entry: _Entry, *, controls: bool = True) -> None:
        """Remove ``entry``'s scene nodes and frame and, with ``controls``, its sidebar controls."""
        entry.layer.dispose()
        if controls:
            self.sidebar.remove(entry.layer)
            entry.layer.viewer = None
        entry.node.remove()

    def _entry(self, layer: Layer) -> _Entry:
        for entry in self._entries:
            if entry.layer is layer:
                return entry
        raise ArgumentError(f"layer {layer.name!r} is not in this viewer; add it first")

    def _holds(self, layer: Layer) -> bool:
        """Whether ``layer`` (this object, not its name) is in the viewer."""
        return any(entry.layer is layer for entry in self._entries)

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
        spans = (self._span(entry) for entry in self._entries)
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
        with self.lock, self.server.atomic():
            for entry in self._entries:
                if entry.layer.visible:
                    self._push(entry, step, force=force)
