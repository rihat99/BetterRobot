"""``FakeServer``: the shipped stand-in for ``viser.ViserServer`` (decision records 0022 and 0035).

No sockets and no viser import. Each viser call the viewer makes is declared here by name with viser's parameter
names and defaults (the parameters BetterRobot passes; the full test tier binds every method against viser's), and
returns a ``FakeHandle`` that records the call's arguments and later attribute writes. So
``Viewer(server=FakeServer())`` builds, pushes and tears down a scene in a test or an example's ``--check`` run, and
the test reads what the browser would have received. A call the viewer does not make raises ``AttributeError``.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator
from typing import Any, Literal, final

_IDENTITY = (1.0, 0.0, 0.0, 0.0)
_ORIGIN = (0.0, 0.0, 0.0)


@final
class FakeHandle:
    """One viser handle: ``kind`` names the viser class it stands for and ``props`` holds the call's arguments.

    Attribute writes are kept (``writes``, and ``(name, attribute)`` in the scene's ``log``) and so are the names of
    the methods called on it (``calls``). A write after ``remove()`` raises ``AttributeError``, as viser refuses it
    (with a ``RuntimeError``) and a frozen dataclass refuses any write.
    ``fire(event)`` plays the browser: it calls every callback that ``on_update`` or ``on_click`` registered.
    """

    def __init__(self, scene: FakeScene | None, kind: str, name: str, props: dict[str, Any], **state: Any) -> None:
        # Written through ``__dict__`` so that construction is not logged as attribute writes.
        self.__dict__.update(
            scene=scene, kind=kind, name=name, props=props, writes=[], calls=[], callbacks=[], removed=False
        )
        self.__dict__.update(state)

    def __setattr__(self, key: str, value: Any) -> None:
        if self.removed:
            raise AttributeError(f"cannot assign {key!r}: this {self.kind} was removed")
        self.__dict__[key] = value
        self.writes.append(key)
        if self.scene is not None:
            self.scene.log.append((self.name, key))

    def __repr__(self) -> str:
        """One line: kind, name and whether it was removed."""
        return f"FakeHandle({self.kind}, {self.name!r}{', removed' if self.removed else ''})"

    def remove(self) -> None:
        """Remove the node or control; later writes raise."""
        self.calls.append("remove")
        self.__dict__["removed"] = True
        if self.scene is not None:
            self.scene.log.append((self.name, "remove"))

    def on_update(self, func: Callable[[Any], None]) -> Callable[[Any], None]:
        """viser's change callback of a control or a transform gizmo."""
        self.calls.append("on_update")
        self.callbacks.append(func)
        return func

    def on_click(self, func: Callable[[Any], None]) -> Callable[[Any], None]:
        """viser's button callback."""
        self.calls.append("on_click")
        self.callbacks.append(func)
        return func

    def fire(self, event: Any = None) -> None:
        """Call every registered callback with ``event``, as the browser does on a click, a drag or an edit."""
        for callback in self.callbacks:
            callback(event)

    def __enter__(self) -> FakeHandle:
        self.calls.append("__enter__")
        return self

    def __exit__(self, *exc: object) -> None:
        pass


@final
class FakeScene:
    """viser's ``server.scene``: every ``add_*`` returns a ``FakeHandle`` kept in ``nodes``; ``log`` lists
    ``(node, event)`` in order; ``pointers`` lists ``(event type, callback)`` registered by ``on_click`` /
    ``on_rect_select``."""

    def __init__(self) -> None:
        self.nodes: dict[str, FakeHandle] = {}
        self.log: list[tuple[str, str]] = []
        self.pointers: list[tuple[str, Callable[[Any], None]]] = []
        self.up_direction: Any = None

    def _node(self, kind: str, arguments: dict[str, Any]) -> FakeHandle:
        """A scene node from a method's ``locals()``: position, orientation and visibility readable as on viser."""
        props = {key: value for key, value in arguments.items() if key != "self"}
        state = {key: props[key] for key in ("position", "wxyz", "visible")}
        handle = FakeHandle(self, kind, props.pop("name"), props, **state)
        self.nodes[handle.name] = handle
        self.log.append((handle.name, "add"))
        return handle

    def add_frame(
        self, name: str, *, show_axes: bool = True, wxyz: Any = _IDENTITY, position: Any = _ORIGIN, visible: bool = True
    ) -> FakeHandle:
        return self._node("FrameHandle", locals())

    def add_batched_axes(
        self,
        name: str,
        batched_wxyzs: Any,
        batched_positions: Any,
        *,
        axes_length: float = 0.5,
        axes_radius: float = 0.025,
        wxyz: Any = _IDENTITY,
        position: Any = _ORIGIN,
        visible: bool = True,
    ) -> FakeHandle:
        return self._node("BatchedAxesHandle", locals())

    def add_batched_meshes_simple(
        self,
        name: str,
        vertices: Any,
        faces: Any,
        batched_wxyzs: Any,
        batched_positions: Any,
        *,
        batched_scales: Any = None,
        batched_colors: Any = (90, 200, 255),
        lod: Any = "auto",
        opacity: float | None = None,
        cast_shadow: bool = True,
        wxyz: Any = _IDENTITY,
        position: Any = _ORIGIN,
        visible: bool = True,
    ) -> FakeHandle:
        return self._node("BatchedMeshHandle", locals())

    def add_camera_frustum(
        self,
        name: str,
        fov: float,
        aspect: float,
        *,
        scale: float = 0.3,
        color: Any = (20, 20, 20),
        image: Any = None,
        wxyz: Any = _IDENTITY,
        position: Any = _ORIGIN,
        visible: bool = True,
    ) -> FakeHandle:
        return self._node("CameraFrustumHandle", locals())

    def add_grid(
        self,
        name: str,
        *,
        width: float = 10.0,
        height: float = 10.0,
        plane: str = "xy",
        cell_color: Any = (200, 200, 200),
        cell_size: float = 0.5,
        section_color: Any = (140, 140, 140),
        section_size: float = 1.0,
        plane_color: Any = (255, 255, 255),
        plane_opacity: float = 0.0,
        wxyz: Any = _IDENTITY,
        position: Any = _ORIGIN,
        visible: bool = True,
    ) -> FakeHandle:
        return self._node("GridHandle", locals())

    def add_line_segments(
        self,
        name: str,
        points: Any,
        colors: Any,
        *,
        line_width: float = 1,
        wxyz: Any = _IDENTITY,
        position: Any = _ORIGIN,
        visible: bool = True,
    ) -> FakeHandle:
        return self._node("LineSegmentsHandle", locals())

    def add_mesh_simple(
        self,
        name: str,
        vertices: Any,
        faces: Any,
        *,
        color: Any = (90, 200, 255),
        opacity: float | None = None,
        flat_shading: bool = False,
        wxyz: Any = _IDENTITY,
        position: Any = _ORIGIN,
        visible: bool = True,
    ) -> FakeHandle:
        return self._node("MeshHandle", locals())

    def add_mesh_skinned(
        self,
        name: str,
        vertices: Any,
        faces: Any,
        bone_wxyzs: Any,
        bone_positions: Any,
        skin_weights: Any,
        *,
        color: Any = (90, 200, 255),
        opacity: float | None = None,
        wxyz: Any = _IDENTITY,
        position: Any = _ORIGIN,
        visible: bool = True,
    ) -> FakeHandle:
        handle = self._node("MeshSkinnedHandle", locals())
        bones = tuple(
            FakeHandle(self, "MeshSkinnedBoneHandle", f"{name}/bone{index}", {}, wxyz=bone_wxyz, position=bone_position)
            for index, (bone_wxyz, bone_position) in enumerate(zip(bone_wxyzs, bone_positions, strict=True))
        )
        handle.__dict__["bones"] = bones  # read-only on viser
        return handle

    def add_mesh_trimesh(
        self, name: str, mesh: Any, *, wxyz: Any = _IDENTITY, position: Any = _ORIGIN, visible: bool = True
    ) -> FakeHandle:
        return self._node("MeshHandle", locals())

    def add_point_cloud(
        self,
        name: str,
        points: Any,
        colors: Any,
        *,
        point_size: float = 0.1,
        point_shape: str = "square",
        precision: str = "float16",
        wxyz: Any = _IDENTITY,
        position: Any = _ORIGIN,
        visible: bool = True,
    ) -> FakeHandle:
        return self._node("PointCloudHandle", locals())

    def add_transform_controls(
        self,
        name: str,
        *,
        scale: float = 1.0,
        disable_axes: bool = False,
        disable_sliders: bool = False,
        wxyz: Any = _IDENTITY,
        position: Any = _ORIGIN,
        visible: bool = True,
    ) -> FakeHandle:
        return self._node("TransformControlsHandle", locals())

    def configure_default_lights(self, enabled: bool = True, cast_shadow: bool = True) -> None:  # noqa: FBT001, FBT002 - viser's signature
        """Accepted; the stand-in draws nothing."""

    def set_up_direction(self, direction: Any) -> None:
        self.up_direction = direction

    def on_click(self, modifier: Any = None) -> Callable[[Callable[[Any], None]], Callable[[Any], None]]:
        """viser's scene click registration: a decorator that adds ``("click", callback)`` to ``pointers``."""
        return self._register("click")

    def on_rect_select(self, modifier: Any = None) -> Callable[[Callable[[Any], None]], Callable[[Any], None]]:
        """viser's box-select registration: a decorator that adds ``("rect-select", callback)`` to ``pointers``."""
        return self._register("rect-select")

    def remove_click_callback(self, callback: Literal["all"] | Callable[[Any], None] = "all") -> None:
        """Remove that click callback, or every one; nothing happens when none matches, as on viser."""
        self._unregister("click", callback)

    def remove_rect_select_callback(self, callback: Literal["all"] | Callable[[Any], None] = "all") -> None:
        """Remove that box-select callback, or every one; nothing happens when none matches, as on viser."""
        self._unregister("rect-select", callback)

    def _register(self, event_type: str) -> Callable[[Callable[[Any], None]], Callable[[Any], None]]:
        def register(callback: Callable[[Any], None]) -> Callable[[Any], None]:
            self.pointers.append((event_type, callback))
            return callback

        return register

    def _unregister(self, event_type: str, callback: Literal["all"] | Callable[[Any], None]) -> None:
        self.pointers = [
            (kind, held)
            for kind, held in self.pointers
            if not (kind == event_type and (isinstance(callback, str) or held == callback))
        ]


@final
class FakeGui:
    """viser's ``server.gui``: every ``add_*`` returns a ``FakeHandle`` kept in ``controls``; ``control(label)``
    finds one."""

    def __init__(self) -> None:
        self.controls: list[FakeHandle] = []
        self.theme: dict[str, Any] = {}

    def control(self, label: str) -> FakeHandle:
        """The first control named ``label``."""
        return next(control for control in self.controls if control.name == label)

    def _control(self, kind: str, label: str, arguments: dict[str, Any]) -> FakeHandle:
        """A control from a method's ``locals()``: value, order, label and visibility readable as on viser."""
        props = {key: value for key, value in arguments.items() if key != "self"}
        state = {"value": props.get("initial_value"), "order": props["order"], "label": label}
        handle = FakeHandle(None, kind, label, props, visible=props["visible"], **state)
        self.controls.append(handle)
        return handle

    def add_folder(self, label: str, *, order: float | None = None, visible: bool = True) -> FakeHandle:
        return self._control("GuiFolderHandle", label, locals())

    def add_button(self, label: str, *, order: float | None = None, visible: bool = True) -> FakeHandle:
        return self._control("GuiButtonHandle", label, locals())

    def add_checkbox(
        self,
        label: str,
        initial_value: bool,  # noqa: FBT001 - viser's signature
        *,
        order: float | None = None,
        visible: bool = True,
    ) -> FakeHandle:
        return self._control("GuiCheckboxHandle", label, locals())

    def add_slider(
        self,
        label: str,
        min: float,  # noqa: A002 - viser's parameter name
        max: float,  # noqa: A002 - viser's parameter name
        step: float,
        initial_value: float,
        *,
        order: float | None = None,
        visible: bool = True,
    ) -> FakeHandle:
        return self._control("GuiSliderHandle", label, locals())

    def add_image(
        self,
        image: Any,
        *,
        label: str | None = None,
        format: str = "auto",  # noqa: A002 - viser's parameter name
        order: float | None = None,
        visible: bool = True,
    ) -> FakeHandle:
        return self._control("GuiImageHandle", label or "image", locals())

    def add_uplot(
        self,
        data: Any,
        series: Any,
        *,
        legend: Any = None,
        aspect: float = 1.0,
        order: float | None = None,
        visible: bool = True,
    ) -> FakeHandle:
        return self._control("GuiUplotHandle", "uplot", locals())

    def configure_theme(
        self,
        *,
        control_layout: str = "floating",
        control_width: str = "medium",
        dark_mode: bool = False,
        show_logo: bool = True,
        show_share_button: bool = True,
        brand_color: Any = None,
    ) -> None:
        self.theme = {key: value for key, value in locals().items() if key != "self"}


@final
class _FakeSerializer:
    """viser's scene serializer: ``insert_sleep`` per recorded frame, ``serialize`` the bytes of a ``.viser`` file."""

    def __init__(self) -> None:
        self.sleeps: list[float] = []

    def insert_sleep(self, duration: float) -> None:
        self.sleeps.append(duration)

    def serialize(self) -> bytes:
        return f"fake viser scene, {len(self.sleeps)} frames".encode()


@final
class FakeServer:
    """A ``viser.ViserServer`` without sockets: ``scene``, ``gui``, ``atomic()``, ``port``, the camera of new clients,
    ``get_scene_serializer``, and ``sleep_forever`` / ``stop``, which return at once."""

    def __init__(self, port: int = 8080) -> None:
        self.scene = FakeScene()
        self.gui = FakeGui()
        self.atomic_blocks = 0
        self.port = port
        self.initial_camera = FakeHandle(None, "InitialCameraConfig", "initial_camera", {})
        self.clients: dict[int, Any] = {}  # connected browsers, each with a ``camera`` handle
        self.stopped = False

    def __repr__(self) -> str:
        """One line: port and how many scene nodes and controls it holds."""
        return f"FakeServer(port={self.port}, nodes={len(self.scene.nodes)}, controls={len(self.gui.controls)})"

    def get_port(self) -> int:
        return self.port

    def get_clients(self) -> dict[int, Any]:
        return dict(self.clients)

    def get_scene_serializer(self) -> _FakeSerializer:
        return _FakeSerializer()

    @contextlib.contextmanager
    def atomic(self) -> Iterator[None]:
        self.atomic_blocks += 1
        yield

    def sleep_forever(self) -> None:
        """Return at once: there is nothing to serve."""

    def stop(self) -> None:
        self.stopped = True
