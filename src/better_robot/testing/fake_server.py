"""``FakeServer``: the shipped stand-in for ``viser.ViserServer`` (decision records 0022 and 0035).

No sockets and no viser import: every ``add_*`` returns a ``FakeHandle`` that records its constructor arguments and
later attribute writes, so ``Viewer(server=FakeServer())`` builds, pushes and tears down a scene in a test or an
example's ``--check`` run, and the test reads what the browser would have received.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any, final


@final
class FakeHandle:
    """Records constructor kwargs and later attribute writes; supports GUI callbacks and folders."""

    def __init__(self, scene: FakeScene | None, kind: str, name: str, kwargs: dict[str, Any]) -> None:
        # Written through ``__dict__`` so that construction is not logged as attribute writes.
        self.__dict__.update(
            scene=scene,
            kind=kind,
            name=name,
            props=dict(kwargs),
            writes=[],
            callbacks=[],
            removed=False,
            visible=kwargs.get("visible", True),
            value=kwargs.get("initial_value"),
            order=kwargs.get("order"),
            label=name,
        )

    def __setattr__(self, key: str, value: Any) -> None:
        self.__dict__[key] = value
        self.writes.append(key)
        if self.scene is not None:
            self.scene.log.append((self.name, key))

    def remove(self) -> None:
        self.__dict__["removed"] = True
        if self.scene is not None:
            self.scene.log.append((self.name, "remove"))

    def on_update(self, callback: Any) -> None:
        self.callbacks.append(callback)

    on_click = on_update
    on_drag_end = on_update

    def fire(self) -> None:
        for callback in self.callbacks:
            callback(None)

    def __enter__(self) -> FakeHandle:
        return self

    def __exit__(self, *exc: Any) -> None:
        pass


@final
class FakeScene:
    """Every ``add_*`` returns a ``FakeHandle``; ``log`` lists (node, event) in order."""

    def __init__(self) -> None:
        self.nodes: dict[str, FakeHandle] = {}
        self.log: list[tuple[str, str]] = []
        self.pointer: tuple[str, Any] | None = None

    def on_pointer_event(self, event_type: str) -> Any:
        """viser's single pointer slot: registering while one is held is a bug in the caller."""
        assert self.pointer is None, "pointer slot already taken"

        def register(callback: Any) -> Any:
            self.pointer = (event_type, callback)
            return callback

        return register

    def remove_pointer_callback(self) -> None:
        self.pointer = None

    def __getattr__(self, method: str) -> Any:
        if not method.startswith(("add_", "set_", "configure_")):
            raise AttributeError(method)

        def call(*args: Any, **kwargs: Any) -> FakeHandle | None:
            if not method.startswith("add_"):
                return None
            name = args[0]
            handle = FakeHandle(self, method[4:], name, {"args": args[1:], **kwargs})
            self.nodes[name] = handle
            self.log.append((name, "add"))
            return handle

        return call


@final
class FakeGui:
    """Every ``add_*`` returns a ``FakeHandle`` kept in ``controls``; ``control(label)`` finds one."""

    def __init__(self) -> None:
        self.controls: list[FakeHandle] = []

    def __getattr__(self, method: str) -> Any:
        if not method.startswith(("add_", "configure_")):
            raise AttributeError(method)

        def call(*args: Any, **kwargs: Any) -> FakeHandle | None:
            if not method.startswith("add_"):
                return None
            label = args[0] if args and isinstance(args[0], str) else method[4:]
            handle = FakeHandle(None, method[4:], label, {"args": args, **kwargs})
            self.controls.append(handle)
            return handle

        return call

    def control(self, label: str) -> FakeHandle:
        return next(control for control in self.controls if control.name == label)


@final
class FakeServer:
    """A ``viser.ViserServer`` without sockets: ``scene``, ``gui``, ``atomic()``, ``port`` and the camera of new
    clients, all recorded."""

    def __init__(self, port: int = 8080) -> None:
        self.scene = FakeScene()
        self.gui = FakeGui()
        self.atomic_blocks = 0
        self.port = port
        self.initial_camera = FakeHandle(None, "camera", "initial_camera", {})
        self.clients: dict[int, Any] = {}  # connected browsers, each with a ``camera`` handle

    def get_port(self) -> int:
        return self.port

    def get_clients(self) -> dict[int, Any]:
        return dict(self.clients)

    @contextlib.contextmanager
    def atomic(self) -> Iterator[None]:
        self.atomic_blocks += 1
        yield
