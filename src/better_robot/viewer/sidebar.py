"""``Sidebar``: the viewer's controls in ordered sections, shared by ``(section, label)``, run under its lock."""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING, final, override

from ..exceptions import ArgumentError, BetterRobotError, UnknownNameError

if TYPE_CHECKING:
    from .layer import Layer

#: Where the viewer's own sections sit; a section an application creates goes after Images, in creation order.
_ORDER = {"Timeline": 0.0, "Robot": 10.0, "Layers": 20.0, "Plots": 30.0, "Images": 35.0, "Info": 90.0}
_APPLICATION_ORDER = 40.0


@final
@dataclass
class _Control:
    handle: Any
    value: Any  # the value the callbacks last applied; a failed change returns the control to it
    callbacks: list[tuple[int, Callable[[Any], None]]] = field(default_factory=list)  # (owner, callback)
    quiet: bool = False  # True while the sidebar itself sets the value: the callbacks stay out
    offers: dict[int, set[str]] = field(default_factory=dict)  # a dropdown's options, per owner that asked for them


@final
@dataclass
class _Section:
    folder: Any
    owners: set[int] = field(default_factory=set)  # the owners that asked for it


@final
class Sidebar:
    """The viewer's controls: sections (viser folders) in a fixed order, holding controls shared by
    ``(section, label)``.

    A section appears with its first control: Timeline, Robot, Layers, Plots, Images, then the sections an
    application creates in creation order, Info last; ``order`` puts sections of its choice first. Asking again for a ``(section, label)`` that exists returns the
    same control and adds the callback to it, so one "Style" dropdown drives every robot; a control and its section
    go when the last layer that asked for them is removed. Every callback runs under the viewer's lock. A ``BetterRobotError``
    raised in one becomes a red notification in the browser and the control returns to its previous value; any other
    exception propagates. ``gui`` is viser's GUI API, for what the sidebar does not make (a chart, an image): add it
    inside ``with sidebar.section(name):``.
    """

    def __init__(self, server: Any, lock: threading.RLock) -> None:
        self.gui = server.gui
        self._server = server
        self._lock = lock
        self._sections: dict[str, _Section] = {}
        self._controls: dict[tuple[str, str], _Control] = {}
        self._owner = 0  # who asks for controls: id() of the layer whose build_gui runs, 0 for the application
        self._created = 0  # application sections created so far: the order of the next one
        self._first: dict[str, float] = {}  # the places ``order`` gave, before every other section

    @override
    def __repr__(self) -> str:
        return f"Sidebar(sections={list(self._sections)}, controls={len(self._controls)})"

    def section(self, name: str) -> Any:
        """The folder of section ``name``, made at its place in the order on first use; a context manager for
        controls added through ``gui``."""
        section = self._sections.get(name)
        if section is None:
            order = self._first.get(name, _ORDER.get(name))
            if order is None:
                order = _APPLICATION_ORDER + self._created
                self._created += 1
            section = self._sections[name] = _Section(self.gui.add_folder(name, order=order))
        section.owners.add(self._owner)
        return section.folder

    def order(self, names: Sequence[str]) -> None:
        """Put the sections ``names`` first, in this order, now and whenever one of them is made again; the other
        sections keep their order after them."""
        with self._lock:
            self._first = {name: float(place - len(names)) for place, name in enumerate(names)}
            for name, place in self._first.items():
                if name in self._sections:
                    self._sections[name].folder.order = place

    def checkbox(self, section: str, label: str, *, value: bool, on_change: Callable[[bool], None]) -> Any:
        """A checkbox; ``on_change(value)`` runs when it is switched."""
        return self._control(section, label, on_change, lambda: self.gui.add_checkbox(label, initial_value=value))

    def dropdown(
        self, section: str, label: str, *, options: Sequence[str], value: str, on_change: Callable[[str], None]
    ) -> Any:
        """A dropdown of ``options``; ``on_change(option)`` runs on a choice. A layer joining an existing dropdown
        adds the options it lacks; when a layer goes, so do the options no other offers, and if the one shown is
        among them, the dropdown shows the first one left without running a callback: removing or replacing a layer
        never changes the others."""
        control = self._controls.get((section, label))
        if control is not None:
            missing = tuple(option for option in options if option not in control.handle.options)
            if missing:
                control.handle.options = (*control.handle.options, *missing)
        handle = self._control(
            section, label, on_change, lambda: self.gui.add_dropdown(label, options=tuple(options), initial_value=value)
        )
        self._controls[(section, label)].offers.setdefault(self._owner, set()).update(options)
        return handle

    def slider(
        self,
        section: str,
        label: str,
        *,
        low: float,
        high: float,
        step: float,
        value: float,
        on_change: Callable[[float], None],
    ) -> Any:
        """A slider from ``low`` to ``high`` in increments of ``step``; ``on_change(value)`` runs as it moves."""
        return self._control(
            section,
            label,
            on_change,
            lambda: self.gui.add_slider(label, min=low, max=high, step=step, initial_value=value),
        )

    def button(self, section: str, label: str, *, on_click: Callable[[], None]) -> Any:
        """A button; ``on_click()`` runs on each click."""
        return self._control(section, label, lambda _: on_click(), lambda: self.gui.add_button(label), click=True)

    def set_value(self, section: str, label: str, *, value: Any) -> None:
        """Show ``value`` on the control ``(section, label)`` without running its callbacks: what a layer calls when
        code changed the state its control shows. Nothing happens when there is no such control."""
        with self._lock:
            control = self._controls.get((section, label))
            if control is None or control.handle.value == value:
                return
            control.quiet = True  # viser runs a control's callbacks for a value set from code too
            try:
                control.handle.value = control.value = value
            finally:
                control.quiet = False

    def build(self, layer: Layer) -> None:
        """Run ``layer.build_gui(self)``; the controls and sections it asks for count as the layer's, so that
        ``remove(layer)`` takes back exactly those."""
        self._owner = id(layer)
        try:
            layer.build_gui(self)
        finally:
            self._owner = 0

    def remove(self, target: str | Layer, label: str | None = None) -> None:
        """Remove the control ``(target, label)``, the whole section ``target`` when ``label`` is ``None``, or, when
        ``target`` is a layer, the callbacks it added: a control or section nobody else uses goes with them."""
        with self._lock:
            if not isinstance(target, str):
                self._release(id(target))
                return
            if target not in self._sections:
                raise UnknownNameError("section", target, valid=list(self._sections))
            keys = [key for key in self._controls if key[0] == target and (label is None or key[1] == label)]
            if label is not None and not keys:
                raise UnknownNameError("control", label, valid=[key[1] for key in self._controls if key[0] == target])
            for key in keys:
                self._controls.pop(key).handle.remove()
            if label is None:
                self._sections.pop(target).folder.remove()

    def _release(self, owner: int) -> None:
        for key, control in list(self._controls.items()):
            control.callbacks = [(held, callback) for held, callback in control.callbacks if held != owner]
            if not control.callbacks:
                self._controls.pop(key).handle.remove()
            elif control.offers.pop(owner, None) is not None:
                self._withdraw(control)
        for name, section in list(self._sections.items()):
            section.owners.discard(owner)
            if not section.owners and not any(key[0] == name for key in self._controls):
                self._sections.pop(name).folder.remove()

    def _withdraw(self, control: _Control) -> None:
        """Keep the options of a dropdown that its remaining owners offer; a choice that went moves quietly to the
        first one left (viser itself moves it on a server), so the layers that stay keep their state."""
        handle = control.handle
        kept = tuple(option for option in handle.options if any(option in offer for offer in control.offers.values()))
        if kept == tuple(handle.options):
            return
        control.quiet = True
        try:
            handle.options = kept
            if handle.value not in kept:
                handle.value = kept[0]
            control.value = handle.value
        finally:
            control.quiet = False

    def _control(
        self, section: str, label: str, callback: Callable[[Any], None], make: Callable[[], Any], *, click: bool = False
    ) -> Any:
        if not label:
            raise ArgumentError("a sidebar control needs a label")
        control = self._controls.get((section, label))
        if control is None:
            with self.section(section):
                handle = make()
            control = self._controls[(section, label)] = _Control(handle, None if click else handle.value)
            register = handle.on_click if click else handle.on_update
            register(lambda event: self._run(control, event))
        else:
            self.section(section)
        control.callbacks.append((self._owner, callback))
        return control.handle

    def _run(self, control: _Control, event: Any) -> None:
        """One change of ``control``: every callback under the lock; on a ``BetterRobotError`` the ones that ran see
        the previous value again, the control shows it and the browser shows the error."""
        with self._lock:
            if control.quiet:
                return
            value = control.handle.value
            done: list[Callable[[Any], None]] = []
            try:
                for _, callback in list(control.callbacks):
                    callback(value)
                    done.append(callback)
            except BetterRobotError as error:
                if control.value is not None:
                    control.quiet = True
                    try:
                        for callback in done:
                            callback(control.value)
                        control.handle.value = control.value
                    finally:
                        control.quiet = False
                client = None if event is None else event.client
                clients = [client] if client is not None else list(self._server.get_clients().values())
                for each in clients:
                    each.add_notification(control.handle.label, str(error), color="red", auto_close_seconds=8.0)
                return
            control.value = value
