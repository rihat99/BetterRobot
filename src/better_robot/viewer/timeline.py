"""``Timeline``: the current step, playback by wall clock, and its GUI controls."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any, TYPE_CHECKING, final, override

from ..exceptions import ArgumentError

if TYPE_CHECKING:
    from .sidebar import Sidebar

#: The playback speeds the sidebar offers, as factors of ``fps``.
_SPEEDS = {"0.25\u00d7": 0.25, "0.5\u00d7": 0.5, "1\u00d7": 1.0, "2\u00d7": 2.0}


@final
class Timeline:
    """Step counter for ``nsteps`` steps at ``fps``.

    Setting ``step`` clamps to ``[0, nsteps - 1]`` and notifies the ``on_step`` callbacks
    (the viewer pushes the step from there). ``play`` starts a thread that advances by wall
    clock, so slow pushes drop steps instead of slowing playback; ``speed`` and ``loop`` take
    effect while it plays, and a step set meanwhile (the slider) becomes where playback goes on.
    The thread advances under ``lock`` (the viewer passes its own), which it never waits for
    without looking whether playback stopped, so ``pause`` is safe while holding it. With ``play_button`` set before
    the sidebar is built, Play is a button that reads Play or Pause instead of a checkbox.
    """

    def __init__(
        self, nsteps: int, *, fps: float = 30.0, loop: bool = True, lock: threading.RLock | None = None
    ) -> None:
        if nsteps < 1 or not fps > 0.0:
            raise ArgumentError(f"Timeline needs nsteps >= 1 and fps > 0, got nsteps={nsteps}, fps={fps}")
        self.nsteps = nsteps
        self.fps = fps
        self.loop = loop
        self.speed = 1.0
        self.play_button = False
        self._step = 0
        self._callbacks: list[Callable[[int], None]] = []
        self._playing = threading.Event()
        self._lock = threading.RLock() if lock is None else lock
        self._thread: threading.Thread | None = None
        self._generation = 0  # which play() a playback thread belongs to: an older one stops by itself
        self._gui_slider: Any = None
        self._gui_play: Any = None

    @override
    def __repr__(self) -> str:
        return f"Timeline(nsteps={self.nsteps}, step={self._step}, fps={self.fps}, playing={self.playing})"

    @property
    def step(self) -> int:
        """Current step index; assigning clamps to ``[0, nsteps - 1]`` and runs the ``on_step`` callbacks."""
        return self._step

    @step.setter
    def step(self, value: int) -> None:
        value = min(max(int(value), 0), self.nsteps - 1)
        self._step = value
        for callback in self._callbacks:
            callback(value)
        if self._gui_slider is not None and self._gui_slider.value != value:
            self._gui_slider.value = value

    @property
    def playing(self) -> bool:
        """Whether the playback thread is advancing steps."""
        return self._playing.is_set()

    @property
    def time(self) -> float:
        """Time of the current step in seconds."""
        return self._step / self.fps

    def on_step(self, callback: Callable[[int], None]) -> None:
        """Call ``callback(step)`` after every step change."""
        self._callbacks.append(callback)

    def extend(self, nsteps: int) -> None:
        """Grow to at least ``nsteps`` steps (layers added later may be longer)."""
        if nsteps > self.nsteps:
            self.resize(nsteps)

    def resize(self, nsteps: int) -> None:
        """Set the number of steps to ``nsteps`` (at least 1); a current step past the end moves to the new last step."""
        if nsteps < 1:
            raise ArgumentError(f"nsteps must be at least 1, got {nsteps}")
        self.nsteps = nsteps
        if self._gui_slider is not None:
            self._gui_slider.max = nsteps - 1
        if self._step > nsteps - 1:
            self.step = nsteps - 1

    def play(self) -> None:
        """Start playback in a background thread; a no-op when already playing or ``nsteps == 1``.

        Without ``loop``, playback that sits on the last step restarts from step 0.
        """
        with self._lock:  # two clients clicking Play together must not start two threads
            if self.playing or self.nsteps == 1:
                return
            self._playing.set()
            self._generation += 1
            thread = self._thread = threading.Thread(
                target=self._run, args=(self._generation,), name="better_robot.viewer.timeline", daemon=True
            )
        if not self.loop and self._step >= self.nsteps - 1:
            self.step = 0
        thread.start()
        self._sync_play_button()

    def pause(self) -> None:
        """Stop playback and wait for the playback thread to exit; the current step is kept.

        Safe while holding ``lock``, as a sidebar callback does: a thread waiting for it sees the stop and ends.
        """
        self._halt()
        thread, self._thread = self._thread, None
        if thread is not None and thread is not threading.current_thread():
            thread.join()

    def _halt(self) -> None:
        """Stop playback without waiting: the playback thread sees it at its next tick and ends."""
        self._playing.clear()
        self._sync_play_button()

    def _run(self, generation: int) -> None:
        position, shown, last = float(self._step), self._step, time.perf_counter()
        while self.playing and generation == self._generation:
            # Never wait for the lock without looking up: pause() may hold it while it waits for this thread to end.
            if not self._lock.acquire(timeout=0.05):
                continue
            try:
                if not (self.playing and generation == self._generation):  # stopped while this thread waited
                    return
                if self._step != shown:  # moved by hand while playing: go on from there
                    position = float(self._step)
                now = time.perf_counter()
                position += (now - last) * self.speed * self.fps
                last = now
                if position >= self.nsteps:
                    if not self.loop:
                        self.step = self.nsteps - 1
                        self._halt()
                        return
                    position %= self.nsteps
                shown = int(position)
                if shown != self._step:
                    self.step = shown
            finally:
                self._lock.release()
            time.sleep(min(0.5 / (self.fps * self.speed), 0.05))

    def build_gui(self, sidebar: Sidebar) -> None:
        """Play, Step, Speed and Loop in the sidebar's Timeline section; like every sidebar control they run under the
        viewer's lock, so none of them waits for the playback thread."""
        if self.play_button:
            self._gui_play = sidebar.button(
                "Timeline", "Play", on_click=lambda: self._halt() if self.playing else self.play()
            )
            self._sync_play_button()
        else:
            self._gui_play = sidebar.checkbox(
                "Timeline", "Play", value=self.playing, on_change=lambda value: self.play() if value else self._halt()
            )
        self._gui_slider = sidebar.slider(
            "Timeline", "Step", low=0, high=self.nsteps - 1, step=1, value=self._step, on_change=self._on_slider
        )
        names = {factor: name for name, factor in _SPEEDS.items()}
        sidebar.dropdown(
            "Timeline",
            "Speed",
            options=list(_SPEEDS),
            value=names.get(self.speed, "1\u00d7"),
            on_change=lambda name: setattr(self, "speed", _SPEEDS[name]),
        )
        sidebar.checkbox("Timeline", "Loop", value=self.loop, on_change=lambda value: setattr(self, "loop", value))

    def _on_slider(self, value: float) -> None:
        if int(value) != self._step:
            self.step = int(value)

    def _sync_play_button(self) -> None:
        if self._gui_play is None:
            return
        if self.play_button:
            label = "Pause" if self.playing else "Play"
            if self._gui_play.label != label:
                self._gui_play.label = label
        elif self._gui_play.value != self.playing:
            self._gui_play.value = self.playing
