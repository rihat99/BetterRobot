"""``Timeline``: the current step, playback by wall clock, and its GUI controls."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any


class Timeline:
    """Frame counter for ``nsteps`` steps at ``fps``.

    Setting ``step`` clamps to ``[0, nsteps - 1]`` and notifies the ``on_step`` callbacks
    (the viewer pushes the step from there). ``play`` starts a thread that advances by wall
    clock, so slow pushes drop steps instead of slowing playback.
    """

    def __init__(self, nsteps: int = 1, fps: float = 30.0, loop: bool = True) -> None:
        if nsteps < 1 or fps <= 0.0:
            raise ValueError("nsteps must be at least 1 and fps positive")
        self.nsteps = nsteps
        self.fps = fps
        self.loop = loop
        self.speed = 1.0
        self._step = 0
        self._callbacks: list[Callable[[int], None]] = []
        self._playing = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._gui_slider: Any = None
        self._gui_play: Any = None

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
            self.nsteps = nsteps
            if self._gui_slider is not None:
                self._gui_slider.max = nsteps - 1

    def play(self) -> None:
        """Start playback in a background thread; a no-op when already playing or ``nsteps == 1``.

        Without ``loop``, playback that sits on the last step restarts from step 0.
        """
        with self._lock:  # two clients clicking Play together must not start two threads
            if self.playing or self.nsteps == 1:
                return
            self._playing.set()
            self._thread = threading.Thread(target=self._run, name="better_robot.viewer.timeline", daemon=True)
        if not self.loop and self._step >= self.nsteps - 1:
            self.step = 0
        self._thread.start()
        self._sync_play_button()

    def pause(self) -> None:
        """Stop playback and wait for the playback thread to exit; the current step is kept."""
        self._playing.clear()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join()
        self._thread = None
        self._sync_play_button()

    def toggle(self) -> None:
        """Pause when playing, play otherwise (the sidebar's Play button)."""
        self.pause() if self.playing else self.play()

    def _run(self) -> None:
        start_time = time.perf_counter()
        start_step = self._step
        while self.playing:
            elapsed = (time.perf_counter() - start_time) * self.speed
            target = start_step + int(elapsed * self.fps)
            if target >= self.nsteps:
                if not self.loop:
                    self.step = self.nsteps - 1
                    self._playing.clear()
                    self._sync_play_button()
                    return
                target %= self.nsteps
            if target != self._step:
                self.step = target
            time.sleep(min(0.5 / (self.fps * self.speed), 0.05))

    def build_gui(self, gui: Any) -> None:
        """Play/pause, step slider, speed and loop controls in the given GUI api or folder."""
        self._gui_play = gui.add_button("Play")
        self._gui_play.on_click(lambda _: self.toggle())
        self._gui_slider = gui.add_slider("Step", min=0, max=self.nsteps - 1, step=1, initial_value=self._step)
        self._gui_slider.on_update(lambda _: self._on_slider())
        speed = gui.add_slider("Speed", min=0.1, max=4.0, step=0.1, initial_value=self.speed)
        speed.on_update(lambda _: self._set_speed(speed.value))
        loop = gui.add_checkbox("Loop", initial_value=self.loop)
        loop.on_update(lambda _: setattr(self, "loop", bool(loop.value)))

    def _on_slider(self) -> None:
        if self._gui_slider.value != self._step:
            self.step = self._gui_slider.value

    def _set_speed(self, speed: float) -> None:
        was_playing = self.playing
        if was_playing:
            self.pause()
        self.speed = float(speed)
        if was_playing:
            self.play()

    def _sync_play_button(self) -> None:
        if self._gui_play is not None:
            self._gui_play.label = "Pause" if self.playing else "Play"
