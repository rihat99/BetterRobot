"""``Timeline``: the current frame, playback by wall clock, and its GUI controls."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any


class Timeline:
    """Frame counter for ``num_frames`` frames at ``fps``.

    Setting ``frame`` clamps to ``[0, num_frames - 1]`` and notifies the ``on_frame`` callbacks
    (the viewer pushes the frame from there). ``play`` starts a thread that advances by wall
    clock, so slow pushes drop frames instead of slowing playback.
    """

    def __init__(self, num_frames: int = 1, fps: float = 30.0, loop: bool = True) -> None:
        if num_frames < 1 or fps <= 0.0:
            raise ValueError("num_frames must be at least 1 and fps positive")
        self.num_frames = num_frames
        self.fps = fps
        self.loop = loop
        self.speed = 1.0
        self._frame = 0
        self._callbacks: list[Callable[[int], None]] = []
        self._playing = threading.Event()
        self._thread: threading.Thread | None = None
        self._gui_slider: Any = None
        self._gui_play: Any = None

    @property
    def frame(self) -> int:
        return self._frame

    @frame.setter
    def frame(self, value: int) -> None:
        value = min(max(int(value), 0), self.num_frames - 1)
        self._frame = value
        for callback in self._callbacks:
            callback(value)
        if self._gui_slider is not None and self._gui_slider.value != value:
            self._gui_slider.value = value

    @property
    def playing(self) -> bool:
        return self._playing.is_set()

    @property
    def time(self) -> float:
        """Time of the current frame in seconds."""
        return self._frame / self.fps

    def on_frame(self, callback: Callable[[int], None]) -> None:
        """Call ``callback(frame)`` after every frame change."""
        self._callbacks.append(callback)

    def extend(self, num_frames: int) -> None:
        """Grow to at least ``num_frames`` frames (layers added later may be longer)."""
        if num_frames > self.num_frames:
            self.num_frames = num_frames
            if self._gui_slider is not None:
                self._gui_slider.max = num_frames - 1

    def play(self) -> None:
        if self.playing or self.num_frames == 1:
            return
        if not self.loop and self._frame >= self.num_frames - 1:
            self.frame = 0
        self._playing.set()
        self._thread = threading.Thread(target=self._run, name="better_robot.viewer.timeline", daemon=True)
        self._thread.start()
        self._sync_play_button()

    def pause(self) -> None:
        self._playing.clear()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join()
        self._thread = None
        self._sync_play_button()

    def toggle(self) -> None:
        self.pause() if self.playing else self.play()

    def _run(self) -> None:
        start_time = time.perf_counter()
        start_frame = self._frame
        while self.playing:
            elapsed = (time.perf_counter() - start_time) * self.speed
            target = start_frame + int(elapsed * self.fps)
            if target >= self.num_frames:
                if not self.loop:
                    self.frame = self.num_frames - 1
                    self._playing.clear()
                    self._sync_play_button()
                    return
                target %= self.num_frames
            if target != self._frame:
                self.frame = target
            time.sleep(min(0.5 / (self.fps * self.speed), 0.05))

    def build_gui(self, gui: Any) -> None:
        """Play/pause, frame slider, speed and loop controls in the given GUI api or folder."""
        self._gui_play = gui.add_button("Play")
        self._gui_play.on_click(lambda _: self.toggle())
        self._gui_slider = gui.add_slider("Frame", min=0, max=self.num_frames - 1, step=1, initial_value=self._frame)
        self._gui_slider.on_update(lambda _: self._on_slider())
        speed = gui.add_slider("Speed", min=0.1, max=4.0, step=0.1, initial_value=self.speed)
        speed.on_update(lambda _: self._set_speed(speed.value))
        loop = gui.add_checkbox("Loop", initial_value=self.loop)
        loop.on_update(lambda _: setattr(self, "loop", bool(loop.value)))

    def _on_slider(self) -> None:
        if self._gui_slider.value != self._frame:
            self.frame = self._gui_slider.value

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
