"""``recorder.py`` — ``VideoRecorder`` / ``render_trajectory`` placeholders.

Video recording (mp4, image sequences, ffmpeg / imageio) is deferred
out of V1. The ``VideoRecorder`` class and ``render_trajectory`` function
exist here as named targets so that future work can drop an
implementation in without restructuring the surrounding code — calling
them today raises ``NotImplementedError`` pointing at
``docs/design/12_VIEWER.md §10.1``.

The ``test_only_recorder_imports_imageio`` layer rule in
``tests/contract/test_layer_dependencies.py`` reserves ``imageio`` imports for
this module, so the future body can land without leaking dependencies.

See ``docs/design/12_VIEWER.md §10.1``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:
    from .camera import Camera
    from .scene import Scene
    from ..data_model.model import Model
    from ..tasks.trajectory import Trajectory


_FUTURE_MSG = (
    "Video recording is future work — see docs/design/12_VIEWER.md §10.1. "
    "V1 of the viewer is interactive-only; `render_trajectory` and "
    "`VideoRecorder` land together with the offscreen backend (§10.2)."
)


class VideoRecorder:
    """Placeholder for the future video-recording API — see §10.1.

    The eventual implementation wraps a ``Scene`` and a
    ``RendererBackend`` with a capture path, drives a
    ``TrajectoryPlayer`` frame-by-frame, and encodes through
    ``imageio-ffmpeg``. Until then every method raises
    ``NotImplementedError``.
    """

    def __init__(
        self,
        scene: "Scene",
        *,
        fps: int = 30,
        resolution: tuple[int, int] = (1280, 720),
    ) -> None:
        raise NotImplementedError(_FUTURE_MSG)

    def record_trajectory(
        self,
        trajectory: "Trajectory",
        path: str,
        *,
        camera: "Camera | None" = None,
        loop: int = 1,
        speed: float = 1.0,
        on_frame: Any = None,
    ) -> None:
        raise NotImplementedError(_FUTURE_MSG)

    def write_frame(self) -> None:
        raise NotImplementedError(_FUTURE_MSG)

    def save(self, path: str) -> None:
        raise NotImplementedError(_FUTURE_MSG)

    def start_live(self, path: str, *, duration: float | None = None) -> None:
        raise NotImplementedError(_FUTURE_MSG)

    def stop_live(self) -> None:
        raise NotImplementedError(_FUTURE_MSG)


def render_trajectory(
    model: "Model",
    trajectory: "Trajectory",
    path: str,
    *,
    modes: Sequence[str] = ("auto",),
    overlays: Sequence[str] = ("grid", "frame_axes"),
    camera: "Camera | None" = None,
    fps: int = 30,
    resolution: tuple[int, int] = (1280, 720),
    loop: int = 1,
    speed: float = 1.0,
    headless: bool = True,
    theme: Any = None,
) -> None:
    """Placeholder for the code-driven render path — see §10.1.

    The eventual signature matches what ``docs/design/12_VIEWER.md §10.1``
    describes; the signature is already frozen here so callers can import
    the symbol today and get an actionable error at call time.
    """
    raise NotImplementedError(_FUTURE_MSG)
