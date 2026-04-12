"""``camera.py`` — static ``Camera`` and ``CameraPath`` placeholder.

V1 keeps the ``Camera`` dataclass so callers can pin a view in future
interactive backends. ``CameraPath`` (orbit, follow_frame, static) is
future work — see ``docs/12_VIEWER.md §10.7``. It lives here as a
named target so an eventual cinematic-camera implementation can slot in
without restructuring the surrounding code.

See ``docs/12_VIEWER.md §10.7``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from ..data_model.model import Model
    from ..tasks.trajectory import Trajectory


_FUTURE_MSG = (
    "CameraPath (orbit / follow_frame / static) is future work — "
    "see docs/12_VIEWER.md §10.7. V1 ships only the static Camera "
    "dataclass."
)


@dataclass(frozen=True)
class Camera:
    """Static camera pose.

    ``position`` and ``look_at`` are world-frame ``(3,)`` tensors.
    Used by future offscreen / cinematic paths — the V1 viser backend
    does not consume it yet.
    """

    position: torch.Tensor           # (3,)
    look_at: torch.Tensor            # (3,)
    up: tuple[float, float, float] = (0.0, 0.0, 1.0)
    fov_deg: float = 50.0
    near: float = 0.01
    far: float = 100.0


class CameraPath:
    """Placeholder for the future cinematic-camera API — see §10.7."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(_FUTURE_MSG)

    def at(self, k: int) -> Camera:
        raise NotImplementedError(_FUTURE_MSG)

    @classmethod
    def orbit(
        cls,
        *,
        center: torch.Tensor,
        radius: float,
        n_frames: int,
        axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
        elevation_deg: float = 20.0,
    ) -> "CameraPath":
        raise NotImplementedError(_FUTURE_MSG)

    @classmethod
    def follow_frame(
        cls,
        model: "Model",
        trajectory: "Trajectory",
        *,
        frame: str,
        offset: torch.Tensor,
    ) -> "CameraPath":
        raise NotImplementedError(_FUTURE_MSG)

    @classmethod
    def static(cls, camera: Camera, n_frames: int) -> "CameraPath":
        raise NotImplementedError(_FUTURE_MSG)
