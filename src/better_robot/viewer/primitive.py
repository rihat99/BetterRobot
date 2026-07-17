"""Public handles for styling renderer primitives."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .renderers.base import RendererBackend


@dataclass(frozen=True)
class PrimitiveHandle:
    """A backend-neutral handle to one rendered primitive.

    Callers can update presentation without reaching into a
    :class:`~better_robot.viewer.scene.Scene` or renderer backend.
    """

    _backend: "RendererBackend"
    name: str

    def set_color(self, rgba: tuple[float, float, float, float]) -> None:
        """Set an RGBA colour with components in ``[0, 1]``."""
        if len(rgba) != 4 or any(
            isinstance(component, bool)
            or not isinstance(component, Real)
            or not math.isfinite(component)
            or component < 0.0
            or component > 1.0
            for component in rgba
        ):
            raise ValueError("rgba must contain four components in [0, 1]")
        normalized = tuple(float(component) for component in rgba)
        self._backend.set_color(self.name, normalized)

    def set_scale(self, scale: float) -> None:
        """Set a positive uniform scale relative to the original primitive."""
        if isinstance(scale, bool) or not isinstance(scale, Real) or not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("scale must be finite and positive")
        self._backend.set_scale(self.name, float(scale))
