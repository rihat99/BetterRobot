"""``render_modes`` — render mode registry.

See ``docs/concepts/viewer.md §4.5``.
"""

from __future__ import annotations

from .base import RenderContext, RenderMode

MODE_REGISTRY: dict[str, type] = {}


def register_mode(cls: type) -> type:
    """Class decorator — registers a mode under ``cls.__name__``."""
    MODE_REGISTRY[cls.__name__] = cls
    return cls


# Register built-in modes
from .skeleton import SkeletonMode  # noqa: E402
from .urdf_mesh import URDFMeshMode  # noqa: E402
from .collision import CollisionMode  # noqa: E402

register_mode(SkeletonMode)
register_mode(URDFMeshMode)
register_mode(CollisionMode)

__all__ = [
    "MODE_REGISTRY",
    "register_mode",
    "RenderContext",
    "RenderMode",
    "SkeletonMode",
    "URDFMeshMode",
    "CollisionMode",
]
