"""``renderers`` — renderer backend registry.

See ``docs/design/12_VIEWER.md §9``.
"""

from __future__ import annotations

from .base import RendererBackend

RENDERER_REGISTRY: dict[str, type] = {}


def get_renderer(name: str) -> type:
    """Look up a registered renderer class by name."""
    if name not in RENDERER_REGISTRY:
        raise KeyError(
            f"Unknown renderer: {name!r}. Available: {list(RENDERER_REGISTRY)}"
        )
    return RENDERER_REGISTRY[name]
