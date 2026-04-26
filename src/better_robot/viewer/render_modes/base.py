"""``render_modes/base.py`` — ``RenderMode`` protocol and ``RenderContext``.

All render modes implement this protocol.  They speak to a
``RendererBackend`` — they never import viser or pyrender directly.

See ``docs/concepts/viewer.md §4.1``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ...data_model.data import Data
    from ...data_model.model import Model


@dataclass
class RenderContext:
    """Handle passed to every ``RenderMode`` at attach time.

    Contains the backend handle, a per-mode namespace for node names (so two
    modes never collide), the current batch index, and the theme.
    """

    backend: Any               # RendererBackend instance
    namespace: str             # e.g. "/panda/skeleton"
    batch_index: int = 0       # which element of the leading batch axis to draw
    theme: Any = None          # Theme instance (or None → DEFAULT_THEME)


@runtime_checkable
class RenderMode(Protocol):
    """Protocol every render mode implements.

    Lifecycle::

        1. is_available(model, data0) → bool       — classmethod
        2. attach(context, model, data0)           — one-time setup
        3. update(data)                            — called on every new Data
        4. set_visible(visible: bool)              — UI toggle
        5. detach()                                — tear down backend nodes
    """

    name: ClassVar[str]         # short UI label, e.g. "Skeleton"
    description: ClassVar[str]  # tooltip text

    @classmethod
    def is_available(cls, model: "Model", data: "Data") -> bool:
        """Return True iff this mode can render this model."""
        ...

    def attach(self, context: RenderContext, model: "Model", data: "Data") -> None:
        """One-time setup: add geometry nodes to the backend."""
        ...

    def update(self, data: "Data") -> None:
        """Push new poses/data to all backend nodes."""
        ...

    def set_visible(self, visible: bool) -> None:
        """Show or hide all nodes owned by this mode."""
        ...

    def detach(self) -> None:
        """Remove all backend nodes created by this mode."""
        ...
