"""``better_robot.backends`` — runtime backend selector.

The default backend is ``torch_native``. Users can switch to the Warp
backend once it lands with ``set_backend("warp")``.

See ``docs/10_BATCHING_AND_BACKENDS.md §7``.
"""

from __future__ import annotations

from typing import Literal

_CURRENT: str = "torch_native"

BackendName = Literal["torch_native", "warp"]


def current_backend() -> str:
    """Return the name of the currently active backend."""
    return _CURRENT


def set_backend(name: BackendName) -> None:
    """Switch the active backend.

    Valid values: ``"torch_native"`` (default) or ``"warp"`` (once
    ``better_robot.backends.warp`` lands). See
    ``docs/10_BATCHING_AND_BACKENDS.md §7``.
    """
    global _CURRENT
    if name not in ("torch_native", "warp"):
        raise ValueError(f"unknown backend: {name!r}")
    if name == "warp":
        raise NotImplementedError(
            "Warp backend is not available yet — see docs/10_BATCHING_AND_BACKENDS.md §7"
        )
    _CURRENT = name


def graph_capture(fn):
    """Decorator: wrap ``fn`` in a CUDA graph capture if the Warp backend is active.

    With ``torch_native`` (default) this is a no-op. See
    ``docs/10_BATCHING_AND_BACKENDS.md §7``.
    """
    return fn


__all__ = ["current_backend", "set_backend", "graph_capture", "BackendName"]
