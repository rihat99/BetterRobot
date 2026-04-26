"""Warp backend — **skeleton only**.

Nothing is wired up in v1. When this lands, users will switch to it via
``backends.set_backend("warp")`` and everything above this layer stays
untouched. See ``docs/concepts/batching_and_backends.md §7``.
"""

from __future__ import annotations


def enable_warp_backend() -> None:
    """Activate the Warp backend at runtime. Raises ``NotImplementedError`` in v1."""
    raise NotImplementedError("Warp backend is phase 7 — see docs/concepts/batching_and_backends.md §7")


def disable_warp_backend() -> None:
    """Deactivate the Warp backend and fall back to torch_native."""
    raise NotImplementedError("Warp backend is phase 7 — see docs/concepts/batching_and_backends.md §7")


__all__ = ["enable_warp_backend", "disable_warp_backend"]
