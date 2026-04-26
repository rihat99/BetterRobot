"""``better_robot.backends`` — runtime backend registry.

The default backend is :mod:`better_robot.backends.torch_native`. Library
hot paths obtain it via :func:`default_backend`; user code can switch the
global default with :func:`set_backend` or pass a different
:class:`~better_robot.backends.protocol.Backend` instance directly via
the public functions' ``backend=`` kwarg (no global mutation).

See ``docs/concepts/batching_and_backends.md §7`` and
``docs/reference/roadmap.md §P1``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from ..exceptions import BackendNotAvailableError
from .protocol import Backend, DynamicsOps, KinematicsOps, LieOps

if TYPE_CHECKING:
    pass

# ──────────────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────────────

BackendName = Literal["torch_native", "warp"]

_KNOWN_BACKENDS: tuple[str, ...] = ("torch_native", "warp")
_DEFAULT_NAME: str = "torch_native"
_INSTANCES: dict[str, Backend] = {}


def _ensure_warp_available() -> None:
    """Raise :class:`BackendNotAvailableError` if ``warp`` cannot be imported."""
    try:
        import warp  # noqa: F401  pylint: disable=import-outside-toplevel
    except ImportError as exc:
        raise BackendNotAvailableError(
            "Warp backend requested but `warp-lang` is not installed. "
            "Install with `pip install warp-lang`."
        ) from exc


def _load(name: str) -> Backend:
    """Construct (or return cached) :class:`Backend` named ``name``."""
    if name in _INSTANCES:
        return _INSTANCES[name]

    if name == "torch_native":
        from .torch_native import BACKEND as _torch_native_backend
        _INSTANCES[name] = _torch_native_backend
        return _torch_native_backend

    if name == "warp":
        _ensure_warp_available()
        # Warp backend lands in P11; until then this is unreachable past
        # the availability check above (warp-lang is not a dependency).
        raise BackendNotAvailableError(
            "Warp backend module is not implemented yet — see P11 of "
            "docs/reference/roadmap.md."
        )

    raise BackendNotAvailableError(
        f"Unknown backend {name!r}; valid: {_KNOWN_BACKENDS}"
    )


def get_backend(name: str) -> Backend:
    """Return the :class:`Backend` instance for ``name``, loading it on demand.

    Raises :class:`BackendNotAvailableError` if the backend's underlying
    dependency is not importable.
    """
    return _load(name)


def default_backend() -> Backend:
    """Return the currently active default :class:`Backend`."""
    return _load(_DEFAULT_NAME)


def current_backend() -> str:
    """Return the *name* of the currently active default backend."""
    return _DEFAULT_NAME


def current() -> Backend:
    """Alias for :func:`default_backend`."""
    return default_backend()


def set_backend(name: str) -> None:
    """Switch the global default backend.

    Raises :class:`BackendNotAvailableError` if the named backend is not
    available; ``_DEFAULT_NAME`` is left unchanged on the failure path.
    """
    global _DEFAULT_NAME
    if name not in _KNOWN_BACKENDS:
        raise BackendNotAvailableError(
            f"Unknown backend {name!r}; valid: {_KNOWN_BACKENDS}"
        )
    # Force-load *before* mutating the default — so a failure doesn't leave
    # the registry inconsistent.
    _load(name)
    _DEFAULT_NAME = name


def graph_capture(fn):
    """Decorator: wrap ``fn`` in a CUDA graph capture if the active backend
    supports it.

    The torch-native backend is a **no-op** — ``fn`` is returned
    unchanged. The real CUDA-graph capture path lands with the Warp
    backend (see ``docs/concepts/batching_and_backends.md §7`` and
    ``docs/conventions/performance.md §5``); calling
    ``set_backend("warp")`` first will route through the Warp
    implementation when it ships. Until then this seam exists so that
    user code can be written against the documented surface and the
    decorator becomes a real graph capture once Warp is available
    without a source diff.
    """
    return fn


__all__ = [
    "Backend",
    "LieOps",
    "KinematicsOps",
    "DynamicsOps",
    "BackendName",
    "default_backend",
    "current",
    "current_backend",
    "set_backend",
    "get_backend",
    "graph_capture",
]
