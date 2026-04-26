"""Residual registry — the ``@register_residual`` decorator.

Third-party users add residuals by decorating a class. The solver has no
idea the registry exists — residuals are composed into a ``CostStack`` and
the stack is passed to the solver.

See ``docs/concepts/residuals_and_costs.md §2``.
"""

from __future__ import annotations

from typing import Callable

from .base import Residual

_REGISTRY: dict[str, type] = {}


def register_residual(name: str) -> Callable[[type], type]:
    """Decorator: register a ``Residual`` subclass under ``name``."""

    def _inner(cls):
        if name in _REGISTRY:
            raise ValueError(f"residual {name!r} already registered")
        _REGISTRY[name] = cls
        cls.name = name
        return cls

    return _inner


def get_residual(name: str) -> type:
    """Look up a registered residual class by name."""
    return _REGISTRY[name]


def registered_residuals() -> tuple[str, ...]:
    """Return the names of all registered residuals."""
    return tuple(_REGISTRY.keys())
