"""Pairwise signed-distance dispatch table.

Users call ``distance(a, b)``; the dispatch table picks the right kernel
based on ``(type(a), type(b))``. New primitive types register new pairs via
``@register_pair``.

See ``docs/design/09_COLLISION_GEOMETRY.md §4``.
"""

from __future__ import annotations

from typing import Callable

import torch

_PAIR: dict[tuple[type, type], Callable] = {}


def register_pair(type_a: type, type_b: type):
    """Decorator: register a signed-distance kernel for ``(type_a, type_b)``.

    See docs/design/09_COLLISION_GEOMETRY.md §4.
    """

    def _inner(fn):
        _PAIR[(type_a, type_b)] = fn
        return fn

    return _inner


def distance(a, b) -> torch.Tensor:
    """Signed distance between primitives ``a`` and ``b``, broadcasting over
    leading batch shapes.

    ``> 0`` separated, ``0`` touching, ``< 0`` penetration.

    See docs/design/09_COLLISION_GEOMETRY.md §4.
    """
    raise NotImplementedError("see docs/design/09_COLLISION_GEOMETRY.md §4")
