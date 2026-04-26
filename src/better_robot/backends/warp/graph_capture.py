"""CUDA graph capture context manager exposed as ``@graph_capture``.

Skeleton only. See ``docs/design/10_BATCHING_AND_BACKENDS.md §7``.
"""

from __future__ import annotations


def graph_capture(fn):
    """Decorator: build a CUDA graph from the decorated function.

    Real implementation lands in phase 7.
    """
    raise NotImplementedError("see docs/design/10_BATCHING_AND_BACKENDS.md §7")
