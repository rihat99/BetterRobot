"""``DampingStrategy`` protocol — controls how ``λ`` evolves across LM iters.

Implementations live beside this file (``Adaptive``, ``Constant``,
``TrustRegion``). A strategy exposes three methods: ``init`` (initial
``λ`` for a fresh problem), ``accept`` (scale after a successful step),
``reject`` (scale after a rejected step).

See ``docs/concepts/solver_stack.md §5`` and ``docs/conventions/extension.md §4``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class DampingStrategy(Protocol):
    """Controls the damping term ``λ`` in Levenberg-Marquardt."""

    def init(self, problem) -> float:
        """Return the initial ``λ`` for ``problem`` (often a fixed constant)."""
        ...

    def accept(self, lam: float) -> float:
        """Return the new ``λ`` after a successful step."""
        ...

    def reject(self, lam: float) -> float:
        """Return the new ``λ`` after a rejected step."""
        ...
