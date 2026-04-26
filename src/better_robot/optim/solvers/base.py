"""``LinearSolver`` protocol — solves ``A x = b`` inside an optimiser.

Implementations live beside this file under ``optim/solvers/`` (``Cholesky``,
``LSTSQ``, ``CG``, ``SparseCholesky``). A concrete class is a ``LinearSolver``
if it provides a ``solve(A, b) -> Tensor`` method with the right signature;
``@runtime_checkable`` lets callers verify that with ``isinstance``.

See ``docs/concepts/solver_stack.md §5`` and ``docs/conventions/extension.md §5``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class LinearSolver(Protocol):
    """Linear solver for the normal-equation step ``A x = b``.

    ``A`` is the damped Gauss-Newton matrix ``JᵀJ + λI``; ``b`` is ``-Jᵀr``.
    Implementations may fall back to pseudo-inverse on singular ``A``.
    """

    def solve(self, A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Return ``x`` such that ``A x ≈ b``. Shape: ``(B..., n)``."""
        ...
