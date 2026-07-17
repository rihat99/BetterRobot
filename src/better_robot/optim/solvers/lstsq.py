"""LSTSQ linear solver — rank-deficient safe.

See ``docs/concepts/solver_stack.md §5``.
"""

from __future__ import annotations

import torch

from .base import _regularized_matrix


class LSTSQ:
    """Dense batched least-squares solver via ``torch.linalg.lstsq``."""

    supported_systems = frozenset(("dense",))

    def solve(
        self,
        A: torch.Tensor,
        b: torch.Tensor,
        ridge: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        """Solve ``(A + ridge I) x ≈ b`` for dense batched inputs."""
        matrix = _regularized_matrix(A, ridge)
        return torch.linalg.lstsq(matrix, b.to(A.dtype)).solution.to(b.dtype)
