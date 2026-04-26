"""LSTSQ linear solver — rank-deficient safe.

See ``docs/concepts/solver_stack.md §5``.
"""

from __future__ import annotations

import torch


class LSTSQ:
    """Least-squares solver via ``torch.linalg.lstsq``."""

    def solve(self, A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Solve via ``torch.linalg.lstsq`` (handles rank-deficient A)."""
        return torch.linalg.lstsq(A, b).solution
