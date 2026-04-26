"""Cholesky linear solver — dense SPD.

See ``docs/design/07_RESIDUALS_COSTS_SOLVERS.md §5``.
"""

from __future__ import annotations

import torch


class Cholesky:
    """Dense Cholesky solver for ``A x = b``, SPD ``A``."""

    def solve(self, A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Solve ``A x = b`` for SPD ``A``. Falls back to lstsq on failure."""
        try:
            L = torch.linalg.cholesky(A)
            return torch.cholesky_solve(b.unsqueeze(-1), L).squeeze(-1)
        except Exception:
            return torch.linalg.lstsq(A, b).solution
