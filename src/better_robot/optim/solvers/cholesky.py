"""Cholesky linear solver — dense SPD.

See ``docs/concepts/solver_stack.md §5``.
"""

from __future__ import annotations

import torch

from .base import _regularized_matrix


class Cholesky:
    """Dense batched Cholesky solver for ``(A + ridge I) x = b``.

    Failed batch elements use a least-squares fallback while successful
    elements retain their Cholesky solutions. Capture-ready LM handles its
    stricter zero-step/info-mask policy in the solver update itself.
    """

    def solve(
        self,
        A: torch.Tensor,
        b: torch.Tensor,
        ridge: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        """Solve the dense system, preserving the right-hand-side dtype."""
        matrix = _regularized_matrix(A, ridge)
        rhs = b.to(A.dtype)
        factor, info = torch.linalg.cholesky_ex(matrix)
        chol_solution = torch.cholesky_solve(rhs.unsqueeze(-1), factor).squeeze(-1)
        lstsq_solution = torch.linalg.lstsq(matrix, rhs).solution
        ok = info == 0
        solution = torch.where(
            ok[..., None],
            torch.nan_to_num(chol_solution),
            lstsq_solution,
        )
        return solution.to(b.dtype)
