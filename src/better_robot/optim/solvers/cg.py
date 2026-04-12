"""Conjugate-gradient linear solver — large-sparse option.

See ``docs/07_RESIDUALS_COSTS_SOLVERS.md §5``.
"""

from __future__ import annotations

import torch


class CG:
    """Conjugate gradient solver for symmetric PD systems."""

    def __init__(self, *, max_iter: int = 100, tol: float = 1e-8) -> None:
        self.max_iter = max_iter
        self.tol = tol

    def solve(self, A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("see docs/07_RESIDUALS_COSTS_SOLVERS.md §5")
