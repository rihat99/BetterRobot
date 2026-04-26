"""Sparse Cholesky solver (torch.sparse / scipy fallback).

See ``docs/design/07_RESIDUALS_COSTS_SOLVERS.md §5``.
"""

from __future__ import annotations

import torch


class SparseCholesky:
    """Sparse Cholesky solver for block-sparse trajopt Jacobians."""

    def solve(self, A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("see docs/design/07_RESIDUALS_COSTS_SOLVERS.md §5")
