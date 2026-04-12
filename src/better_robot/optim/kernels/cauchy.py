"""Cauchy robust kernel.

See ``docs/07_RESIDUALS_COSTS_SOLVERS.md §5``.
"""

from __future__ import annotations

import torch


class Cauchy:
    """Cauchy (Lorentzian) loss."""

    def __init__(self, *, c: float = 1.0) -> None:
        self.c = c

    def weight(self, squared_norm: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("see docs/07_RESIDUALS_COSTS_SOLVERS.md §5")
