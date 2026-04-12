"""Huber robust kernel.

See ``docs/07_RESIDUALS_COSTS_SOLVERS.md §5``.
"""

from __future__ import annotations

import torch


class Huber:
    """Huber loss — quadratic inside ``delta``, linear outside."""

    def __init__(self, *, delta: float = 1.0) -> None:
        self.delta = delta

    def weight(self, squared_norm: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("see docs/07_RESIDUALS_COSTS_SOLVERS.md §5")
