"""Tukey biweight robust kernel.

See ``docs/concepts/solver_stack.md §5``.
"""

from __future__ import annotations

import torch


class Tukey:
    """Tukey biweight loss — re-descending M-estimator."""

    def __init__(self, *, c: float = 4.685) -> None:
        self.c = c

    def rho(self, squared_norm: torch.Tensor) -> torch.Tensor:
        """Tukey loss: bounded above by ``c²/6`` (re-descending)."""
        c2 = self.c * self.c
        s = squared_norm
        inside = (c2 / 6.0) * (1.0 - (1.0 - s / c2) ** 3)
        outside = torch.full_like(s, c2 / 6.0)
        return torch.where(s <= c2, inside, outside)

    def weight(self, squared_norm: torch.Tensor) -> torch.Tensor:
        """IRLS weight: zero outside the cutoff (rejecting outliers entirely)."""
        c2 = self.c * self.c
        s = squared_norm
        u = (1.0 - s / c2).clamp(min=0.0)
        return u * u
