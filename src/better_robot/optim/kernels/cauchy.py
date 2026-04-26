"""Cauchy robust kernel.

See ``docs/concepts/solver_stack.md §5``.
"""

from __future__ import annotations

import torch


class Cauchy:
    """Cauchy (Lorentzian) loss."""

    def __init__(self, *, c: float = 1.0) -> None:
        self.c = c

    def rho(self, squared_norm: torch.Tensor) -> torch.Tensor:
        """Cauchy loss value: ``rho(s) = (c²/2) · log(1 + s/c²)``."""
        c2 = self.c * self.c
        return 0.5 * c2 * torch.log1p(squared_norm / c2)

    def weight(self, squared_norm: torch.Tensor) -> torch.Tensor:
        """IRLS weight ``rho'(s) = 1 / (1 + s / c²)``."""
        c2 = self.c * self.c
        return 1.0 / (1.0 + squared_norm / c2)
