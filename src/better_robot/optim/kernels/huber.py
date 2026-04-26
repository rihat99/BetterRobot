"""Huber robust kernel.

See ``docs/design/07_RESIDUALS_COSTS_SOLVERS.md §5``.
"""

from __future__ import annotations

import torch


class Huber:
    """Huber loss — quadratic inside ``delta``, linear outside."""

    def __init__(self, *, delta: float = 1.0) -> None:
        self.delta = delta

    def rho(self, squared_norm: torch.Tensor) -> torch.Tensor:
        """Huber loss value.

        ``rho(s) = s/2`` for ``s ≤ delta²``, else ``delta·sqrt(s) − delta²/2``.
        """
        d = self.delta
        d2 = d * d
        s = squared_norm
        inside = 0.5 * s
        outside = d * torch.sqrt(s.clamp(min=0.0)) - 0.5 * d2
        return torch.where(s <= d2, inside, outside)

    def weight(self, squared_norm: torch.Tensor) -> torch.Tensor:
        """IRLS weight ``rho'(s) = min(1, delta / sqrt(s))``.

        Returns 1.0 inside the quadratic region (``s ≤ delta²``) and decays
        as ``delta / sqrt(s)`` outside it. The ``sqrt`` of the resulting
        weight multiplies both the residual and its Jacobian — the standard
        IRLS form (Triggs et al. 2000, §4).
        """
        d = self.delta
        s = squared_norm.clamp(min=0.0)
        return torch.where(
            s <= d * d,
            torch.ones_like(s),
            d / torch.sqrt(s + 1e-30),
        )
