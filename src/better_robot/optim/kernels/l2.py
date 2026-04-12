"""L2 robust kernel — the trivial identity.

See ``docs/07_RESIDUALS_COSTS_SOLVERS.md §5``.
"""

from __future__ import annotations

import torch


class L2:
    """Trivial identity kernel — weights = 1 everywhere."""

    def weight(self, squared_norm: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(squared_norm)
