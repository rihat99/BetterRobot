"""L2 robust kernel — the trivial identity.

See ``docs/concepts/solver_stack.md §5``.
"""

from __future__ import annotations

import torch


class L2:
    """Trivial identity kernel — weights = 1 everywhere."""

    def rho(self, squared_norm: torch.Tensor) -> torch.Tensor:
        return 0.5 * squared_norm

    def weight(self, squared_norm: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(squared_norm)
