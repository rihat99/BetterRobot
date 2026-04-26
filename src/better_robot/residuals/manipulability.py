"""Yoshikawa manipulability residual.

Scalar-valued (``dim = 1``); maximises the Yoshikawa index
``sqrt(det(J J^T))`` as a cost. Analytic Jacobian is expensive — v1 uses
autodiff (``.jacobian() → None``).

See ``docs/concepts/residuals_and_costs.md §2``.
"""

from __future__ import annotations

import torch

from .base import Residual, ResidualState
from .registry import register_residual


@register_residual("manipulability")
class YoshikawaResidual(Residual):
    """Scalar manipulability residual (``dim = 1``)."""

    def __init__(self, *, frame_id: int, weight: float = 1.0) -> None:
        self.frame_id = frame_id
        self.weight = weight
        self.dim = 1

    def __call__(self, state: ResidualState) -> torch.Tensor:
        raise NotImplementedError("see docs/concepts/residuals_and_costs.md §2")

    def jacobian(self, state: ResidualState) -> torch.Tensor | None:
        """No analytic form in v1 — return ``None`` so the solver falls back to autodiff."""
        return None
