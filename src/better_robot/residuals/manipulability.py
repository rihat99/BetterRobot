"""Yoshikawa manipulability residual.

Scalar-valued (``dim = 1``); maximises the Yoshikawa index
``sqrt(det(J J^T))`` as a cost. Analytic Jacobian is expensive, so v1 uses
autodiff.

See ``docs/concepts/residuals_and_costs.md §2``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from .base import Residual


class YoshikawaResidual(Residual):
    """Scalar manipulability residual (``dim = 1``)."""

    name: str = "manipulability"
    reads = ("q", "data")

    def __init__(self, *, frame_id: int, weight: float = 1.0) -> None:
        self.frame_id = frame_id
        self.weight = weight
        self.dim = 1

    def __call__(self, ctx: Mapping[str, Any]) -> torch.Tensor:
        del ctx
        raise NotImplementedError("see docs/concepts/residuals_and_costs.md §2")
