"""``RobustKernel`` protocol — re-weighting applied to a residual.

Implementations live beside this file (``L2``, ``Huber``, ``Cauchy``).
Given the squared norm of a residual, a kernel returns a per-row weight
in ``[0, 1]`` that the optimiser multiplies into both the residual and
its Jacobian (IRLS / M-estimator form).

See ``docs/design/07_RESIDUALS_COSTS_SOLVERS.md §5`` and ``docs/conventions/15_EXTENSION.md §6``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class RobustKernel(Protocol):
    """Down-weight outliers without discarding them."""

    def weight(self, squared_norm: torch.Tensor) -> torch.Tensor:
        """Map squared residual norms to per-row weights in ``[0, 1]``.

        Input and output share the same shape (``(B...,)`` or richer).
        """
        ...
