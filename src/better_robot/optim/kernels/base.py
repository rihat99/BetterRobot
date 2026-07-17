"""``RobustKernel`` protocol — robust objective and IRLS reweighting.

Implementations live beside this file (``L2``, ``Huber``, ``Cauchy``,
``Tukey``).
Given a squared residual, a kernel returns both its robust objective value
and a per-row weight in ``[0, 1]``. The optimiser uses ``rho`` for trial-step
acceptance and multiplies the residual and Jacobian by ``sqrt(weight)`` for
the IRLS normal equations.

See ``docs/concepts/solver_stack.md §5`` and ``docs/conventions/extension.md §6``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class RobustKernel(Protocol):
    """Down-weight outliers without discarding them."""

    def rho(self, squared_norm: torch.Tensor) -> torch.Tensor:
        """Map squared residuals to robust objective values of the same shape."""
        ...

    def weight(self, squared_norm: torch.Tensor) -> torch.Tensor:
        """Map squared residual norms to per-row weights in ``[0, 1]``.

        Input and output share the same shape (``(B...,)`` or richer).
        """
        ...
