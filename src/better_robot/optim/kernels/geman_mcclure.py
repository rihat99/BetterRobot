"""Geman--McClure robust kernel.

The implementation uses the same half-quadratic normalization as the other
BetterRobot kernels: near zero, ``rho(s) = s / 2 + O(s**2)`` and the
corresponding IRLS row weight is ``weight(s) = 2 * rho'(s)``.
"""

from __future__ import annotations

import math

import torch


class GemanMcClure:
    """Smooth, bounded, re-descending Geman--McClure loss.

    Parameters
    ----------
    c
        Positive residual scale. The loss approaches ``c**2 / 2`` for
        arbitrarily large squared norms.
    """

    def __init__(self, *, c: float = 1.0) -> None:
        c = float(c)
        if not math.isfinite(c) or c <= 0.0:
            raise ValueError(f"c must be finite and positive, got {c!r}")
        self.c = c

    def rho(self, squared_norm: torch.Tensor) -> torch.Tensor:
        """Return ``(c²/2) * s / (c² + s)`` for squared norm ``s``."""
        c2 = self.c * self.c
        return 0.5 * c2 * squared_norm / (c2 + squared_norm)

    def weight(self, squared_norm: torch.Tensor) -> torch.Tensor:
        """Return normalized IRLS weight ``c⁴ / (c² + s)²``."""
        c2 = self.c * self.c
        return (c2 / (c2 + squared_norm)) ** 2
