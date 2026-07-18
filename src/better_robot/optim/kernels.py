"""Robust objectives and IRLS weights for least-squares problems."""

from __future__ import annotations

import math
from typing import Protocol, runtime_checkable

import torch


def _group_rows(rows: torch.Tensor, group_size: int) -> torch.Tensor:
    """View the last residual axis as contiguous robust groups."""
    return rows.reshape(*rows.shape[:-1], rows.shape[-1] // group_size, group_size)


@runtime_checkable
class RobustKernel(Protocol):
    """Down-weight outliers without changing residual structure."""

    def rho(self, squared_norm: torch.Tensor) -> torch.Tensor:
        """Map squared residual norms to objective values."""
        ...

    def weight(self, squared_norm: torch.Tensor) -> torch.Tensor:
        """Map squared residual norms to IRLS row weights."""
        ...


class L2:
    """The ordinary half-squared L2 objective."""

    def rho(self, squared_norm: torch.Tensor) -> torch.Tensor:
        return 0.5 * squared_norm

    def weight(self, squared_norm: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(squared_norm)


class Huber:
    """Quadratic loss inside ``delta`` and linear loss outside it."""

    def __init__(self, *, delta: float = 1.0) -> None:
        self.delta = delta

    def rho(self, squared_norm: torch.Tensor) -> torch.Tensor:
        d2 = self.delta * self.delta
        outside = self.delta * torch.sqrt(squared_norm.clamp(min=0.0)) - 0.5 * d2
        return torch.where(squared_norm <= d2, 0.5 * squared_norm, outside)

    def weight(self, squared_norm: torch.Tensor) -> torch.Tensor:
        squared_norm = squared_norm.clamp(min=0.0)
        return torch.where(
            squared_norm <= self.delta * self.delta,
            torch.ones_like(squared_norm),
            self.delta / torch.sqrt(squared_norm + 1e-30),
        )


class Cauchy:
    """Cauchy (Lorentzian) loss."""

    def __init__(self, *, c: float = 1.0) -> None:
        self.c = c

    def rho(self, squared_norm: torch.Tensor) -> torch.Tensor:
        c2 = self.c * self.c
        return 0.5 * c2 * torch.log1p(squared_norm / c2)

    def weight(self, squared_norm: torch.Tensor) -> torch.Tensor:
        c2 = self.c * self.c
        return 1.0 / (1.0 + squared_norm / c2)


class Tukey:
    """Tukey biweight loss with a re-descending cutoff."""

    def __init__(self, *, c: float = 4.685) -> None:
        self.c = c

    def rho(self, squared_norm: torch.Tensor) -> torch.Tensor:
        c2 = self.c * self.c
        inside = (c2 / 6.0) * (1.0 - (1.0 - squared_norm / c2) ** 3)
        return torch.where(squared_norm <= c2, inside, torch.full_like(squared_norm, c2 / 6.0))

    def weight(self, squared_norm: torch.Tensor) -> torch.Tensor:
        return (1.0 - squared_norm / (self.c * self.c)).clamp(min=0.0).square()


class GemanMcClure:
    """Smooth, bounded, re-descending Geman--McClure loss."""

    def __init__(self, *, c: float = 1.0) -> None:
        c = float(c)
        if not math.isfinite(c) or c <= 0.0:
            raise ValueError(f"c must be finite and positive, got {c!r}")
        self.c = c

    def rho(self, squared_norm: torch.Tensor) -> torch.Tensor:
        c2 = self.c * self.c
        return 0.5 * c2 * squared_norm / (c2 + squared_norm)

    def weight(self, squared_norm: torch.Tensor) -> torch.Tensor:
        c2 = self.c * self.c
        return (c2 / (c2 + squared_norm)).square()


__all__ = ["Cauchy", "GemanMcClure", "Huber", "L2", "RobustKernel", "Tukey"]
