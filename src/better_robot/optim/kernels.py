"""Robust kernels: objectives and IRLS weights that down-weight outliers of least-squares terms."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import final

import torch

from ..exceptions import ArgumentError


class RobustKernel(ABC):
    """Down-weights outliers without changing residual structure; a new kernel subclasses this.

    ``Problem.residual(..., kernel=)`` applies it to the squared norm of each group (the last axis of the
    weighted residual); without a kernel the loss is the plain half squared norm.
    """

    @abstractmethod
    def rho(self, squared_norm: torch.Tensor) -> torch.Tensor:
        """Objective values ``(...)`` of squared residual norms ``(...)``."""

    @abstractmethod
    def weight(self, squared_norm: torch.Tensor) -> torch.Tensor:
        """IRLS row weights ``(...)`` of squared residual norms ``(...)``: ``rho'``."""


@final
class Huber(RobustKernel):
    """Quadratic loss inside ``delta`` and linear loss outside it."""

    def __init__(self, *, delta: float = 1.0) -> None:
        self.delta = delta

    def rho(self, squared_norm: torch.Tensor) -> torch.Tensor:
        d2 = self.delta * self.delta
        safe_squared_norm = torch.where(
            squared_norm > d2,
            squared_norm,
            torch.ones_like(squared_norm),
        )
        outside = self.delta * torch.sqrt(safe_squared_norm) - 0.5 * d2
        return torch.where(squared_norm <= d2, 0.5 * squared_norm, outside)

    def weight(self, squared_norm: torch.Tensor) -> torch.Tensor:
        d2 = self.delta * self.delta
        squared_norm = squared_norm.clamp(min=0.0)
        safe_squared_norm = torch.where(
            squared_norm > d2,
            squared_norm,
            torch.ones_like(squared_norm),
        )
        return torch.where(
            squared_norm <= d2,
            torch.ones_like(squared_norm),
            self.delta / torch.sqrt(safe_squared_norm + 1e-30),
        )


@final
class Cauchy(RobustKernel):
    """Cauchy (Lorentzian) loss."""

    def __init__(self, *, c: float = 1.0) -> None:
        self.c = c

    def rho(self, squared_norm: torch.Tensor) -> torch.Tensor:
        c2 = self.c * self.c
        return 0.5 * c2 * torch.log1p(squared_norm / c2)

    def weight(self, squared_norm: torch.Tensor) -> torch.Tensor:
        c2 = self.c * self.c
        return 1.0 / (1.0 + squared_norm / c2)


@final
class Tukey(RobustKernel):
    """Tukey biweight loss with a re-descending cutoff."""

    def __init__(self, *, c: float = 4.685) -> None:
        self.c = c

    def rho(self, squared_norm: torch.Tensor) -> torch.Tensor:
        c2 = self.c * self.c
        inside = (c2 / 6.0) * (1.0 - (1.0 - squared_norm / c2) ** 3)
        return torch.where(squared_norm <= c2, inside, torch.full_like(squared_norm, c2 / 6.0))

    def weight(self, squared_norm: torch.Tensor) -> torch.Tensor:
        return (1.0 - squared_norm / (self.c * self.c)).clamp(min=0.0).square()


@final
class GemanMcClure(RobustKernel):
    """Smooth, bounded, re-descending Geman--McClure loss."""

    def __init__(self, *, c: float = 1.0) -> None:
        c = float(c)
        if not math.isfinite(c) or c <= 0.0:
            raise ArgumentError(f"c must be finite and positive, got {c!r}")
        self.c = c

    def rho(self, squared_norm: torch.Tensor) -> torch.Tensor:
        c2 = self.c * self.c
        return 0.5 * c2 * squared_norm / (c2 + squared_norm)

    def weight(self, squared_norm: torch.Tensor) -> torch.Tensor:
        c2 = self.c * self.c
        return (c2 / (c2 + squared_norm)).square()


__all__ = ["Cauchy", "GemanMcClure", "Huber", "RobustKernel", "Tukey"]
