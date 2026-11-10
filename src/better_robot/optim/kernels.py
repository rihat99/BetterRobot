"""Robust kernels: objectives and IRLS weights that down-weight outliers of least-squares terms."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import final, override

import torch

from ..exceptions import ArgumentError


class RobustKernel(ABC):
    """Down-weights outliers without changing residual structure; a new kernel subclasses this.

    ``Problem.residual(..., kernel=)`` applies it to the squared norm of each group (the last axis of the
    weighted residual); without a kernel the loss is the plain half squared norm. The library's kernels are
    frozen values, so a kernel held by a registered term cannot change under it.
    ``better_robot.testing.check_robust_kernel`` checks a new kernel against this contract.
    """

    @abstractmethod
    def rho(self, squared_norm: torch.Tensor) -> torch.Tensor:
        """Objective values ``(...)`` of squared residual norms ``squared_norm: (...)``."""

    @abstractmethod
    def weight(self, squared_norm: torch.Tensor) -> torch.Tensor:
        """IRLS row weights ``(...)`` of squared residual norms ``squared_norm: (...)``: ``2 d rho / ds``.

        Twice the derivative, because the solvers scale against the plain loss ``s / 2``, whose weight is 1.
        """


@final
@dataclass(frozen=True, kw_only=True)
class Huber(RobustKernel):
    """Quadratic loss inside ``delta`` and linear loss outside it."""

    delta: float = 1.0

    @override
    def rho(self, squared_norm: torch.Tensor) -> torch.Tensor:
        d2 = self.delta * self.delta
        safe_squared_norm = torch.where(
            squared_norm > d2,
            squared_norm,
            torch.ones_like(squared_norm),
        )
        outside = self.delta * torch.sqrt(safe_squared_norm) - 0.5 * d2
        return torch.where(squared_norm <= d2, 0.5 * squared_norm, outside)

    @override
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
@dataclass(frozen=True, kw_only=True)
class Cauchy(RobustKernel):
    """Cauchy (Lorentzian) loss."""

    c: float = 1.0

    @override
    def rho(self, squared_norm: torch.Tensor) -> torch.Tensor:
        c2 = self.c * self.c
        return 0.5 * c2 * torch.log1p(squared_norm / c2)

    @override
    def weight(self, squared_norm: torch.Tensor) -> torch.Tensor:
        c2 = self.c * self.c
        return 1.0 / (1.0 + squared_norm / c2)


@final
@dataclass(frozen=True, kw_only=True)
class Tukey(RobustKernel):
    """Tukey biweight loss with a re-descending cutoff."""

    c: float = 4.685

    @override
    def rho(self, squared_norm: torch.Tensor) -> torch.Tensor:
        c2 = self.c * self.c
        inside = (c2 / 6.0) * (1.0 - (1.0 - squared_norm / c2) ** 3)
        return torch.where(squared_norm <= c2, inside, torch.full_like(squared_norm, c2 / 6.0))

    @override
    def weight(self, squared_norm: torch.Tensor) -> torch.Tensor:
        return (1.0 - squared_norm / (self.c * self.c)).clamp(min=0.0).square()


@final
@dataclass(frozen=True, kw_only=True)
class GemanMcClure(RobustKernel):
    """Smooth, bounded, re-descending Geman--McClure loss."""

    c: float = 1.0

    def __post_init__(self) -> None:
        """Raise ``ArgumentError`` unless the scale ``c`` is finite and positive."""
        if not math.isfinite(self.c) or self.c <= 0.0:
            raise ArgumentError(f"c must be finite and positive, got {self.c!r}")

    @override
    def rho(self, squared_norm: torch.Tensor) -> torch.Tensor:
        c2 = self.c * self.c
        return 0.5 * c2 * squared_norm / (c2 + squared_norm)

    @override
    def weight(self, squared_norm: torch.Tensor) -> torch.Tensor:
        c2 = self.c * self.c
        return (c2 / (c2 + squared_norm)).square()


__all__ = ["Cauchy", "GemanMcClure", "Huber", "RobustKernel", "Tukey"]
