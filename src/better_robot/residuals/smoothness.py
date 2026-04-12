"""Smoothness residuals — 5-point finite-difference velocity/accel/jerk.

These consume trajectory-shaped variables ``(B, T, nq)`` and produce
smoothness penalties vectorised along the time axis. Analytic Jacobians
are tridiagonal (sparse) — see ``ResidualSpec`` in ``optim/jacobian_spec.py``.

See ``docs/07_RESIDUALS_COSTS_SOLVERS.md §2`` + ``docs/08_TASKS.md §3``.
"""

from __future__ import annotations

import torch

from .base import Residual, ResidualState
from .registry import register_residual


@register_residual("velocity_5pt")
class Velocity5pt(Residual):
    """5-point finite-difference velocity residual. ``dim = nv * (T - 4)``."""

    def __init__(self, *, weight: float = 1.0, dt: float = 1.0) -> None:
        self.weight = weight
        self.dt = dt
        self.dim = 0  # computed at first call from variables.shape

    def __call__(self, state: ResidualState) -> torch.Tensor:
        raise NotImplementedError("see docs/07_RESIDUALS_COSTS_SOLVERS.md §2")

    def jacobian(self, state: ResidualState) -> torch.Tensor | None:
        raise NotImplementedError("see docs/07_RESIDUALS_COSTS_SOLVERS.md §2")


@register_residual("accel_5pt")
class Accel5pt(Residual):
    """5-point finite-difference acceleration residual."""

    def __init__(self, *, weight: float = 1.0, dt: float = 1.0) -> None:
        self.weight = weight
        self.dt = dt
        self.dim = 0

    def __call__(self, state: ResidualState) -> torch.Tensor:
        raise NotImplementedError("see docs/07_RESIDUALS_COSTS_SOLVERS.md §2")

    def jacobian(self, state: ResidualState) -> torch.Tensor | None:
        raise NotImplementedError("see docs/07_RESIDUALS_COSTS_SOLVERS.md §2")


@register_residual("jerk_5pt")
class Jerk5pt(Residual):
    """5-point finite-difference jerk residual."""

    def __init__(self, *, weight: float = 1.0, dt: float = 1.0) -> None:
        self.weight = weight
        self.dt = dt
        self.dim = 0

    def __call__(self, state: ResidualState) -> torch.Tensor:
        raise NotImplementedError("see docs/07_RESIDUALS_COSTS_SOLVERS.md §2")

    def jacobian(self, state: ResidualState) -> torch.Tensor | None:
        raise NotImplementedError("see docs/07_RESIDUALS_COSTS_SOLVERS.md §2")
