"""Manifold-aware smoothness residuals over trajectory variables."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping
import math
from numbers import Real

import torch

from ._temporal_jacobian import dense_temporal_residual
from .utils import RobotLike, RobotVariableLike as _RobotTrajectory, matches
from .base import Residual, Weight
from .structure import TemporalPattern


def _trajectory(variable: _RobotTrajectory, name: str) -> tuple[torch.Tensor, int]:
    if not isinstance(variable, RobotLike):
        raise TypeError(f"{name} q must be a RobotVariable, got {type(variable).__name__}")
    if variable.time_axis != 0 or len(variable.shape) != 2:
        raise ValueError(f"{name} q must declare time_axis=0 with event shape (T, nq), got {variable.shape}")
    horizon = variable.time_length
    if horizon < 3:
        raise ValueError(f"{name} needs at least 3 timesteps, got T={horizon}")
    return variable.tensor, horizon


def _validate_dt(dt: Real, name: str) -> float:
    if isinstance(dt, bool) or not isinstance(dt, Real) or not math.isfinite(float(dt)) or float(dt) <= 0.0:
        raise ValueError(f"{name} dt must be finite and positive, got {dt!r}")
    return float(dt)


def _anchor_constant_block(block: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    anchor = q.sum(dim=(-2, -1)) * 0.0
    return block + anchor[..., None, None, None]


class _SmoothnessResidual(Residual):
    _offsets: tuple[int, ...]
    _coefficients: tuple[float, ...]

    def __init__(self, q, *, dt, weight, row_weight, kernel, name) -> None:
        _value, horizon = _trajectory(q, type(self).__name__)
        self.q, self.model = q, q.model
        self.dt = _validate_dt(dt, type(self).__name__)
        self.horizon = horizon
        super().__init__(
            q,
            dim=(horizon - 2) * q.model.nv,
            weight=weight,
            row_weight=row_weight,
            kernel=kernel,
            name=name,
        )

    def temporal_structure(self, variable: _RobotTrajectory | str) -> TemporalPattern | None:
        return TemporalPattern(self.horizon - 2, self.model.nv, 1, self._offsets) if matches(variable, self.q) else None

    @abstractmethod
    def _row_scale(self) -> float:
        """Return this finite-difference stencil's row scale."""

    def temporal_jacobian_blocks(self, variable: _RobotTrajectory | str) -> Mapping[int, torch.Tensor]:
        if not matches(variable, self.q):
            return {}
        q, horizon = _trajectory(self.q, type(self).__name__)
        identity = torch.eye(self.model.nv, dtype=q.dtype, device=q.device)
        base = identity.expand(*q.shape[:-2], horizon - 2, self.model.nv, self.model.nv)
        scale = self._row_scale()
        return {
            offset: _anchor_constant_block(coefficient * scale * base, q)
            for offset, coefficient in zip(self._offsets, self._coefficients, strict=True)
        }

    def jacobian(self) -> tuple[torch.Tensor, ...]:
        return dense_temporal_residual(self, self.q, self.horizon)


class VelocityResidual(_SmoothnessResidual):
    """Three-point central-difference velocity in robot tangent space."""

    _offsets, _coefficients = (-1, 1), (-1.0, 1.0)

    def __init__(
        self,
        q: _RobotTrajectory,
        *,
        dt: Real,
        weight: Real | torch.Tensor = 1.0,
        row_weight: Weight | Real | torch.Tensor = 1.0,
        kernel: object | None = None,
        name: str = "velocity",
    ) -> None:
        super().__init__(q, dt=dt, weight=weight, row_weight=row_weight, kernel=kernel, name=name)

    def _row_scale(self) -> float:
        return 1.0 / (2.0 * self.dt)

    def error(self) -> torch.Tensor:
        q, _horizon = _trajectory(self.q, type(self).__name__)
        velocity = self.model.difference(q[..., :-2, :], q[..., 2:, :]) / (2.0 * self.dt)
        return velocity.reshape(*q.shape[:-2], self.dim)


class AccelerationResidual(_SmoothnessResidual):
    """Three-point tangent-space acceleration over a robot trajectory."""

    _offsets, _coefficients = (-1, 0, 1), (1.0, -2.0, 1.0)

    def __init__(
        self,
        q: _RobotTrajectory,
        *,
        dt: Real,
        weight: Real | torch.Tensor = 1.0,
        row_weight: Weight | Real | torch.Tensor = 1.0,
        kernel: object | None = None,
        name: str = "acceleration",
    ) -> None:
        super().__init__(q, dt=dt, weight=weight, row_weight=row_weight, kernel=kernel, name=name)

    def _row_scale(self) -> float:
        return 1.0 / self.dt**2

    def error(self) -> torch.Tensor:
        q, _horizon = _trajectory(self.q, type(self).__name__)
        forward = self.model.difference(q[..., 1:-1, :], q[..., 2:, :])
        backward = self.model.difference(q[..., :-2, :], q[..., 1:-1, :])
        acceleration = (forward - backward) / self.dt**2
        return acceleration.reshape(*q.shape[:-2], self.dim)


__all__ = ["AccelerationResidual", "VelocityResidual"]
