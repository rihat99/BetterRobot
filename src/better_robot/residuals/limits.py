"""Joint position and velocity limit residuals."""

from __future__ import annotations

from collections.abc import Mapping
import math
from numbers import Real

import torch

from ._temporal_jacobian import dense_temporal_residual
from .utils import (
    RobotLike,
    RobotVariableLike as _RobotVariable,
    VariableLike as _Variable,
    current_value,
    matches,
    static_value,
)
from .base import Residual, Weight
from .structure import TemporalPattern


class JointPositionLimit(Residual):
    """One-sided clamped penalty on a robot variable's position limits."""

    def __init__(
        self,
        q: _RobotVariable,
        *,
        knot: int | None = None,
        weight: Weight | Real | torch.Tensor = 1.0,
        kernel: object | None = None,
        name: str = "joint_position_limit",
    ) -> None:
        if not isinstance(q, RobotLike):
            raise TypeError(f"q must be a RobotVariable, got {type(q).__name__}")
        if q.time_axis is None:
            if knot is not None:
                raise ValueError("knot requires q.time_axis=0")
            horizon = None
        else:
            if q.time_axis != 0:
                raise ValueError(f"q time_axis must be 0 or None, got {q.time_axis}")
            horizon = q.time_length
            if knot is not None:
                if isinstance(knot, bool) or not isinstance(knot, int):
                    raise TypeError(f"knot must be an int or None, got {type(knot).__name__}")
                if not -horizon <= knot < horizon:
                    raise ValueError(f"knot={knot} must index trajectory length {horizon}")
                knot %= horizon
        self.q = q
        self.model = q.model
        self.horizon = horizon
        self.knot = knot
        self._knot_dim = 2 * q.model.nq
        dim = self._knot_dim if horizon is None or knot is not None else horizon * self._knot_dim
        super().__init__(q, dim=dim, weight=weight, kernel=kernel, name=name)

    def _configuration_at(self, knot: int | None) -> torch.Tensor:
        q = self.q.tensor
        return q if knot is None else q[..., knot, :]

    def _limits(self, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        lower, upper = self.model.lower_pos_limit, self.model.upper_pos_limit
        if lower.dtype != q.dtype or lower.device != q.device:
            raise ValueError("model position limits must share q dtype/device")
        return lower, upper

    def _error_at(self, knot: int | None) -> torch.Tensor:
        q = self._configuration_at(knot)
        lower, upper = self._limits(q)
        rows = torch.cat((torch.clamp(lower - q, min=0.0), torch.clamp(q - upper, min=0.0)), dim=-1)
        if knot is None and self.horizon is not None:
            return rows.reshape(*q.shape[:-2], self.horizon * self._knot_dim)
        return rows

    def error(self) -> torch.Tensor:
        return self._error_at(self.knot)

    def _dq_dv(self, q: torch.Tensor) -> torch.Tensor:
        projection = q.new_zeros(self.model.nq, self.model.nv)
        for nq_joint, nv_joint, iq, iv in zip(
            self.model.nqs,
            self.model.nvs,
            self.model.idx_qs,
            self.model.idx_vs,
            strict=True,
        ):
            if nq_joint and nq_joint == nv_joint:
                projection[iq : iq + nq_joint, iv : iv + nv_joint].fill_diagonal_(1.0)
        return projection

    def _full_jacobian_at(self, knot: int | None) -> torch.Tensor:
        q = self._configuration_at(knot)
        lower, upper = self._limits(q)
        lower_diagonal = torch.where(q < lower, -torch.ones_like(q), torch.zeros_like(q))
        upper_diagonal = torch.where(q > upper, torch.ones_like(q), torch.zeros_like(q))
        projection = self._dq_dv(q)
        return torch.cat(
            (lower_diagonal.unsqueeze(-1) * projection, upper_diagonal.unsqueeze(-1) * projection),
            dim=-2,
        )

    def _tangent_jacobian_at(self, knot: int) -> torch.Tensor:
        return self._full_jacobian_at(knot)

    def temporal_structure(self, variable: _RobotVariable | str) -> TemporalPattern | None:
        if self.horizon is None or not matches(variable, self.q):
            return None
        if self.knot is None:
            return TemporalPattern(self.horizon, self._knot_dim, 0, (0,))
        return TemporalPattern(1, self._knot_dim, self.knot, (0,))

    def temporal_jacobian_blocks(
        self,
        variable: _RobotVariable | str,
    ) -> Mapping[int, torch.Tensor]:
        pattern = self.temporal_structure(variable)
        if pattern is None:
            return {}
        if self.knot is None:
            blocks = [self._tangent_jacobian_at(index) for index in range(self.horizon or 0)]
            block = torch.stack(blocks, dim=-3)
        else:
            block = self._tangent_jacobian_at(self.knot).unsqueeze(-3)
        return {0: block}

    def jacobian(self) -> tuple[torch.Tensor, ...]:
        if self.horizon is None:
            return (self._full_jacobian_at(None),)
        return dense_temporal_residual(self, self.q, self.horizon)


class JointVelocityLimit(Residual):
    """One-sided clamped penalty on a velocity-valued Variable."""

    def __init__(
        self,
        velocity: _Variable,
        limit: torch.Tensor | _Variable,
        *,
        weight: Weight | Real | torch.Tensor = 1.0,
        kernel: object | None = None,
        name: str = "joint_velocity_limit",
    ) -> None:
        if not isinstance(velocity, _Variable):
            raise TypeError(f"velocity must be a Variable, got {type(velocity).__name__}")
        limit_tensor, targets = static_value(limit, name="limit")
        if not velocity.shape or tuple(limit_tensor.shape) != velocity.shape[-1:]:
            raise ValueError(f"limit must have shape {velocity.shape[-1:]}, got {tuple(limit_tensor.shape)}")
        self.velocity = velocity
        self.limit = limit
        event_width = math.prod(velocity.shape)
        super().__init__(
            velocity,
            *targets,
            dim=2 * event_width,
            weight=weight,
            kernel=kernel,
            name=name,
        )

    def error(self) -> torch.Tensor:
        velocity = self.velocity.tensor
        limit = current_value(self.limit, velocity, name="limit", preserve=False)
        rows = torch.cat((torch.clamp(-limit - velocity, min=0.0), torch.clamp(velocity - limit, min=0.0)), dim=-1)
        return rows.reshape(*velocity.shape[: -len(self.velocity.shape)], self.dim)


__all__ = ["JointPositionLimit", "JointVelocityLimit"]
