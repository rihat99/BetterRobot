"""Manifold-aware smoothness residuals over trajectory variables."""

from __future__ import annotations

from collections.abc import Mapping
import math
from numbers import Real

import torch

from .._validation import check_tensor
from ._temporal_jacobian import dense_temporal_residual
from .utils import RobotLike, RobotVariableLike as _RobotTrajectory, matches
from .base import DiagonalWeight, Residual, Weight
from .structure import TemporalPattern


def _trajectory(variable: _RobotTrajectory, name: str) -> tuple[torch.Tensor, int]:
    if not isinstance(variable, RobotLike):
        raise TypeError(f"{name} q must be a RobotVariable, got {type(variable).__name__}")
    if variable.time_axis != 0 or len(variable.shape) != 2:
        raise ValueError(f"{name} q must declare time_axis=0 with event shape (T, nq), got {variable.shape}")
    return variable.tensor, variable.time_length


def _validate_dt(dt: Real, name: str) -> float:
    if isinstance(dt, bool) or not isinstance(dt, Real) or not math.isfinite(float(dt)) or float(dt) <= 0.0:
        raise ValueError(f"{name} dt must be finite and positive, got {dt!r}")
    return float(dt)


def _anchor_constant_block(block: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    anchor = q.sum(dim=(-2, -1)) * 0.0
    return block + anchor[..., None, None, None]


class _RepeatedCoordinateWeight(Weight):
    """Lazily tile a live per-coordinate row weight over temporal rows."""

    def __init__(self, coordinate: torch.Tensor, rows: int) -> None:
        self.coordinate = coordinate
        self.rows = rows

    def _weight(self) -> DiagonalWeight:
        return DiagonalWeight(self.coordinate.repeat(self.rows))

    def apply(self, error: torch.Tensor) -> torch.Tensor:
        return self._weight().apply(error)

    def apply_jacobian(self, blocks: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        return self._weight().apply_jacobian(blocks)


class _SmoothnessResidual(Residual):
    def __init__(
        self,
        q: _RobotTrajectory,
        *,
        offsets: tuple[int, ...],
        coefficients: tuple[float, ...],
        derivative_order: int,
        scale_divisor: float,
        dt: Real,
        coordinate_weight: torch.Tensor | None,
        weight: Real | torch.Tensor,
        row_weight: Weight | Real | torch.Tensor,
        kernel: object | None,
        name: str,
    ) -> None:
        _value, horizon = _trajectory(q, type(self).__name__)
        span = offsets[-1] - offsets[0]
        if horizon <= span:
            raise ValueError(f"{type(self).__name__} needs at least {span + 1} timesteps, got T={horizon}")
        self.q, self.model = q, q.model
        self.dt = _validate_dt(dt, type(self).__name__)
        self.horizon, self.rows = horizon, horizon - span
        self._offsets, self._coefficients = offsets, coefficients
        self._row_origin = -offsets[0]
        self._scale = 1.0 / (scale_divisor * self.dt**derivative_order)
        pending = list(self.model.joint_models)
        non_affine_kinds: set[str] = set()
        while pending:
            joint = pending.pop()
            if joint.kind == "composite":
                pending.extend(joint.sub_joints)
            elif joint.nv > 1:
                non_affine_kinds.add(joint.kind)
        non_affine = tuple(sorted(non_affine_kinds))
        if non_affine:
            self._autodiff_fallback_reason = (
                f"model {self.model.name!r} contains unsupported non-scalar joint kinds {non_affine}"
            )
            self.temporal_jacobian_blocks = None  # type: ignore[method-assign]
        if coordinate_weight is not None:
            coordinate_weight = check_tensor(
                "coordinate_weight",
                coordinate_weight,
                shape=(self.model.nv,),
                floating=True,
                dtype=q.tensor.dtype,
                device=q.tensor.device,
            )
            if coordinate_weight.ndim != 1:
                raise ValueError(
                    f"coordinate_weight must have exact shape ({self.model.nv},), got {tuple(coordinate_weight.shape)}"
                )
            row_weight = _RepeatedCoordinateWeight(coordinate_weight, self.rows)
        super().__init__(
            q,
            dim=self.rows * q.model.nv,
            weight=weight,
            row_weight=row_weight,
            kernel=kernel,
            name=name,
        )

    def temporal_structure(self, variable: _RobotTrajectory | str) -> TemporalPattern | None:
        return (
            TemporalPattern(self.rows, self.model.nv, self._row_origin, self._offsets)
            if matches(variable, self.q)
            else None
        )

    def temporal_jacobian_blocks(self, variable: _RobotTrajectory | str) -> Mapping[int, torch.Tensor]:
        if not matches(variable, self.q):
            return {}
        q, _horizon = _trajectory(self.q, type(self).__name__)
        identity = torch.eye(self.model.nv, dtype=q.dtype, device=q.device)
        base = identity.expand(*q.shape[:-2], self.rows, self.model.nv, self.model.nv)
        return {
            offset: _anchor_constant_block(coefficient * self._scale * base, q)
            for offset, coefficient in zip(self._offsets, self._coefficients, strict=True)
        }

    def jacobian(self) -> tuple[torch.Tensor, ...] | None:
        if hasattr(self, "_autodiff_fallback_reason"):
            return None
        return dense_temporal_residual(self, self.q, self.horizon)


class VelocityResidual(_SmoothnessResidual):
    """Three-point central-difference velocity in robot tangent space.

    Non-scalar joint kinds warn and use dense autodiff instead of constant blocks.
    """

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
        super().__init__(
            q,
            offsets=(-1, 1),
            coefficients=(-1.0, 1.0),
            derivative_order=1,
            scale_divisor=2.0,
            dt=dt,
            coordinate_weight=None,
            weight=weight,
            row_weight=row_weight,
            kernel=kernel,
            name=name,
        )

    def error(self) -> torch.Tensor:
        q, _horizon = _trajectory(self.q, type(self).__name__)
        velocity = self.model.difference(q[..., :-2, :], q[..., 2:, :]) / (2.0 * self.dt)
        return velocity.reshape(*q.shape[:-2], self.dim)


class SmoothnessResidual(_SmoothnessResidual):
    """Construct an order 2--4 forward tangent difference scaled by ``dt**order``.

    The trajectory must have ``T > order``. ``coordinate_weight`` has exact
    shape ``(nv,)`` and supplies square-root-information multipliers repeated
    across the ``T - order`` rows; values multiply rows directly, so callers do
    not take another square root. Non-scalar joint kinds disable analytic and
    direct-banded blocks; automatic linearization warns before dense autodiff.
    """

    def __init__(
        self,
        q: _RobotTrajectory,
        *,
        order: int,
        dt: Real,
        coordinate_weight: torch.Tensor | None = None,
        weight: Real | torch.Tensor = 1.0,
        kernel: object | None = None,
        name: str = "smoothness",
    ) -> None:
        if isinstance(order, bool) or not isinstance(order, int) or order not in (2, 3, 4):
            raise ValueError(f"order must be one of (2, 3, 4), got {order!r}")
        self.order = order
        offsets = tuple(range(order + 1))
        coefficients = tuple(float((-1) ** (order - index) * math.comb(order, index)) for index in offsets)
        super().__init__(
            q,
            offsets=offsets,
            coefficients=coefficients,
            derivative_order=order,
            scale_divisor=1.0,
            dt=dt,
            coordinate_weight=coordinate_weight,
            weight=weight,
            row_weight=1.0,
            kernel=kernel,
            name=name,
        )

    def error(self) -> torch.Tensor:
        q, _horizon = _trajectory(self.q, type(self).__name__)
        differences = self.model.difference(q[..., :-1, :], q[..., 1:, :])
        for _ in range(self.order - 1):
            differences = differences[..., 1:, :] - differences[..., :-1, :]
        return (differences / self.dt**self.order).reshape(*q.shape[:-2], self.dim)


__all__ = ["SmoothnessResidual", "VelocityResidual"]
