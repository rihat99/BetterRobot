"""Time-local adapter for object-referenced trajectory residuals."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import torch

from ._temporal_jacobian import dense_temporal_residual
from .utils import RobotVariableLike as _TemporalVariable, TemporalLike, matches
from .base import Residual
from .structure import TemporalPattern


class TimeIndexedResidual(Residual):
    """Evaluate a time-local residual at one knot of its trajectory variable.

    The wrapped residual must expose the private knot primitives implemented by
    the built-in pose and limit residuals. New user residuals should generally
    accept ``knot=`` directly instead of using this adapter.
    """

    def __init__(
        self,
        inner: Residual,
        t_idx: int,
        *,
        name: str | None = None,
    ) -> None:
        if not isinstance(inner, Residual):
            raise TypeError(f"inner must be a Residual, got {type(inner).__name__}")
        temporal = tuple(variable for variable in inner.variables if getattr(variable, "time_axis", None) == 0)
        if len(temporal) != 1 or not isinstance(temporal[0], TemporalLike):
            raise ValueError("inner must reference exactly one Variable with time_axis=0")
        if not callable(getattr(inner, "_error_at", None)) or not callable(
            getattr(inner, "_tangent_jacobian_at", None)
        ):
            raise TypeError("inner must support built-in knot error and Jacobian primitives")
        q = cast(_TemporalVariable, temporal[0])
        if isinstance(t_idx, bool) or not isinstance(t_idx, int):
            raise TypeError(f"t_idx must be an int, got {type(t_idx).__name__}")
        if not -q.time_length <= t_idx < q.time_length:
            raise ValueError(f"t_idx={t_idx} must index trajectory length {q.time_length}")
        self.inner = inner
        self.q = q
        self.t_idx = t_idx % q.time_length
        self.horizon = q.time_length
        self.nodes = tuple(getattr(inner, "nodes", ()))
        dim = getattr(inner, "_knot_dim", inner.dim)
        super().__init__(
            *inner.variables,
            dim=dim,
            weight=inner.weight,
            row_weight=inner.row_weight,
            reduce=inner.reduce,
            kernel=inner.kernel,
            group_size=inner.group_size,
            name=name or f"{inner.name}_t{self.t_idx}",
            enabled=inner.enabled,
        )

    def error(self) -> torch.Tensor:
        output = self.inner._error_at(self.t_idx)
        if not isinstance(output, torch.Tensor) or output.shape[-1] != self.dim:
            actual = tuple(output.shape) if isinstance(output, torch.Tensor) else type(output).__name__
            raise ValueError(f"inner knot error must end in {self.dim}, got {actual}")
        return output

    def temporal_structure(self, variable: _TemporalVariable | str) -> TemporalPattern | None:
        if not matches(variable, self.q):
            return None
        return TemporalPattern(1, self.dim, self.t_idx, (0,))

    def temporal_jacobian_blocks(
        self,
        variable: _TemporalVariable | str,
    ) -> Mapping[int, torch.Tensor]:
        if not matches(variable, self.q):
            return {}
        block = self.inner._tangent_jacobian_at(self.t_idx)
        return {0: block.unsqueeze(-3)}

    def jacobian(self) -> tuple[torch.Tensor, ...]:
        return dense_temporal_residual(self, self.q, self.horizon)


__all__ = ["TimeIndexedResidual"]
