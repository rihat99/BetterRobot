"""Robot configuration and trajectory regularization residuals."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Real

import torch

from .._validation import check_tensor
from ._temporal_jacobian import dense_temporal_residual
from .utils import (
    RobotVariableLike as _RobotVariable,
    VariableLike as _Variable,
    current_value,
    matches,
    require_robot,
    static_value,
)
from .base import Residual, Weight
from .structure import TemporalPattern


class RestResidual(Residual):
    """Tangent rest displacement; a bare ``q_rest`` is a construction-time constant."""

    def __init__(
        self,
        q: _RobotVariable,
        q_rest: torch.Tensor | _Variable,
        *,
        weight: Real | torch.Tensor = 1.0,
        row_weight: Weight | Real | torch.Tensor = 1.0,
        kernel: object | None = None,
        name: str = "rest",
    ) -> None:
        require_robot(q, type(self).__name__, temporal=False)
        initial, targets = static_value(q_rest, name="q_rest")
        if tuple(initial.shape[-1:]) != (q.model.nq,):
            raise ValueError(f"q_rest must end in ({q.model.nq},), got {tuple(initial.shape)}")
        self.q = q
        self.model = q.model
        self.q_rest = q_rest
        super().__init__(
            q,
            *targets,
            dim=q.model.nv,
            weight=weight,
            row_weight=row_weight,
            kernel=kernel,
            name=name,
        )

    def error(self) -> torch.Tensor:
        q = self.q.tensor
        q_rest = current_value(self.q_rest, q, name="q_rest")
        return self.model.difference(q_rest, q)

    def jacobian(self) -> tuple[torch.Tensor, ...]:
        q = self.q.tensor
        identity = torch.eye(self.model.nv, dtype=q.dtype, device=q.device)
        return (identity.expand(*q.shape[:-1], *identity.shape),)


class JointRotationPrior(Residual):
    """Per-joint tangent prior; bare ``q_mean`` and weights are construction-time constants."""

    def __init__(
        self,
        q: _RobotVariable,
        q_mean: torch.Tensor | _Variable,
        per_joint_weight: torch.Tensor,
        *,
        weight: Real | torch.Tensor = 1.0,
        row_weight: Weight | Real | torch.Tensor = 1.0,
        kernel: object | None = None,
        name: str = "joint_rotation_prior",
    ) -> None:
        require_robot(q, type(self).__name__, temporal=False)
        mean, targets = static_value(q_mean, name="q_mean")
        if tuple(mean.shape) != (q.model.nq,):
            raise ValueError(f"q_mean must have shape ({q.model.nq},), got {tuple(mean.shape)}")
        per_joint_weight = check_tensor("per_joint_weight", per_joint_weight, floating=True)
        if tuple(per_joint_weight.shape) == (q.model.njoints,):
            repeats = torch.tensor(q.model.nvs, dtype=torch.long, device=per_joint_weight.device)
            tangent_weight = torch.repeat_interleave(per_joint_weight, repeats)
        elif tuple(per_joint_weight.shape) == (q.model.nv,):
            tangent_weight = per_joint_weight
        else:
            raise ValueError(
                f"per_joint_weight must have shape ({q.model.njoints},) or ({q.model.nv},), "
                f"got {tuple(per_joint_weight.shape)}"
            )
        if bool(torch.any(tangent_weight < 0.0)):
            raise ValueError("per_joint_weight must be non-negative")
        self.q = q
        self.model = q.model
        self.q_mean = q_mean
        self.per_joint_weight = tangent_weight
        super().__init__(
            q,
            *targets,
            dim=q.model.nv,
            weight=weight,
            row_weight=row_weight,
            kernel=kernel,
            name=name,
        )

    def error(self) -> torch.Tensor:
        q = self.q.tensor
        q_mean = current_value(self.q_mean, q, name="q_mean")
        if self.per_joint_weight.dtype != q.dtype or self.per_joint_weight.device != q.device:
            raise ValueError("per_joint_weight must share q dtype/device")
        return self.model.difference(q_mean, q) * self.per_joint_weight


class ReferenceTrajectoryResidual(Residual):
    """Tangent trajectory error; bare reference and frame weights are construction constants."""

    def __init__(
        self,
        q: _RobotVariable,
        q_ref: torch.Tensor | _Variable,
        *,
        weight: Real | torch.Tensor = 1.0,
        row_weight: Weight | Real | torch.Tensor = 1.0,
        weight_per_frame: torch.Tensor | None = None,
        kernel: object | None = None,
        name: str = "reference_trajectory",
    ) -> None:
        require_robot(q, type(self).__name__, temporal=True)
        reference, targets = static_value(q_ref, name="q_ref")
        if tuple(reference.shape[-2:]) != q.shape:
            raise ValueError(f"q_ref must end in trajectory event shape {q.shape}, got {tuple(reference.shape)}")
        if weight_per_frame is not None:
            weight_per_frame = check_tensor("weight_per_frame", weight_per_frame, floating=True)
            if tuple(weight_per_frame.shape) != (q.time_length,):
                raise ValueError(
                    f"weight_per_frame must have shape ({q.time_length},), got {tuple(weight_per_frame.shape)}"
                )
        self.q = q
        self.model = q.model
        self.q_ref = q_ref
        self.weight_per_frame = weight_per_frame
        self.horizon = q.time_length
        super().__init__(
            q,
            *targets,
            dim=self.horizon * q.model.nv,
            weight=weight,
            row_weight=row_weight,
            kernel=kernel,
            name=name,
        )

    def _per_frame_scale(self, q: torch.Tensor) -> torch.Tensor:
        if self.weight_per_frame is None:
            return q.new_ones(self.horizon)
        if self.weight_per_frame.dtype != q.dtype or self.weight_per_frame.device != q.device:
            raise ValueError("weight_per_frame must share q dtype/device")
        return self.weight_per_frame

    def error(self) -> torch.Tensor:
        q = self.q.tensor
        reference = current_value(self.q_ref, q, name="q_ref")
        difference = self.model.difference(reference, q)
        scaled = difference * self._per_frame_scale(q).unsqueeze(-1)
        return scaled.reshape(*q.shape[:-2], self.dim)

    def temporal_structure(self, variable: _RobotVariable | str) -> TemporalPattern | None:
        if not matches(variable, self.q):
            return None
        return TemporalPattern(self.horizon, self.model.nv, 0, (0,))

    def temporal_jacobian_blocks(
        self,
        variable: _RobotVariable | str,
    ) -> Mapping[int, torch.Tensor]:
        if not matches(variable, self.q):
            return {}
        q = self.q.tensor
        identity = torch.eye(self.model.nv, dtype=q.dtype, device=q.device)
        base = identity.expand(*q.shape[:-2], self.horizon, self.model.nv, self.model.nv)
        block = base * self._per_frame_scale(q)[..., None, None]
        anchor = q.sum(dim=(-2, -1)) * 0.0
        return {0: block + anchor[..., None, None, None]}

    def jacobian(self) -> tuple[torch.Tensor, ...]:
        return dense_temporal_residual(self, self.q, self.horizon)


__all__ = ["JointRotationPrior", "ReferenceTrajectoryResidual", "RestResidual"]
