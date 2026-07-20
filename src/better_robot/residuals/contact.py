"""Contact-consistency residual for trajectory robot variables."""

from __future__ import annotations

from collections.abc import Mapping
import math
from numbers import Real

import torch

from .._validation import check_tensor
from ..data_model.data import Data
from ._temporal_jacobian import dense_temporal_jacobian
from .utils import RobotVariableLike as _RobotVariableLike, VariableLike as _VariableLike, matches, value
from .base import Residual, Weight
from .nodes import RobotState, robot_state
from .structure import TemporalPattern


class ContactConsistencyResidual(Residual):
    """Cartesian velocity penalty on tagged contact frames.

    ``contact_weights`` are domain mask amplitudes averaged across each pair
    of adjacent knots. The inherited ``weight`` is the optimizer-level row
    multiplier and is therefore not duplicated inside :meth:`error` or
    :meth:`jacobian`.
    """

    def __init__(
        self,
        q_or_state: _RobotVariableLike | RobotState,
        frame_ids: tuple[int, ...],
        contact_weights: _VariableLike | torch.Tensor,
        *,
        dt: float,
        weight: Weight | Real | torch.Tensor = 1.0,
        kernel: object | None = None,
        name: str = "contact_consistency",
    ) -> None:
        q, state = robot_state(q_or_state)
        if q.time_axis != 0 or len(q.shape) != 2:
            raise ValueError("ContactConsistencyResidual q must declare time_axis=0")
        if not frame_ids:
            raise ValueError("ContactConsistencyResidual requires at least one contact frame")
        ids = tuple(frame_ids)
        if any(isinstance(frame_id, bool) or not isinstance(frame_id, int) for frame_id in ids):
            raise TypeError("frame_ids must contain only integer frame ids")
        if any(frame_id < 0 or frame_id >= q.model.nframes for frame_id in ids):
            raise ValueError(f"frame_ids must index model frame rows in [0, {q.model.nframes})")
        weights = check_tensor("contact_weights", value(contact_weights, "contact_weights"), floating=True)
        if weights.ndim != 2 or weights.shape[1] != len(ids):
            raise ValueError(f"contact_weights must be (T, {len(ids)}); got {tuple(weights.shape)}")
        if weights.shape[0] < 2:
            raise ValueError("ContactConsistencyResidual requires at least two timesteps")
        if q.shape[0] != weights.shape[0]:
            raise ValueError(f"q horizon {q.shape[0]} != contact_weights length {weights.shape[0]}")
        dt = float(dt)
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError(f"dt must be finite and positive, got {dt!r}")

        self.state = state
        self.nodes = (state,)
        self.q = q
        self.model = q.model
        self.frame_ids = ids
        self.contact_weights = contact_weights
        self.dt = dt
        self.horizon = int(weights.shape[0])
        direct = (contact_weights,) if isinstance(contact_weights, _VariableLike) else ()
        super().__init__(
            *direct,
            dim=3 * len(ids) * (self.horizon - 1),
            weight=weight,
            kernel=kernel,
            group_size=3,
            name=name,
        )

    def _frame_positions(self, data: Data) -> torch.Tensor:
        if data.frame_pose_world is None:
            raise RuntimeError("ContactConsistencyResidual requires frame placements")
        frame_idx = torch.as_tensor(self.frame_ids, device=data.frame_pose_world.device)
        return data.frame_pose_world[..., :3].index_select(-2, frame_idx)

    def _domain_weights(self, exemplar: torch.Tensor) -> torch.Tensor:
        return check_tensor(
            "contact_weights",
            value(self.contact_weights, "contact_weights"),
            shape=(self.horizon, len(self.frame_ids)),
            floating=True,
            dtype=exemplar.dtype,
            device=exemplar.device,
        )

    def error(self) -> torch.Tensor:
        q = self.q.tensor
        if q.ndim < 2 or q.shape[-2] != self.horizon:
            raise ValueError(f"trajectory must end in ({self.horizon}, nq), got {tuple(q.shape)}")
        positions = self._frame_positions(self.state._checked_value(Data))
        displacement = (positions[..., 1:, :, :] - positions[..., :-1, :, :]) / self.dt
        weights = self._domain_weights(q)
        pair_weights = 0.5 * (weights[:-1] + weights[1:]).unsqueeze(-1)
        rows = pair_weights * displacement
        return rows.reshape(*q.shape[:-2], self.dim)

    def temporal_structure(self, variable: _RobotVariableLike | str) -> TemporalPattern | None:
        if not matches(variable, self.q):
            return None
        return TemporalPattern(
            rows=self.horizon - 1,
            row_width=3 * len(self.frame_ids),
            row_origin=0,
            offsets=(0, 1),
        )

    def _frame_jacobians(self, data: Data) -> torch.Tensor:
        from ..kinematics.jacobian import get_frame_jacobian  # noqa: PLC0415

        return torch.stack(
            [get_frame_jacobian(self.model, data, frame_id)[..., :3, :] for frame_id in self.frame_ids],
            dim=-3,
        )

    def _temporal_blocks(self, indices: torch.Tensor) -> dict[int, torch.Tensor]:
        q = self.q.tensor
        contacts = len(self.frame_ids)
        frame_jacobians = self._frame_jacobians(self.state._checked_value(Data)).index_select(-1, indices)
        pair_weights = 0.5 * (self._domain_weights(q)[:-1] + self._domain_weights(q)[1:])
        scale_shape = (*((1,) * len(q.shape[:-2])), self.horizon - 1, contacts, 1, 1)
        scales = (pair_weights / self.dt).reshape(scale_shape)
        left = -scales * frame_jacobians[..., :-1, :, :, :]
        right = scales * frame_jacobians[..., 1:, :, :, :]
        shape = (*q.shape[:-2], self.horizon - 1, 3 * contacts, indices.numel())
        return {0: left.reshape(shape), 1: right.reshape(shape)}

    def temporal_jacobian_blocks(
        self,
        variable: _RobotVariableLike | str,
    ) -> Mapping[int, torch.Tensor]:
        if not matches(variable, self.q):
            return {}
        indices = torch.arange(self.model.nv, device=self.q.tensor.device)
        return self._temporal_blocks(indices)

    def jacobian(self) -> tuple[torch.Tensor, ...] | None:
        direct_trainables = tuple(variable for variable in self.variables if variable.trainable)
        if direct_trainables:
            return None
        if not self.q.trainable:
            return ()
        pattern = self.temporal_structure(self.q)
        assert pattern is not None
        full_indices = torch.arange(self.model.nv, device=self.q.tensor.device)
        full = dense_temporal_jacobian(
            pattern,
            self._temporal_blocks(full_indices),
            horizon=self.horizon,
        )
        return (full,)


__all__ = ["ContactConsistencyResidual"]
