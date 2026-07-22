"""Pose, position, and orientation residuals over robot state nodes."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping
from numbers import Real

import torch

from ..data_model.data import Data
from ..data_model.model import Model
from ..kinematics.jacobian import get_frame_jacobian
from ..lie import se3, so3
from ..lie.tangents import right_jacobian_inv_se3, right_jacobian_inv_so3
from ._temporal_jacobian import dense_temporal_residual
from .utils import (
    RobotVariableLike as _RobotVariable,
    VariableLike as _Variable,
    current_value,
    matches,
    static_value,
)
from .base import Residual, Weight
from .nodes import RobotState, robot_state
from .structure import TemporalPattern


def _get_frame_pose(model: Model, data: Data, frame_id: int) -> torch.Tensor:
    if data.frame_pose_world is not None:
        return data.frame_pose_world[..., frame_id, :]
    parent_joint = model.structure.frame_parent_joint_ids[frame_id]
    parent = data.joint_pose_world[..., parent_joint, :]
    placement = model.values.frame_placements[..., frame_id, :]
    return se3.compose(parent, placement)


def _resolve_frame_id(model: Model, frame: str | None, frame_id: int | None) -> int:
    if frame is not None and frame_id is not None:
        raise TypeError("provide either frame=... or frame_id=..., not both")
    if frame is not None:
        if not isinstance(frame, str) or not frame:
            raise TypeError("frame must be a non-empty string")
        return model.frame_id(frame)
    if isinstance(frame_id, bool) or not isinstance(frame_id, int):
        raise TypeError("frame_id must be an int when frame is omitted")
    if not 0 <= frame_id < model.nframes:
        raise ValueError(f"frame_id must be in [0, {model.nframes}), got {frame_id}")
    return frame_id


def _normalize_knot(q: _RobotVariable, knot: int | None) -> int | None:
    if q.time_axis is None:
        if knot is not None:
            raise ValueError("knot requires q.time_axis=0")
        return None
    if q.time_axis != 0:
        raise ValueError(f"q time_axis must be 0 or None, got {q.time_axis}")
    if knot is None:
        return None
    if isinstance(knot, bool) or not isinstance(knot, int):
        raise TypeError(f"knot must be an int or None, got {type(knot).__name__}")
    if not -q.time_length <= knot < q.time_length:
        raise ValueError(f"knot={knot} must index trajectory length {q.time_length}")
    return knot % q.time_length


class _KinematicResidual(Residual):
    def __init__(
        self,
        q_or_state: _RobotVariable | RobotState,
        *,
        frame: str | None,
        frame_id: int | None,
        target: torch.Tensor | _Variable,
        knot: int | None,
        dim: int,
        weight: Real | torch.Tensor,
        row_weight: Weight | Real | torch.Tensor,
        kernel: object | None,
        name: str,
    ) -> None:
        q, state = robot_state(q_or_state)
        initial_target, targets = static_value(target, name="target")
        self.q = q
        self.state = state
        self.nodes = (state,)
        self.model = q.model
        self.frame_id = _resolve_frame_id(q.model, frame, frame_id)
        self.target = target
        self.knot = _normalize_knot(q, knot)
        self._knot_dim = dim
        self._validate_target_shape(initial_target)
        super().__init__(
            q,
            *targets,
            dim=dim,
            weight=weight,
            row_weight=row_weight,
            kernel=kernel,
            name=name,
        )

    @abstractmethod
    def _validate_target_shape(self, target: torch.Tensor) -> None:
        """Validate one concrete pose target shape."""

    def _require_knot(self, knot: int | None) -> int | None:
        if self.q.time_axis == 0 and knot is None:
            raise ValueError(f"{type(self).__name__} over a trajectory requires knot=...")
        return knot

    def _frame_pose_at(self, knot: int | None) -> torch.Tensor:
        pose = _get_frame_pose(self.model, self.state._checked_value(Data), self.frame_id)
        return pose if knot is None else pose[..., knot, :]

    def _frame_jacobian_at(self, knot: int | None) -> torch.Tensor:
        jacobian = get_frame_jacobian(self.model, self.state._checked_value(Data), self.frame_id)
        return jacobian if knot is None else jacobian[..., knot, :, :]

    def _target_tensor(self, exemplar: torch.Tensor) -> torch.Tensor:
        return current_value(self.target, exemplar, name="target")

    @abstractmethod
    def _error_at(self, knot: int | None) -> torch.Tensor:
        """Evaluate one concrete kinematic error."""

    def error(self) -> torch.Tensor:
        return self._error_at(self._require_knot(self.knot))

    @abstractmethod
    def _full_jacobian_at(self, knot: int | None) -> torch.Tensor:
        """Evaluate one concrete full-coordinate Jacobian."""

    def _tangent_jacobian_at(self, knot: int) -> torch.Tensor:
        return self._full_jacobian_at(knot)

    def temporal_structure(self, variable: _RobotVariable | str) -> TemporalPattern | None:
        if self.q.time_axis != 0 or self.knot is None or not matches(variable, self.q):
            return None
        return TemporalPattern(1, self.dim, self.knot, (0,))

    def temporal_jacobian_blocks(
        self,
        variable: _RobotVariable | str,
    ) -> Mapping[int, torch.Tensor]:
        pattern = self.temporal_structure(variable)
        if pattern is None:
            return {}
        return {0: self._tangent_jacobian_at(self.knot).unsqueeze(-3)}

    def jacobian(self) -> tuple[torch.Tensor, ...]:
        if self.q.time_axis is None:
            return (self._full_jacobian_at(None),)
        self._require_knot(self.knot)
        return dense_temporal_residual(self, self.q, self.q.time_length)


class PoseResidual(_KinematicResidual):
    """Six-dimensional frame pose error in the target frame."""

    def __init__(
        self,
        q_or_state: _RobotVariable | RobotState,
        *,
        frame: str | None = None,
        frame_id: int | None = None,
        target: torch.Tensor | _Variable,
        knot: int | None = None,
        pos_weight: Real = 1.0,
        ori_weight: Real = 1.0,
        weight: Real | torch.Tensor = 1.0,
        row_weight: Weight | Real | torch.Tensor = 1.0,
        kernel: object | None = None,
        name: str = "pose",
    ) -> None:
        self.pos_weight = float(pos_weight)
        self.ori_weight = float(ori_weight)
        super().__init__(
            q_or_state,
            frame=frame,
            frame_id=frame_id,
            target=target,
            knot=knot,
            dim=6,
            weight=weight,
            row_weight=row_weight,
            kernel=kernel,
            name=name,
        )

    def _validate_target_shape(self, target: torch.Tensor) -> None:
        if not target.shape or target.shape[-1] != 7:
            raise ValueError(f"target must end in pose width 7, got {tuple(target.shape)}")

    def _error_at(self, knot: int | None) -> torch.Tensor:
        end_effector = self._frame_pose_at(knot)
        target = self._target_tensor(end_effector)
        error = se3.log(se3.compose(se3.inverse(target), end_effector))
        return torch.cat(
            (error[..., :3] * self.pos_weight, error[..., 3:] * self.ori_weight),
            dim=-1,
        )

    def _full_jacobian_at(self, knot: int | None) -> torch.Tensor:
        end_effector = self._frame_pose_at(knot)
        target = self._target_tensor(end_effector)
        error = se3.log(se3.compose(se3.inverse(target), end_effector))
        world = self._frame_jacobian_at(knot)
        rotation = so3.to_matrix(end_effector[..., 3:])
        local = torch.cat(
            (rotation.mT @ world[..., :3, :], rotation.mT @ world[..., 3:, :]),
            dim=-2,
        )
        jacobian = right_jacobian_inv_se3(error) @ local
        return torch.cat(
            (jacobian[..., :3, :] * self.pos_weight, jacobian[..., 3:, :] * self.ori_weight),
            dim=-2,
        )


class PositionResidual(_KinematicResidual):
    """Three-dimensional frame position error."""

    def __init__(
        self,
        q_or_state: _RobotVariable | RobotState,
        *,
        frame: str | None = None,
        frame_id: int | None = None,
        target: torch.Tensor | _Variable,
        knot: int | None = None,
        weight: Real | torch.Tensor = 1.0,
        row_weight: Weight | Real | torch.Tensor = 1.0,
        kernel: object | None = None,
        name: str = "position",
    ) -> None:
        super().__init__(
            q_or_state,
            frame=frame,
            frame_id=frame_id,
            target=target,
            knot=knot,
            dim=3,
            weight=weight,
            row_weight=row_weight,
            kernel=kernel,
            name=name,
        )

    def _validate_target_shape(self, target: torch.Tensor) -> None:
        if not target.shape or target.shape[-1] not in (3, 7):
            raise ValueError(f"target must end in position width 3 or pose width 7, got {tuple(target.shape)}")

    def _position_target(self, exemplar: torch.Tensor) -> torch.Tensor:
        target = self._target_tensor(exemplar)
        return target[..., :3]

    def _error_at(self, knot: int | None) -> torch.Tensor:
        end_effector = self._frame_pose_at(knot)
        return end_effector[..., :3] - self._position_target(end_effector)

    def _full_jacobian_at(self, knot: int | None) -> torch.Tensor:
        return self._frame_jacobian_at(knot)[..., :3, :]


class OrientationResidual(_KinematicResidual):
    """Three-dimensional logarithmic frame orientation error."""

    def __init__(
        self,
        q_or_state: _RobotVariable | RobotState,
        *,
        frame: str | None = None,
        frame_id: int | None = None,
        target: torch.Tensor | _Variable,
        knot: int | None = None,
        weight: Real | torch.Tensor = 1.0,
        row_weight: Weight | Real | torch.Tensor = 1.0,
        kernel: object | None = None,
        name: str = "orientation",
    ) -> None:
        super().__init__(
            q_or_state,
            frame=frame,
            frame_id=frame_id,
            target=target,
            knot=knot,
            dim=3,
            weight=weight,
            row_weight=row_weight,
            kernel=kernel,
            name=name,
        )

    def _validate_target_shape(self, target: torch.Tensor) -> None:
        if not target.shape or target.shape[-1] not in (4, 7):
            raise ValueError(f"target must end in quaternion width 4 or pose width 7, got {tuple(target.shape)}")

    def _orientation_target(self, exemplar: torch.Tensor) -> torch.Tensor:
        target = self._target_tensor(exemplar)
        return target if target.shape[-1] == 4 else target[..., 3:]

    def _error_at(self, knot: int | None) -> torch.Tensor:
        end_effector = self._frame_pose_at(knot)
        target = self._orientation_target(end_effector)
        return so3.log(so3.compose(so3.inverse(target), end_effector[..., 3:]))

    def _full_jacobian_at(self, knot: int | None) -> torch.Tensor:
        end_effector = self._frame_pose_at(knot)
        target = self._orientation_target(end_effector)
        error = so3.log(so3.compose(so3.inverse(target), end_effector[..., 3:]))
        angular = self._frame_jacobian_at(knot)[..., 3:, :]
        rotation = so3.to_matrix(end_effector[..., 3:])
        return right_jacobian_inv_so3(error) @ (rotation.mT @ angular)


__all__ = ["OrientationResidual", "PoseResidual", "PositionResidual"]
