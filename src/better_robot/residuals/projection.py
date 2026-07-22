"""Camera-thin pinhole projection residual over model frame-table rows."""

from __future__ import annotations

from collections.abc import Sequence
import math
from numbers import Real

import torch

from .._validation import check_tensor
from ..data_model.data import Data
from ..kinematics.jacobian import get_frame_jacobian
from .utils import RobotVariableLike as _RobotVariableLike, VariableLike as _VariableLike, value, variables
from .base import Residual, Weight
from .nodes import RobotState, robot_state


def _transform_points(points_world: torch.Tensor, extrinsics: torch.Tensor) -> torch.Tensor:
    rotation = extrinsics[..., :3, :3]
    translation = extrinsics[..., :3, 3]
    rotated = torch.matmul(rotation.unsqueeze(-3), points_world.unsqueeze(-1)).squeeze(-1)
    return rotated + translation.unsqueeze(-2)


def _project_points(
    points_camera: torch.Tensor,
    intrinsics: torch.Tensor,
    *,
    min_depth: float,
) -> torch.Tensor:
    homogeneous = torch.matmul(intrinsics.unsqueeze(-3), points_camera.unsqueeze(-1)).squeeze(-1)
    depth = points_camera[..., 2].clamp_min(min_depth)
    return homogeneous[..., :2] / depth.unsqueeze(-1)


def _projection_jacobian(
    points_camera: torch.Tensor,
    intrinsics: torch.Tensor,
    *,
    min_depth: float,
) -> torch.Tensor:
    homogeneous = torch.matmul(intrinsics.unsqueeze(-3), points_camera.unsqueeze(-1)).squeeze(-1)
    numerator = homogeneous[..., :2]
    depth_raw = points_camera[..., 2]
    depth = depth_raw.clamp_min(min_depth)
    numerator_jacobian = intrinsics[..., :2, :].unsqueeze(-3)
    depth_axis = torch.stack(
        (torch.zeros_like(depth_raw), torch.zeros_like(depth_raw), torch.ones_like(depth_raw)),
        dim=-1,
    )
    depth_active = (depth_raw >= min_depth).to(dtype=points_camera.dtype)
    denominator_term = (
        numerator.unsqueeze(-1)
        * depth_axis.unsqueeze(-2)
        * depth_active.unsqueeze(-1).unsqueeze(-1)
        / depth.square().unsqueeze(-1).unsqueeze(-1)
    )
    return numerator_jacobian / depth.unsqueeze(-1).unsqueeze(-1) - denominator_term


def _normalize_point_ids(point_ids: Sequence[int] | torch.Tensor) -> tuple[int, ...]:
    if isinstance(point_ids, torch.Tensor):
        if point_ids.ndim != 1 or point_ids.dtype not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }:
            raise TypeError("point_ids tensor must be one-dimensional with an integer dtype")
        raw_ids = point_ids.detach().cpu().tolist()  # bench-ok: one-time static metadata
    else:
        raw_ids = list(point_ids)
    if not raw_ids:
        raise ValueError("point_ids must contain at least one frame-table row")
    if any(isinstance(point_id, bool) or not isinstance(point_id, int) for point_id in raw_ids):
        raise TypeError("point_ids must contain integers")
    return tuple(raw_ids)


class ProjectionResidual(Residual):
    """Pinhole reprojection error for frame, marker, or site rows.

    Bare observations are construction-time constants; static Variables are
    updatable through ``Problem.update()``. ``weights`` and ``valid_mask`` are
    domain multipliers. The singular optimizer ``weight`` is an outer
    coefficient applied once by :class:`Residual`; robust kernels consume one
    two-row group per projected point.
    """

    def __init__(
        self,
        q_or_state: _RobotVariableLike | RobotState,
        point_ids: Sequence[int] | torch.Tensor,
        K: _VariableLike | torch.Tensor,
        extrinsics: _VariableLike | torch.Tensor,
        target_px: _VariableLike | torch.Tensor,
        *,
        weights: _VariableLike | torch.Tensor | None = None,
        valid_mask: _VariableLike | torch.Tensor | None = None,
        min_depth: float = 1.0e-6,
        weight: Real | torch.Tensor = 1.0,
        row_weight: Weight | Real | torch.Tensor = 1.0,
        kernel: object | None = None,
        name: str = "projection",
    ) -> None:
        q, state = robot_state(q_or_state)
        ids = _normalize_point_ids(point_ids)
        if any(point_id < 0 or point_id >= q.model.nframes for point_id in ids):
            raise ValueError(f"point_ids must index model frame rows in [0, {q.model.nframes})")
        self.point_ids, self.K, self.extrinsics = ids, K, extrinsics
        self.target_px, self.weights, self.valid_mask = target_px, weights, valid_mask
        _intrinsics, camera, _target, _weights, _validity = self._camera_tensors()
        min_depth = float(min_depth)
        if not math.isfinite(min_depth) or min_depth <= 0.0:
            raise ValueError(f"min_depth must be finite and positive, got {min_depth!r}")

        self.state = state
        self.nodes = (state,)
        self.q = q
        self.model = q.model
        self._point_ids = torch.tensor(ids, dtype=torch.long, device=camera.device)
        self.min_depth = min_depth
        super().__init__(
            *variables(K, extrinsics, target_px, weights, valid_mask),
            dim=2 * len(ids),
            weight=weight,
            row_weight=row_weight,
            kernel=kernel,
            group_size=2,
            name=name,
        )

    def _frame_points(self, data: Data) -> torch.Tensor:
        if data.frame_pose_world is None:
            raise RuntimeError("ProjectionResidual requires FK frame placements")
        return data.frame_pose_world.index_select(
            -2,
            self._point_ids.to(device=data.frame_pose_world.device),
        )[..., :3]

    def _camera_tensors(
        self,
        exemplar: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        count = len(self.point_ids)
        options = {} if exemplar is None else {"dtype": exemplar.dtype, "device": exemplar.device}
        intrinsics = check_tensor("K", value(self.K, "K"), shape=(3, 3), floating=True, **options)
        exemplar = intrinsics
        extrinsics = check_tensor(
            "extrinsics",
            value(self.extrinsics, "extrinsics"),
            shape=(4, 4),
            floating=True,
            dtype=exemplar.dtype,
            device=exemplar.device,
        )
        target_px = check_tensor(
            "target_px",
            value(self.target_px, "target_px"),
            shape=(count, 2),
            floating=True,
            dtype=exemplar.dtype,
            device=exemplar.device,
        )
        weights = None
        if self.weights is not None:
            weights = check_tensor(
                "weights",
                value(self.weights, "weights"),
                shape=(count,),
                floating=True,
                dtype=exemplar.dtype,
                device=exemplar.device,
            )
            if bool(torch.any(weights < 0.0)):
                raise ValueError("weights must be non-negative")
        valid_mask = None
        if self.valid_mask is not None:
            valid_mask = check_tensor(
                "valid_mask",
                value(self.valid_mask, "valid_mask"),
                shape=(count,),
                dtype=torch.bool,
                device=exemplar.device,
            )
        return intrinsics, extrinsics, target_px, weights, valid_mask

    @staticmethod
    def _multipliers(
        exemplar: torch.Tensor,
        weights: torch.Tensor | None,
        valid_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        multiplier = torch.ones_like(exemplar)
        active = torch.ones_like(exemplar, dtype=torch.bool)
        if weights is not None:
            multiplier = multiplier * weights
            active = active & (weights != 0.0)
        if valid_mask is not None:
            multiplier = multiplier * valid_mask
            active = active & valid_mask
        return multiplier, active

    def error(self) -> torch.Tensor:
        data = self.state._checked_value(Data)
        points_world = self._frame_points(data)
        intrinsics, extrinsics, target_px, weights, valid_mask = self._camera_tensors(points_world)
        points_camera = _transform_points(points_world, extrinsics)
        predicted_px = _project_points(points_camera, intrinsics, min_depth=self.min_depth)
        multiplier, active = self._multipliers(predicted_px[..., 0], weights, valid_mask)
        rows = (predicted_px - target_px) * multiplier.unsqueeze(-1)
        rows = torch.where(active.unsqueeze(-1), rows, torch.zeros_like(rows))
        return rows.reshape(*rows.shape[:-2], self.dim)

    def jacobian(self) -> tuple[torch.Tensor, ...] | None:
        if any(variable.trainable for variable in self.variables):
            return None
        if not self.q.trainable:
            return ()
        data = self.state._checked_value(Data)
        points_world = self._frame_points(data)
        intrinsics, extrinsics, _target_px, weights, valid_mask = self._camera_tensors(points_world)
        points_camera = _transform_points(points_world, extrinsics)
        frame_jacobians = torch.stack(
            [get_frame_jacobian(self.model, data, point_id) for point_id in self.point_ids],
            dim=-3,
        )
        point_jacobians_world = frame_jacobians[..., :3, :]
        rotation = extrinsics[..., :3, :3]
        point_jacobians_camera = torch.matmul(rotation.unsqueeze(-3), point_jacobians_world)
        pixel_from_camera = _projection_jacobian(points_camera, intrinsics, min_depth=self.min_depth)
        full = torch.matmul(pixel_from_camera, point_jacobians_camera)
        multiplier, active = self._multipliers(points_camera[..., 0], weights, valid_mask)
        full = full * multiplier.unsqueeze(-1).unsqueeze(-1)
        full = torch.where(active.unsqueeze(-1).unsqueeze(-1), full, torch.zeros_like(full))
        full = full.reshape(*full.shape[:-3], self.dim, self.model.nv)
        return (full,)


__all__ = ["ProjectionResidual"]
