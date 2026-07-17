"""Camera-thin pinhole projection residual over model frame-table rows.

Operational frames, marker sites, and body frames share the model's frame
table. :class:`ProjectionResidual` therefore needs only frame-row indices;
there is no separate vertex or landmark kinematics path.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

import torch

from ..data_model.model import Model
from ..kinematics import ReferenceFrame
from ..kinematics.jacobian import get_frame_jacobian


def _transform_points(points_world: torch.Tensor, extrinsics: torch.Tensor) -> torch.Tensor:
    """Apply a world-to-camera homogeneous transform to ``(..., N, 3)`` points."""
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
    """Project ``(..., N, 3)`` camera points, clamping depth from below."""
    homogeneous = torch.matmul(
        intrinsics.unsqueeze(-3),
        points_camera.unsqueeze(-1),
    ).squeeze(-1)
    depth = points_camera[..., 2].clamp_min(min_depth)
    return homogeneous[..., :2] / depth.unsqueeze(-1)


def _projection_jacobian(
    points_camera: torch.Tensor,
    intrinsics: torch.Tensor,
    *,
    min_depth: float,
) -> torch.Tensor:
    """Return ``d(project(points_camera))/d(points_camera)``.

    The denominator is camera-space ``z``. Below ``min_depth`` it is a
    constant, matching :func:`_project_points` and keeping behind-camera
    rows finite.
    """
    homogeneous = torch.matmul(
        intrinsics.unsqueeze(-3),
        points_camera.unsqueeze(-1),
    ).squeeze(-1)
    numerator = homogeneous[..., :2]
    depth_raw = points_camera[..., 2]
    depth = depth_raw.clamp_min(min_depth)
    numerator_jacobian = intrinsics[..., :2, :].unsqueeze(-3)

    depth_axis = points_camera.new_tensor((0.0, 0.0, 1.0))
    depth_active = (depth_raw > min_depth).to(dtype=points_camera.dtype)
    denominator_term = (
        numerator.unsqueeze(-1)
        * depth_axis
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
        raw_ids = point_ids.detach().cpu().tolist()
    else:
        raw_ids = list(point_ids)
    if not raw_ids:
        raise ValueError("point_ids must contain at least one frame-table row")
    if any(isinstance(point_id, bool) or not isinstance(point_id, int) for point_id in raw_ids):
        raise TypeError("point_ids must contain integers")
    return tuple(raw_ids)


def _require_floating_tensor(value: Any, *, name: str, suffix: tuple[int, ...]) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise TypeError(f"{name} must be a floating torch.Tensor")
    if tuple(value.shape[-len(suffix) :]) != suffix:
        raise ValueError(f"{name} must end in shape {suffix}, got {tuple(value.shape)}")
    return value


def _require_same_dtype_device(
    reference: torch.Tensor,
    value: torch.Tensor,
    *,
    name: str,
) -> None:
    if value.dtype != reference.dtype:
        raise TypeError(f"{name} must have dtype {reference.dtype}, got {value.dtype}")
    if value.device != reference.device:
        raise ValueError(f"{name} must be on device {reference.device}, got {value.device}")


class ProjectionResidual:
    """Pinhole reprojection error for frame, marker, or site rows.

    Parameters
    ----------
    model
        Model whose frame table owns ``point_ids``.
    point_ids
        One-dimensional sequence of frame-table row indices. Body frames and
        marker/site operational frames use the same path.
    K
        Camera intrinsics with shape ``(..., 3, 3)``.
    extrinsics
        World-to-camera homogeneous transform with shape ``(..., 4, 4)``.
    target_px
        Target pixels with shape ``(..., N, 2)``.
    weights
        Optional confidence multipliers with shape ``(..., N)``. These are
        ordinary raw-residual multipliers; robust weighting belongs on the
        surrounding ``ResidualItem`` with ``group_size=2``.
    valid_mask
        Optional boolean mask with shape ``(..., N)``. Invalid points produce
        two finite zero rows.
    min_depth
        Positive camera-space depth floor. Points at or behind the floor are
        projected with the clamped denominator instead of producing NaNs.

    Notes
    -----
    All observation tensors are fixed-size and may carry leading batch axes.
    Variable-size point sets must be padded and represented by
    ``valid_mask``. This class intentionally does not define a camera object
    or visibility policy.
    """

    reads = ("data",)

    def __init__(
        self,
        model: Model,
        point_ids: Sequence[int] | torch.Tensor,
        K: torch.Tensor,
        extrinsics: torch.Tensor,
        target_px: torch.Tensor,
        *,
        weights: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
        min_depth: float = 1.0e-6,
        name: str = "projection",
    ) -> None:
        ids = _normalize_point_ids(point_ids)
        if any(point_id < 0 or point_id >= model.nframes for point_id in ids):
            raise ValueError(f"point_ids must index model frame rows in [0, {model.nframes})")
        count = len(ids)
        K = _require_floating_tensor(K, name="K", suffix=(3, 3))
        extrinsics = _require_floating_tensor(
            extrinsics,
            name="extrinsics",
            suffix=(4, 4),
        )
        target_px = _require_floating_tensor(
            target_px,
            name="target_px",
            suffix=(count, 2),
        )
        _require_same_dtype_device(K, extrinsics, name="extrinsics")
        _require_same_dtype_device(K, target_px, name="target_px")
        if weights is not None:
            weights = _require_floating_tensor(
                weights,
                name="weights",
                suffix=(count,),
            )
            if not bool(torch.isfinite(weights).all()) or bool(torch.any(weights < 0.0)):
                raise ValueError("weights must be finite and non-negative")
            _require_same_dtype_device(K, weights, name="weights")
        if valid_mask is not None:
            if not isinstance(valid_mask, torch.Tensor) or valid_mask.dtype != torch.bool:
                raise TypeError("valid_mask must be a boolean torch.Tensor")
            if tuple(valid_mask.shape[-1:]) != (count,):
                raise ValueError(f"valid_mask must end in shape ({count},), got {tuple(valid_mask.shape)}")
            if valid_mask.device != K.device:
                raise ValueError(f"valid_mask must be on device {K.device}, got {valid_mask.device}")
        min_depth = float(min_depth)
        if not math.isfinite(min_depth) or min_depth <= 0.0:
            raise ValueError(f"min_depth must be finite and positive, got {min_depth!r}")
        if not isinstance(name, str) or not name:
            raise ValueError("name must be a non-empty string")

        self.model = model
        self.name = name
        self.point_ids = ids
        self._point_ids = torch.tensor(ids, dtype=torch.long, device=K.device)
        self.K = K
        self.extrinsics = extrinsics
        self.target_px = target_px
        self.weights = weights
        self.valid_mask = valid_mask
        self.min_depth = min_depth
        self.dim = 2 * count

    def _frame_points(self, ctx: Mapping[str, Any]) -> torch.Tensor:
        data = ctx["data"]
        frame_pose_world = getattr(data, "frame_pose_world", None)
        if frame_pose_world is None:
            raise RuntimeError("ProjectionResidual requires RobotStateProvider FK with frame placements")
        if frame_pose_world.dtype != self.K.dtype:
            raise TypeError(
                "ProjectionResidual observation dtype does not match the evaluated "
                f"model state: {self.K.dtype} != {frame_pose_world.dtype}"
            )
        if frame_pose_world.device != self.K.device:
            raise ValueError(
                "ProjectionResidual observations and evaluated model state must be "
                f"on the same device: {self.K.device} != {frame_pose_world.device}"
            )
        return frame_pose_world.index_select(-2, self._point_ids)[..., :3]

    def _camera_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.K, self.extrinsics, self.target_px

    def _multipliers(self, exemplar: torch.Tensor) -> torch.Tensor:
        multiplier = torch.ones_like(exemplar)
        if self.weights is not None:
            multiplier = multiplier * self.weights
        if self.valid_mask is not None:
            multiplier = multiplier * self.valid_mask
        return multiplier

    def __call__(self, ctx: Mapping[str, Any]) -> torch.Tensor:
        points_world = self._frame_points(ctx)
        intrinsics, extrinsics, target_px = self._camera_tensors()
        points_camera = _transform_points(points_world, extrinsics)
        predicted_px = _project_points(
            points_camera,
            intrinsics,
            min_depth=self.min_depth,
        )
        multiplier = self._multipliers(predicted_px[..., 0])
        rows = (predicted_px - target_px) * multiplier.unsqueeze(-1)
        return rows.reshape(*rows.shape[:-2], self.dim)

    def jacobian_blocks(self, ctx: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        """Return the complete mask-reduced analytic ``q`` Jacobian block."""
        data = ctx["data"]
        points_world = self._frame_points(ctx)
        intrinsics, extrinsics, _target_px = self._camera_tensors()
        points_camera = _transform_points(points_world, extrinsics)

        frame_jacobians = torch.stack(
            [
                get_frame_jacobian(
                    self.model,
                    data,
                    point_id,
                    reference=ReferenceFrame.LOCAL_WORLD_ALIGNED,
                )
                for point_id in self.point_ids
            ],
            dim=-3,
        )
        point_jacobians_world = frame_jacobians[..., :3, :]
        rotation = extrinsics[..., :3, :3]
        point_jacobians_camera = torch.matmul(
            rotation.unsqueeze(-3),
            point_jacobians_world,
        )
        pixel_from_camera = _projection_jacobian(
            points_camera,
            intrinsics,
            min_depth=self.min_depth,
        )
        full = torch.matmul(pixel_from_camera, point_jacobians_camera)
        multiplier = self._multipliers(points_camera[..., 0])
        full = full * multiplier.unsqueeze(-1).unsqueeze(-1)
        full = full.reshape(*full.shape[:-3], self.dim, self.model.nv)

        indices = ctx.free_indices("q").to(device=full.device)
        return {"q": full.index_select(-1, indices)}


__all__ = ["ProjectionResidual"]
