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

from .._validation import check_tensor
from ..data_model.model import Model
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

    depth_axis = torch.stack(
        (
            torch.zeros_like(depth_raw),
            torch.zeros_like(depth_raw),
            torch.ones_like(depth_raw),
        ),
        dim=-1,
    )
    # ``clamp_min`` chooses the identity derivative at the boundary.
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
        raw_ids = point_ids.detach().cpu().tolist()  # bench-ok: one-time static frame-id metadata
    else:
        raw_ids = list(point_ids)
    if not raw_ids:
        raise ValueError("point_ids must contain at least one frame-table row")
    if any(isinstance(point_id, bool) or not isinstance(point_id, int) for point_id in raw_ids):
        raise TypeError("point_ids must contain integers")
    return tuple(raw_ids)


def _validate_parameter_names(names: Mapping[str, str | None]) -> tuple[str, ...]:
    declared: list[str] = []
    for label, value in names.items():
        if value is None:
            continue
        if not isinstance(value, str) or not value:
            raise TypeError(f"{label} must be a non-empty string or None")
        if value == "data":
            raise ValueError(f"{label} cannot use the reserved provider output 'data'")
        declared.append(value)
    if len(set(declared)) != len(declared):
        raise ValueError("ProjectionResidual parameter names must be unique")
    return tuple(declared)


def _context_tensor(
    ctx: Mapping[str, Any],
    fallback: torch.Tensor | None,
    parameter_name: str | None,
) -> torch.Tensor | None:
    return fallback if parameter_name is None else ctx[parameter_name]


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
    K_name, extrinsics_name, target_name, weights_name, valid_mask_name
        Optional names for resolving the corresponding tensors from a
        :class:`~better_robot.optim.Problem` context. Named tensors take
        precedence over their constructor fallbacks and are included in
        ``reads``, making differentiable observations explicit.
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
        K_name: str | None = None,
        extrinsics_name: str | None = None,
        target_name: str | None = None,
        weights_name: str | None = None,
        valid_mask_name: str | None = None,
        min_depth: float = 1.0e-6,
        name: str = "projection",
    ) -> None:
        ids = _normalize_point_ids(point_ids)
        if any(point_id < 0 or point_id >= model.nframes for point_id in ids):
            raise ValueError(f"point_ids must index model frame rows in [0, {model.nframes})")
        count = len(ids)
        K = check_tensor("K", K, shape=(3, 3), floating=True)
        extrinsics = check_tensor("extrinsics", extrinsics, shape=(4, 4), floating=True, dtype=K.dtype, device=K.device)
        target_px = check_tensor(
            "target_px", target_px, shape=(count, 2), floating=True, dtype=K.dtype, device=K.device
        )
        if weights is not None:
            weights = check_tensor("weights", weights, shape=(count,), floating=True, dtype=K.dtype, device=K.device)
            if bool(torch.any(weights < 0.0)):
                raise ValueError("weights must be non-negative")
        if valid_mask is not None:
            valid_mask = check_tensor("valid_mask", valid_mask, shape=(count,), dtype=torch.bool, device=K.device)
        min_depth = float(min_depth)
        if not math.isfinite(min_depth) or min_depth <= 0.0:
            raise ValueError(f"min_depth must be finite and positive, got {min_depth!r}")
        if not isinstance(name, str) or not name:
            raise ValueError("name must be a non-empty string")
        declared_parameters = _validate_parameter_names(
            {
                "K_name": K_name,
                "extrinsics_name": extrinsics_name,
                "target_name": target_name,
                "weights_name": weights_name,
                "valid_mask_name": valid_mask_name,
            }
        )

        self.model = model
        self.name = name
        self.reads = ("data", *declared_parameters)
        self.point_ids = ids
        self._point_ids = torch.tensor(ids, dtype=torch.long, device=model.q_neutral.device)
        self.K = K
        self.extrinsics = extrinsics
        self.target_px = target_px
        self.weights = weights
        self.valid_mask = valid_mask
        self.K_name = K_name
        self.extrinsics_name = extrinsics_name
        self.target_name = target_name
        self.weights_name = weights_name
        self.valid_mask_name = valid_mask_name
        self.min_depth = min_depth
        self.dim = 2 * count

    def _frame_points(self, ctx: Mapping[str, Any]) -> torch.Tensor:
        data = ctx["data"]
        frame_pose_world = getattr(data, "frame_pose_world", None)
        if frame_pose_world is None:
            raise RuntimeError("ProjectionResidual requires RobotStateProvider FK with frame placements")
        return frame_pose_world.index_select(-2, self._point_ids)[..., :3]

    def _camera_tensors(
        self,
        ctx: Mapping[str, Any],
        exemplar: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        count = len(self.point_ids)
        intrinsics = _context_tensor(ctx, self.K, self.K_name)
        extrinsics = _context_tensor(ctx, self.extrinsics, self.extrinsics_name)
        target_px = _context_tensor(ctx, self.target_px, self.target_name)
        weights = _context_tensor(ctx, self.weights, self.weights_name)
        valid_mask = _context_tensor(ctx, self.valid_mask, self.valid_mask_name)
        assert intrinsics is not None and extrinsics is not None and target_px is not None
        intrinsics = check_tensor(
            "K", intrinsics, shape=(3, 3), floating=True, dtype=exemplar.dtype, device=exemplar.device
        )
        extrinsics = check_tensor(
            "extrinsics",
            extrinsics,
            shape=(4, 4),
            floating=True,
            dtype=exemplar.dtype,
            device=exemplar.device,
        )
        target_px = check_tensor(
            "target_px",
            target_px,
            shape=(count, 2),
            floating=True,
            dtype=exemplar.dtype,
            device=exemplar.device,
        )
        if weights is not None:
            weights = check_tensor(
                "weights",
                weights,
                shape=(count,),
                floating=True,
                dtype=exemplar.dtype,
                device=exemplar.device,
            )
            if bool(torch.any(weights < 0.0)):
                raise ValueError("weights must be non-negative")
        if valid_mask is not None:
            valid_mask = check_tensor(
                "valid_mask", valid_mask, shape=(count,), dtype=torch.bool, device=exemplar.device
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

    def __call__(self, ctx: Mapping[str, Any]) -> torch.Tensor:
        points_world = self._frame_points(ctx)
        intrinsics, extrinsics, target_px, weights, valid_mask = self._camera_tensors(
            ctx,
            points_world,
        )
        points_camera = _transform_points(points_world, extrinsics)
        predicted_px = _project_points(
            points_camera,
            intrinsics,
            min_depth=self.min_depth,
        )
        multiplier, active = self._multipliers(predicted_px[..., 0], weights, valid_mask)
        rows = (predicted_px - target_px) * multiplier.unsqueeze(-1)
        rows = torch.where(active.unsqueeze(-1), rows, torch.zeros_like(rows))
        return rows.reshape(*rows.shape[:-2], self.dim)

    def jacobian_blocks(self, ctx: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        """Return the complete mask-reduced analytic ``q`` Jacobian block."""
        data = ctx["data"]
        points_world = self._frame_points(ctx)
        intrinsics, extrinsics, _target_px, weights, valid_mask = self._camera_tensors(
            ctx,
            points_world,
        )
        points_camera = _transform_points(points_world, extrinsics)

        frame_jacobians = torch.stack(
            [
                get_frame_jacobian(
                    self.model,
                    data,
                    point_id,
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
        multiplier, active = self._multipliers(points_camera[..., 0], weights, valid_mask)
        full = full * multiplier.unsqueeze(-1).unsqueeze(-1)
        full = torch.where(
            active.unsqueeze(-1).unsqueeze(-1),
            full,
            torch.zeros_like(full),
        )
        full = full.reshape(*full.shape[:-3], self.dim, self.model.nv)

        indices = ctx.free_indices("q").to(device=full.device)
        return {"q": full.index_select(-1, indices)}


__all__ = ["ProjectionResidual"]
