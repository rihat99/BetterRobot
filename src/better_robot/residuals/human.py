"""Rotation-limit residuals for spherical human joints.

Swing/twist decomposition is expressed in each joint's local coordinates.
The decomposition has an unavoidable singularity at a pure 180-degree swing;
this module uses the conventional zero-twist representative there and leaves
Jacobian construction to the named-block tangent-AD path.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any

import torch

from ..data_model.model import Model
from ..lie import so3


def _configuration(ctx: Mapping[str, Any]) -> torch.Tensor:
    q = ctx["q"]
    if not isinstance(q, torch.Tensor):
        raise TypeError("named context entry 'q' must be a torch.Tensor")
    return q


def _per_joint_values(
    value: Real | torch.Tensor,
    *,
    count: int,
    label: str,
) -> torch.Tensor:
    """Normalize a scalar or ``(count,)`` floating value table."""
    if isinstance(value, Real):
        result = torch.full(
            (count,),
            float(value),  # bench-ok: constructor-only Python scalar, never a tensor
            dtype=torch.float32,
        )
    elif isinstance(value, torch.Tensor):
        if not value.is_floating_point():
            raise TypeError(f"{label} must use a floating dtype")
        if value.ndim == 0:
            result = value.expand(count)
        elif tuple(value.shape) == (count,):
            result = value
        else:
            raise ValueError(f"{label} must be scalar or have shape ({count},), got {tuple(value.shape)}")
    else:
        raise TypeError(f"{label} must be a real number or torch.Tensor")
    if not bool(torch.isfinite(result).all()):
        raise ValueError(f"{label} must contain only finite values")
    return result


class SwingTwistLimitResidual:
    """One-sided swing/twist limits for selected spherical joints.

    For every selected joint the three rows are::

        [relu(swing - swing_max),
         relu(twist_min - twist),
         relu(twist - twist_max)]

    ``twist_axis`` is expressed in the joint-local frame and may be one unit
    axis shared by all joints or a ``(J, 3)`` table. ``twist_range`` must be a
    non-wrapping interval inside ``(-pi, pi)``. The output dimension is
    ``3 * J`` and rows are grouped per joint in ``joint_ids`` order.

    Quaternion sign is folded by wrapping twist through ``atan2(sin, cos)``.
    A pure pi swing has no unique twist; the residual assigns zero twist in a
    tiny neighbourhood of that singularity. Consequently there is no truthful
    globally analytic Jacobian. Named-block problems use tangent-space AD;
    callers may select the explicit finite-difference debug strategy.
    """

    reads = ("q",)

    def __init__(  # noqa: PLR0912, PLR0915 - validates one static residual contract
        self,
        model: Model,
        joint_ids: Sequence[int],
        twist_axis: torch.Tensor,
        swing_max: Real | torch.Tensor,
        twist_range: tuple[Real, Real] | torch.Tensor,
        *,
        name: str = "swing_twist_limit",
    ) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError("name must be a non-empty string")
        ids = tuple(joint_ids)
        if not ids:
            raise ValueError("joint_ids must contain at least one spherical joint")
        if any(isinstance(joint_id, bool) or not isinstance(joint_id, int) for joint_id in ids):
            raise TypeError("joint_ids must contain only integer joint ids")
        if len(set(ids)) != len(ids):
            raise ValueError("joint_ids must be unique")
        for joint_id in ids:
            if joint_id < 0 or joint_id >= model.njoints:
                raise ValueError(f"joint id {joint_id} is outside [0, {model.njoints})")
            kind = model.joint_models[joint_id].kind
            if kind != "spherical":
                raise ValueError(
                    f"joint {joint_id} ({model.joint_names[joint_id]!r}) has kind {kind!r}; "
                    "SwingTwistLimitResidual accepts spherical joints only"
                )

        count = len(ids)
        if not isinstance(twist_axis, torch.Tensor) or not twist_axis.is_floating_point():
            raise TypeError("twist_axis must be a floating torch.Tensor")
        if tuple(twist_axis.shape) == (3,):
            axes = twist_axis.expand(count, 3)
        elif tuple(twist_axis.shape) == (count, 3):
            axes = twist_axis
        else:
            raise ValueError(f"twist_axis must have shape (3,) or ({count}, 3), got {tuple(twist_axis.shape)}")
        if not bool(torch.isfinite(axes).all()):
            raise ValueError("twist_axis must contain only finite values")
        axis_norm = torch.linalg.vector_norm(axes, dim=-1)
        if not torch.allclose(axis_norm, torch.ones_like(axis_norm), atol=2e-6, rtol=2e-5):
            raise ValueError("every twist_axis row must have unit norm")

        swing = _per_joint_values(swing_max, count=count, label="swing_max")
        if bool(torch.any((swing <= 0.0) | (swing >= math.pi))):
            raise ValueError("swing_max must lie strictly inside (0, pi)")

        if isinstance(twist_range, torch.Tensor):
            if not twist_range.is_floating_point():
                raise TypeError("twist_range must use a floating dtype")
            ranges = twist_range
        else:
            try:
                ranges = torch.tensor(tuple(twist_range), dtype=torch.float32)
            except (TypeError, ValueError) as exc:
                raise TypeError("twist_range must be a pair of real numbers or a floating tensor") from exc
        if tuple(ranges.shape) == (2,):
            ranges = ranges.expand(count, 2)
        elif tuple(ranges.shape) != (count, 2):
            raise ValueError(f"twist_range must have shape (2,) or ({count}, 2), got {tuple(ranges.shape)}")
        if not bool(torch.isfinite(ranges).all()):
            raise ValueError("twist_range must contain only finite values")
        lower, upper = ranges.unbind(dim=-1)
        if bool(torch.any((lower <= -math.pi) | (upper >= math.pi) | (lower > upper))):
            raise ValueError("twist_range rows must be non-wrapping intervals strictly inside (-pi, pi)")

        q_indices: list[int] = []
        for joint_id in ids:
            start = model.idx_qs[joint_id]
            q_indices.extend(range(start, start + 4))

        self.model = model
        self.name = name
        self.joint_ids = ids
        self.twist_axis = axes / axis_norm.unsqueeze(-1)
        self.swing_max = swing
        self.twist_range = ranges
        self._q_indices = torch.tensor(q_indices, dtype=torch.long)
        self._joint_count = count
        self.dim = 3 * count

    def __call__(self, ctx: Mapping[str, Any]) -> torch.Tensor:
        q = _configuration(ctx)
        indices = self._q_indices.to(device=q.device)
        joint_q = q.index_select(-1, indices).reshape(*q.shape[:-1], self._joint_count, 4)
        joint_q = so3.normalize(joint_q)

        axes = self.twist_axis.to(device=q.device, dtype=q.dtype)
        vector = joint_q[..., :3]
        scalar = joint_q[..., 3]
        along = (vector * axes).sum(dim=-1)
        perpendicular = vector - along.unsqueeze(-1) * axes

        swing_sin_half = torch.linalg.vector_norm(perpendicular, dim=-1)
        twist_norm = torch.linalg.vector_norm(torch.stack((scalar, along), dim=-1), dim=-1)
        swing = 2.0 * torch.atan2(swing_sin_half, twist_norm)

        # At pure pi swing, scalar == along == 0 and twist is not defined.
        # Feed atan2 a benign zero-twist representative so inactive twist rows
        # cannot inherit NaNs from the undefined decomposition.
        singular = twist_norm <= 16.0 * torch.finfo(q.dtype).eps
        safe_along = torch.where(singular, torch.zeros_like(along), along)
        safe_scalar = torch.where(singular, torch.ones_like(scalar), scalar)
        raw_twist = 2.0 * torch.atan2(safe_along, safe_scalar)
        twist = torch.atan2(torch.sin(raw_twist), torch.cos(raw_twist))

        swing_max = self.swing_max.to(device=q.device, dtype=q.dtype)
        twist_range = self.twist_range.to(device=q.device, dtype=q.dtype)
        lower = twist_range[..., 0]
        upper = twist_range[..., 1]
        rows = torch.stack(
            (
                torch.relu(swing - swing_max),
                torch.relu(lower - twist),
                torch.relu(twist - upper),
            ),
            dim=-1,
        )
        return rows.reshape(*q.shape[:-1], self.dim)


__all__ = ["SwingTwistLimitResidual"]
