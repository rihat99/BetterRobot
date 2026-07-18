"""``PoseResidual``, ``PositionResidual``, ``OrientationResidual``.

Analytic Jacobians via ``get_frame_jacobian`` composed with
``right_jacobian_inv_se3(log_err)``. Replaces the legacy ``Jlog ≈ I``
approximation.

See ``docs/concepts/kinematics_and_jacobians.md`` and
``docs/concepts/residuals_costs_and_solvers.md``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from ..data_model.data import Data
from ..data_model.model import Model
from ..kinematics.jacobian import get_frame_jacobian
from ..lie import se3, so3
from ..lie.tangents import right_jacobian_inv_se3, right_jacobian_inv_so3
from .base import _configuration


def _context_state(
    ctx: Mapping[str, Any],
    model: Model | None,
) -> tuple[Model, torch.Tensor, Data]:
    if model is None:
        raise TypeError("kinematic residuals require model=... for evaluation")
    q = _configuration(ctx)
    data = ctx["data"]
    if not isinstance(data, Data):
        raise TypeError(f"data must be Data, got {type(data).__name__}")
    return model, q, data


def _get_frame_pose(model: Model, data: Data, frame_id: int) -> torch.Tensor:
    """Get world pose of ``frame_id`` from ``data.frame_pose_world`` or
    compute on-the-fly."""
    if data.frame_pose_world is not None:
        return data.frame_pose_world[..., frame_id, :]  # (B..., 7)
    frame = model.frames[frame_id]
    T_parent = data.joint_pose_world[..., frame.parent_joint, :]
    T_local = model.values.frame_placements[..., frame_id, :]
    return se3.compose(T_parent, T_local)


def _target_from_input(
    ctx: Mapping[str, Any],
    fallback: torch.Tensor,
    target_name: str | None,
) -> torch.Tensor:
    """Resolve a declared block parameter or use the constructor fallback."""
    if target_name is None:
        return fallback
    target = ctx[target_name]
    if not isinstance(target, torch.Tensor):
        raise TypeError(f"{target_name!r} must be a tensor, got {type(target).__name__}")
    return target


def _validate_target_name(target_name: str | None) -> None:
    if target_name is not None and (not isinstance(target_name, str) or not target_name):
        raise TypeError("target_name must be a non-empty string or None")


def _resolve_frame_id(
    model: Model | None,
    frame: str | None,
    frame_id: int | None,
) -> int:
    if frame is not None and frame_id is not None:
        raise TypeError("provide either frame=... or frame_id=..., not both")
    if frame is not None:
        if not isinstance(frame, str) or not frame:
            raise TypeError("frame must be a non-empty string")
        if model is None:
            raise TypeError("PoseResidual frame=... requires a model")
        return model.frame_id(frame)
    if isinstance(frame_id, bool) or not isinstance(frame_id, int):
        raise TypeError("PoseResidual requires frame=... or an integer frame_id=...")
    return frame_id


class PoseResidual:
    """6-DOF pose residual targeting a frame. ``dim = 6``.

    ``r = log(T_target^{-1} ⊕ T_ee)`` — error in the target frame.
    Position (lin) and orientation (ang) parts are weighted independently.
    """

    name: str = "pose"
    reads = ("q", "data")

    def __init__(
        self,
        model: Model | None = None,
        *,
        frame: str | None = None,
        frame_id: int | None = None,
        target: torch.Tensor,
        pos_weight: float = 1.0,
        ori_weight: float = 1.0,
        name: str = "pose",
        target_name: str | None = None,
    ) -> None:
        _validate_target_name(target_name)
        self.model = model
        self.name = name
        self.target_name = target_name
        self.reads = ("q", "data", target_name) if target_name is not None else ("q", "data")
        self.frame_id = _resolve_frame_id(model, frame, frame_id)
        self.target = target
        self.pos_weight = pos_weight
        self.ori_weight = ori_weight
        self._weight = torch.tensor(
            [pos_weight, pos_weight, pos_weight, ori_weight, ori_weight, ori_weight],
            dtype=target.dtype,
            device=target.device,
        )
        self.dim = 6

    def __call__(self, ctx: Mapping[str, Any]) -> torch.Tensor:
        target = _target_from_input(ctx, self.target, self.target_name)
        model, q, data = _context_state(ctx, self.model)
        T_target = target.to(device=q.device, dtype=q.dtype)
        T_ee = _get_frame_pose(model, data, self.frame_id)  # (B..., 7)
        T_err = se3.compose(se3.inverse(T_target), T_ee)  # (B..., 7)
        r = se3.log(T_err)  # (B..., 6)
        weight = self._weight.to(device=r.device, dtype=r.dtype)
        return r * weight

    def _analytic_jacobian(
        self,
        ctx: Mapping[str, Any],
    ) -> torch.Tensor:
        """Analytic Jacobian: ``Jr^{-1}(r) @ Ad(T_ee^{-1}) @ J_frame_world``.

        See docs/concepts/kinematics_and_jacobians.md.
        """
        target = _target_from_input(ctx, self.target, self.target_name)
        model, q, data = _context_state(ctx, self.model)
        T_target = target.to(device=q.device, dtype=q.dtype)
        T_ee = _get_frame_pose(model, data, self.frame_id)
        T_err = se3.compose(se3.inverse(T_target), T_ee)
        r = se3.log(T_err)  # (B..., 6)

        # World-frame spatial Jacobian of the end-effector frame
        # (LOCAL_WORLD_ALIGNED convention: [v_frame_origin_world, omega_world])
        J_world = get_frame_jacobian(model, data, self.frame_id)  # (B..., 6, nv)

        # Body-frame Jacobian: just rotate both halves by R_ee^T.
        # get_frame_jacobian returns the velocity of the frame origin (not the world
        # origin), so only the rotation part of Ad(T_ee^{-1}) applies — the
        # cross-term -R^T @ hat(p) @ omega would be spurious here.
        R_ee = so3.to_matrix(T_ee[..., 3:])  # (B..., 3, 3)
        J_local = torch.cat(
            [
                torch.matmul(R_ee.mT, J_world[..., :3, :]),
                torch.matmul(R_ee.mT, J_world[..., 3:, :]),
            ],
            dim=-2,
        )  # (B..., 6, nv)

        # Analytic Jacobian: Jr^{-1}(r) @ J_local
        Jr_inv = right_jacobian_inv_se3(r)  # (B..., 6, 6)
        J_analytic = torch.matmul(Jr_inv, J_local)  # (B..., 6, nv)

        # Apply weights row-wise
        weight = self._weight.to(device=r.device, dtype=r.dtype)
        return J_analytic * weight.unsqueeze(-1)  # (B..., 6, nv)

    def jacobian_blocks(
        self,
        ctx: Mapping[str, Any],
    ) -> dict[str, torch.Tensor]:
        """Return the mask-reduced analytic ``q`` block."""
        full = self._analytic_jacobian(ctx)
        indices = ctx.free_indices("q").to(device=full.device)
        return {"q": full.index_select(-1, indices)}


class PositionResidual:
    """3-DOF position residual. ``dim = 3``.

    ``r = (p_ee - p_target) * weight`` — Euclidean position error.
    Analytic Jacobian is just the top-3 rows of the spatial Jacobian.
    """

    name: str = "position"
    reads = ("q", "data")

    def __init__(
        self,
        *,
        frame_id: int,
        target: torch.Tensor,
        weight: float = 1.0,
        model: Model | None = None,
        name: str = "position",
        target_name: str | None = None,
    ) -> None:
        _validate_target_name(target_name)
        self.model = model
        self.name = name
        self.target_name = target_name
        self.reads = ("q", "data", target_name) if target_name is not None else ("q", "data")
        self.frame_id = frame_id
        self.target = target
        self.weight = weight
        self.dim = 3

    def __call__(self, ctx: Mapping[str, Any]) -> torch.Tensor:
        target = _target_from_input(ctx, self.target, self.target_name)
        model, q, data = _context_state(ctx, self.model)
        T_target = target.to(device=q.device, dtype=q.dtype)
        T_ee = _get_frame_pose(model, data, self.frame_id)
        p_ee = T_ee[..., :3]
        p_target = T_target[..., :3]
        return (p_ee - p_target) * self.weight

    def _analytic_jacobian(
        self,
        ctx: Mapping[str, Any],
    ) -> torch.Tensor:
        model, _q, data = _context_state(ctx, self.model)
        J_world = get_frame_jacobian(model, data, self.frame_id)  # (B..., 6, nv)
        return J_world[..., :3, :] * self.weight  # (B..., 3, nv)

    def jacobian_blocks(
        self,
        ctx: Mapping[str, Any],
    ) -> dict[str, torch.Tensor]:
        """Return the mask-reduced analytic ``q`` block."""
        full = self._analytic_jacobian(ctx)
        indices = ctx.free_indices("q").to(device=full.device)
        return {"q": full.index_select(-1, indices)}


class OrientationResidual:
    """3-DOF orientation residual. ``dim = 3``.

    ``r = log_SO3(q_target^{-1} ⊕ q_ee) * weight``.
    """

    name: str = "orientation"
    reads = ("q", "data")

    def __init__(
        self,
        *,
        frame_id: int,
        target: torch.Tensor,
        weight: float = 1.0,
        model: Model | None = None,
        name: str = "orientation",
        target_name: str | None = None,
    ) -> None:
        _validate_target_name(target_name)
        self.model = model
        self.name = name
        self.target_name = target_name
        self.reads = ("q", "data", target_name) if target_name is not None else ("q", "data")
        self.frame_id = frame_id
        self.target = target
        self.weight = weight
        self.dim = 3

    def __call__(self, ctx: Mapping[str, Any]) -> torch.Tensor:
        target = _target_from_input(ctx, self.target, self.target_name)
        model, q, data = _context_state(ctx, self.model)
        T_target = target.to(device=q.device, dtype=q.dtype)
        T_ee = _get_frame_pose(model, data, self.frame_id)
        q_target = T_target[..., 3:]  # (B..., 4) quaternion
        q_ee = T_ee[..., 3:]
        r_so3 = so3.log(so3.compose(so3.inverse(q_target), q_ee))  # (B..., 3)
        return r_so3 * self.weight

    def _analytic_jacobian(
        self,
        ctx: Mapping[str, Any],
    ) -> torch.Tensor:
        target = _target_from_input(ctx, self.target, self.target_name)
        model, q, data = _context_state(ctx, self.model)
        T_target = target.to(device=q.device, dtype=q.dtype)
        T_ee = _get_frame_pose(model, data, self.frame_id)
        q_target = T_target[..., 3:]
        q_ee = T_ee[..., 3:]
        r = so3.log(so3.compose(so3.inverse(q_target), q_ee))  # (B..., 3)

        J_world = get_frame_jacobian(model, data, self.frame_id)
        J_ang = J_world[..., 3:, :]  # angular rows (B..., 3, nv)

        Jr_inv = right_jacobian_inv_so3(r)  # (B..., 3, 3)
        # Local angular Jacobian: J_ang_local = R_ee^T @ J_ang_world
        R_ee = so3.to_matrix(q_ee)  # (B..., 3, 3)
        J_ang_local = torch.matmul(R_ee.mT, J_ang)  # (B..., 3, nv)
        return torch.matmul(Jr_inv, J_ang_local) * self.weight

    def jacobian_blocks(
        self,
        ctx: Mapping[str, Any],
    ) -> dict[str, torch.Tensor]:
        """Return the mask-reduced analytic ``q`` block."""
        full = self._analytic_jacobian(ctx)
        indices = ctx.free_indices("q").to(device=full.device)
        return {"q": full.index_select(-1, indices)}
