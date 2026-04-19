"""``PoseResidual``, ``PositionResidual``, ``OrientationResidual``.

Analytic Jacobians via ``get_frame_jacobian`` composed with
``right_jacobian_inv_se3(log_err)``. Replaces the legacy ``Jlog ≈ I``
approximation.

See ``docs/05_KINEMATICS.md §5`` and
``docs/07_RESIDUALS_COSTS_SOLVERS.md §2``.
"""

from __future__ import annotations

import torch

from ..lie import se3, so3
from ..lie.tangents import right_jacobian_inv_se3, right_jacobian_inv_so3
from .base import Residual, ResidualState
from .registry import register_residual


def _get_frame_pose(state: ResidualState, frame_id: int) -> torch.Tensor:
    """Get world pose of ``frame_id`` from ``data.frame_pose_world`` or
    compute on-the-fly."""
    if state.data.frame_pose_world is not None:
        return state.data.frame_pose_world[..., frame_id, :]  # (B..., 7)
    frame = state.model.frames[frame_id]
    T_parent = state.data.joint_pose_world[..., frame.parent_joint, :]
    T_local = frame.joint_placement.to(
        device=state.variables.device, dtype=state.variables.dtype
    )
    return se3.compose(T_parent, T_local)


@register_residual("pose")
class PoseResidual:
    """6-DOF pose residual targeting a frame. ``dim = 6``.

    ``r = log(T_target^{-1} ⊕ T_ee)`` — error in the target frame.
    Position (lin) and orientation (ang) parts are weighted independently.
    """

    name: str = "pose"

    def __init__(
        self,
        *,
        frame_id: int,
        target: torch.Tensor,
        pos_weight: float = 1.0,
        ori_weight: float = 1.0,
    ) -> None:
        self.frame_id = frame_id
        self.target = target
        self.pos_weight = pos_weight
        self.ori_weight = ori_weight
        self.dim = 6

    def __call__(self, state: ResidualState) -> torch.Tensor:
        T_target = self.target.to(
            device=state.variables.device, dtype=state.variables.dtype
        )
        T_ee = _get_frame_pose(state, self.frame_id)              # (B..., 7)
        T_err = se3.compose(se3.inverse(T_target), T_ee)          # (B..., 7)
        r = se3.log(T_err)                                        # (B..., 6)
        weight = r.new_tensor([self.pos_weight] * 3 + [self.ori_weight] * 3)
        return r * weight

    def jacobian(self, state: ResidualState) -> torch.Tensor | None:
        """Analytic Jacobian: ``Jr^{-1}(r) @ Ad(T_ee^{-1}) @ J_frame_world``.

        See docs/05_KINEMATICS.md §5.
        """
        from ..kinematics.jacobian import get_frame_jacobian

        T_target = self.target.to(
            device=state.variables.device, dtype=state.variables.dtype
        )
        T_ee = _get_frame_pose(state, self.frame_id)
        T_err = se3.compose(se3.inverse(T_target), T_ee)
        r = se3.log(T_err)                                        # (B..., 6)

        # World-frame spatial Jacobian of the end-effector frame
        # (LOCAL_WORLD_ALIGNED convention: [v_frame_origin_world, omega_world])
        J_world = get_frame_jacobian(state.model, state.data, self.frame_id)  # (B..., 6, nv)

        # Body-frame Jacobian: just rotate both halves by R_ee^T.
        # get_frame_jacobian returns the velocity of the frame origin (not the world
        # origin), so only the rotation part of Ad(T_ee^{-1}) applies — the
        # cross-term -R^T @ hat(p) @ omega would be spurious here.
        R_ee = so3.to_matrix(T_ee[..., 3:])                       # (B..., 3, 3)
        J_local = torch.cat([
            torch.matmul(R_ee.mT, J_world[..., :3, :]),
            torch.matmul(R_ee.mT, J_world[..., 3:, :]),
        ], dim=-2)                                                  # (B..., 6, nv)

        # Analytic Jacobian: Jr^{-1}(r) @ J_local
        Jr_inv = right_jacobian_inv_se3(r)                        # (B..., 6, 6)
        J_analytic = torch.matmul(Jr_inv, J_local)                # (B..., 6, nv)

        # Apply weights row-wise
        weight = r.new_tensor([self.pos_weight] * 3 + [self.ori_weight] * 3)
        return J_analytic * weight.unsqueeze(-1)                  # (B..., 6, nv)


@register_residual("position")
class PositionResidual:
    """3-DOF position residual. ``dim = 3``.

    ``r = (p_ee - p_target) * weight`` — Euclidean position error.
    Analytic Jacobian is just the top-3 rows of the spatial Jacobian.
    """

    name: str = "position"

    def __init__(
        self,
        *,
        frame_id: int,
        target: torch.Tensor,
        weight: float = 1.0,
    ) -> None:
        self.frame_id = frame_id
        self.target = target
        self.weight = weight
        self.dim = 3

    def __call__(self, state: ResidualState) -> torch.Tensor:
        T_target = self.target.to(
            device=state.variables.device, dtype=state.variables.dtype
        )
        T_ee = _get_frame_pose(state, self.frame_id)
        p_ee = T_ee[..., :3]
        p_target = T_target[..., :3]
        return (p_ee - p_target) * self.weight

    def jacobian(self, state: ResidualState) -> torch.Tensor | None:
        from ..kinematics.jacobian import get_frame_jacobian

        J_world = get_frame_jacobian(state.model, state.data, self.frame_id)  # (B..., 6, nv)
        return J_world[..., :3, :] * self.weight  # (B..., 3, nv)


@register_residual("orientation")
class OrientationResidual:
    """3-DOF orientation residual. ``dim = 3``.

    ``r = log_SO3(q_target^{-1} ⊕ q_ee) * weight``.
    """

    name: str = "orientation"

    def __init__(
        self,
        *,
        frame_id: int,
        target: torch.Tensor,
        weight: float = 1.0,
    ) -> None:
        self.frame_id = frame_id
        self.target = target
        self.weight = weight
        self.dim = 3

    def __call__(self, state: ResidualState) -> torch.Tensor:
        T_target = self.target.to(
            device=state.variables.device, dtype=state.variables.dtype
        )
        T_ee = _get_frame_pose(state, self.frame_id)
        q_target = T_target[..., 3:]  # (B..., 4) quaternion
        q_ee = T_ee[..., 3:]
        r_so3 = so3.log(so3.compose(so3.inverse(q_target), q_ee))  # (B..., 3)
        return r_so3 * self.weight

    def jacobian(self, state: ResidualState) -> torch.Tensor | None:
        from ..kinematics.jacobian import get_frame_jacobian

        T_target = self.target.to(
            device=state.variables.device, dtype=state.variables.dtype
        )
        T_ee = _get_frame_pose(state, self.frame_id)
        q_target = T_target[..., 3:]
        q_ee = T_ee[..., 3:]
        r = so3.log(so3.compose(so3.inverse(q_target), q_ee))  # (B..., 3)

        J_world = get_frame_jacobian(state.model, state.data, self.frame_id)
        J_ang = J_world[..., 3:, :]  # angular rows (B..., 3, nv)

        Jr_inv = right_jacobian_inv_so3(r)  # (B..., 3, 3)
        # Local angular Jacobian: J_ang_local = R_ee^T @ J_ang_world
        R_ee = so3.to_matrix(q_ee)           # (B..., 3, 3)
        J_ang_local = torch.matmul(R_ee.mT, J_ang)  # (B..., 3, nv)
        return torch.matmul(Jr_inv, J_ang_local) * self.weight
