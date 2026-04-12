"""Unified Jacobian entry points.

Pinocchio-style canonical functions — a single dispatch path that replaces
the legacy four-way fixed/floating × analytic/autodiff mess.

See ``docs/05_KINEMATICS.md §3``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch

from ..data_model.data import Data
from ..data_model.model import Model
from ..lie import se3, so3
from ..lie.tangents import hat_so3
from .jacobian_strategy import JacobianStrategy

if TYPE_CHECKING:
    from ..residuals.base import Residual, ResidualState

ReferenceFrame = Literal["world", "local", "local_world_aligned"]


def compute_joint_jacobians(model: Model, data: Data) -> Data:
    """Populate ``data.J`` with the spatial Jacobian of every joint.

    Shape: ``data.J = (B..., njoints, 6, nv)``. Requires ``data.oMi`` to be
    populated (call ``forward_kinematics`` first).

    Uses the propagation trick: ``J[j] = J[parent[j]]`` then adds the
    contribution of joint ``j`` itself.  This is equivalent to iterating
    over the support chain for every joint.

    See docs/05_KINEMATICS.md §3.
    """
    assert data.oMi is not None, "call forward_kinematics before compute_joint_jacobians"
    oMi = data.oMi  # (B..., njoints, 7)
    q = data.q
    *batch, _ = q.shape
    device, dtype = q.device, q.dtype

    J = torch.zeros(*batch, model.njoints, 6, model.nv, device=device, dtype=dtype)

    for j in model.topo_order:
        parent = model.parents[j]
        if parent >= 0:
            J[..., j, :, :] = J[..., parent, :, :]

        nv_j = model.nvs[j]
        v_j = model.idx_vs[j]

        if nv_j == 0:
            continue

        T_j = oMi[..., j, :]          # (B..., 7)
        p_j = T_j[..., :3]             # (B..., 3)
        R_j = so3.to_matrix(T_j[..., 3:])  # (B..., 3, 3)
        hat_p = hat_so3(p_j)            # (B..., 3, 3)

        nq_j = model.nqs[j]
        q_j = q[..., model.idx_qs[j] : model.idx_qs[j] + nq_j]
        S_local = model.joint_models[j].joint_motion_subspace(q_j)
        S_local = S_local.to(device=device, dtype=dtype)  # (B..., 6, nv_j)

        S_lin = S_local[..., :3, :]  # (B..., 3, nv_j)
        S_ang = S_local[..., 3:, :]  # (B..., 3, nv_j)

        # Ad(oMi[j]) @ S_local = [R @ S_lin + hat(p) @ R @ S_ang; R @ S_ang]
        R_S_lin = torch.matmul(R_j, S_lin)            # (B..., 3, nv_j)
        R_S_ang = torch.matmul(R_j, S_ang)            # (B..., 3, nv_j)
        hat_p_R_S_ang = torch.matmul(hat_p, R_S_ang)  # (B..., 3, nv_j)

        J[..., j, :3, v_j : v_j + nv_j] = R_S_lin + hat_p_R_S_ang
        J[..., j, 3:, v_j : v_j + nv_j] = R_S_ang

    data.J = J
    return data


def get_joint_jacobian(
    model: Model,
    data: Data,
    joint_id: int,
    *,
    reference: ReferenceFrame = "world",
) -> torch.Tensor:
    """Extract the spatial Jacobian of a single joint from ``data.J``.

    Shape: ``(B..., 6, nv)``. Reference frames mirror Pinocchio's
    ``ReferenceFrame`` enum.

    See docs/05_KINEMATICS.md §3.
    """
    if data.J is None:
        compute_joint_jacobians(model, data)

    J_j = data.J[..., joint_id, :, :]  # (B..., 6, nv)

    if reference == "world":
        return J_j
    elif reference == "local":
        T_j = data.oMi[..., joint_id, :]
        return torch.matmul(se3.adjoint_inv(T_j), J_j)
    else:
        raise ValueError(f"Unsupported reference frame: {reference!r}")


def get_frame_jacobian(
    model: Model,
    data: Data,
    frame_id: int,
    *,
    reference: ReferenceFrame = "world",
) -> torch.Tensor:
    """Spatial Jacobian of an arbitrary frame.

    Computes the Jacobian from ``data.oMi`` using the support chain, then
    adjusts for the frame's local placement offset.

    Shape: ``(B..., 6, nv)``.

    See docs/05_KINEMATICS.md §3.
    """
    assert data.oMi is not None, "call forward_kinematics before get_frame_jacobian"

    frame = model.frames[frame_id]
    parent_joint = frame.parent_joint
    q = data.q
    *batch, _ = q.shape
    device, dtype = q.device, q.dtype
    oMi = data.oMi

    # -- Build the parent joint's world-frame Jacobian --
    # Use data.J if already computed; otherwise compute directly.
    if data.J is not None:
        J_parent = data.J[..., parent_joint, :, :]  # (B..., 6, nv)
    else:
        J_parent = torch.zeros(*batch, 6, model.nv, device=device, dtype=dtype)
        support = model.get_support(parent_joint)
        for j in support:
            nv_j = model.nvs[j]
            v_j = model.idx_vs[j]
            if nv_j == 0:
                continue
            T_j = oMi[..., j, :]
            p_j = T_j[..., :3]
            R_j = so3.to_matrix(T_j[..., 3:])
            hat_p = hat_so3(p_j)

            nq_j = model.nqs[j]
            q_j = q[..., model.idx_qs[j] : model.idx_qs[j] + nq_j]
            S_local = model.joint_models[j].joint_motion_subspace(q_j)
            S_local = S_local.to(device=device, dtype=dtype)

            S_lin = S_local[..., :3, :]
            S_ang = S_local[..., 3:, :]

            R_S_ang = torch.matmul(R_j, S_ang)
            R_S_lin = torch.matmul(R_j, S_lin)
            hat_p_R_S_ang = torch.matmul(hat_p, R_S_ang)

            J_parent[..., :3, v_j : v_j + nv_j] = R_S_lin + hat_p_R_S_ang
            J_parent[..., 3:, v_j : v_j + nv_j] = R_S_ang

    # -- Adjust for frame offset from parent joint origin to frame position --
    # J_world[parent] has linear velocity at world origin: v_lin_origin = hat(p_i) @ a_world
    # Velocity at frame position p_f: v_at_pf = v_lin_origin + omega × p_f
    #   = J_lin + hat(J_ang) @ p_f = J_lin + (-hat(p_f)) @ J_ang (via cross product)
    # So: J_frame_lin = J_lin - hat(p_f) @ J_ang
    T_local = frame.joint_placement.to(device=device, dtype=dtype)
    T_parent = oMi[..., parent_joint, :]
    T_frame = se3.compose(T_parent, T_local)   # (B..., 7) world frame pose
    p_frame = T_frame[..., :3]                 # (B..., 3) world position

    hat_pf = hat_so3(p_frame)                  # (B..., 3, 3)
    J_lin = J_parent[..., :3, :] - torch.matmul(hat_pf, J_parent[..., 3:, :])
    J_frame_world = torch.cat([J_lin, J_parent[..., 3:, :]], dim=-2)  # (B..., 6, nv)

    if reference == "world":
        return J_frame_world
    elif reference == "local":
        # Body-frame Jacobian: J_local = Ad(T_frame^{-1}) @ J_world
        return torch.matmul(se3.adjoint_inv(T_frame), J_frame_world)
    elif reference == "local_world_aligned":
        # Linear in world frame, angular in world frame (same as world here)
        return J_frame_world
    else:
        raise ValueError(f"Unsupported reference frame: {reference!r}")


def residual_jacobian(
    residual: "Residual",
    state: "ResidualState",
    *,
    strategy: JacobianStrategy = JacobianStrategy.AUTO,
) -> torch.Tensor:
    """Unified residual Jacobian — ``(B..., dim, nv)``.

    Dispatches analytic / autodiff per-residual. This is the single place
    where analytic/autodiff coexist cleanly — the solver never writes
    Jacobian code itself.

    See docs/05_KINEMATICS.md §3.
    """
    from ..residuals.base import ResidualState as RS

    if strategy in (JacobianStrategy.ANALYTIC, JacobianStrategy.AUTO):
        try:
            J = residual.jacobian(state)
            if J is not None:
                return J
        except NotImplementedError:
            pass
        if strategy == JacobianStrategy.ANALYTIC:
            raise ValueError(
                f"Residual {residual.name!r} has no analytic Jacobian "
                f"(strategy=ANALYTIC requires one)"
            )

    # Autodiff: finite-difference through model.integrate and FK.
    # Note: torch.autograd.functional.jacobian through PyPose's Log() gives
    # incorrect gradients (factor-of-2 error in quaternion backward), so we
    # use central finite differences which are always correct.
    model = state.model
    q = state.variables

    def _fn(v: torch.Tensor) -> torch.Tensor:
        q_new = model.integrate(q.detach(), v)
        from .forward import forward_kinematics
        data_new = forward_kinematics(model, q_new, compute_frames=True)
        state_new = RS(model=model, data=data_new, variables=q_new)
        return residual(state_new)

    v0 = torch.zeros(model.nv, dtype=q.dtype, device=q.device)
    r0 = _fn(v0)
    dim = r0.numel()
    # Choose eps near sqrt(machine_epsilon) * characteristic_scale to balance
    # truncation vs cancellation error in float32/64.
    eps = 1e-3 if q.dtype == torch.float32 else 1e-7
    J = torch.zeros(dim, model.nv, dtype=q.dtype, device=q.device)
    for i in range(model.nv):
        v_p = v0.clone(); v_p[i] += eps
        v_m = v0.clone(); v_m[i] -= eps
        J[:, i] = (_fn(v_p) - _fn(v_m)) / (2.0 * eps)
    return J  # (dim, nv)
