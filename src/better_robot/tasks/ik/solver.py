"""Inverse kinematics — unified fixed-base and floating-base solver."""
from __future__ import annotations

import functools

import torch

from ...models.robot_model import RobotModel
from ...math.se3 import (
    se3_identity, se3_log, se3_compose, se3_inverse, se3_exp
)
from ...math.spatial import adjoint_se3
from ...costs.pose import pose_residual
from ...costs.limits import limit_residual
from ...costs.regularization import rest_residual
from ...costs.cost_term import CostTerm
from ...algorithms.kinematics.jacobian import (
    compute_jacobian, limit_jacobian, rest_jacobian,
)
from ...solvers.problem import Problem
from ...solvers.registry import SOLVERS
from .config import IKConfig
from .variable import IKVariable


def solve_ik(
    model: RobotModel,
    targets: dict[str, torch.Tensor],
    config: IKConfig | None = None,
    initial_q: torch.Tensor | None = None,
    initial_base_pose: torch.Tensor | None = None,
    max_iter: int = 100,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Solve inverse kinematics for one or more end-effector targets.

    Args:
        model: RobotModel instance.
        targets: {link_name: (7,) SE3 target [tx, ty, tz, qx, qy, qz, qw]}.
        config: IK configuration (weights, position/orientation balance).
        initial_q: (n,) starting joint config. Defaults to model._q_default.
        initial_base_pose: (7,) SE3 initial base pose. When provided, the base
            is also optimized (floating-base mode). Default None = fixed base.
        max_iter: Solver iterations.

    Returns:
        Fixed base: (num_actuated_joints,) optimized joint config tensor.
        Floating base: tuple (base_pose (7,), q (num_actuated_joints,)).
    """
    if config is None:
        config = IKConfig()
    if initial_base_pose is not None:
        return _solve_floating(model, targets, config, initial_q, initial_base_pose, max_iter)
    return _solve_fixed(model, targets, config, initial_q, max_iter)


# ---------------------------------------------------------------------------
# Fixed-base path
# ---------------------------------------------------------------------------

def _solve_fixed(
    model: RobotModel,
    targets: dict[str, torch.Tensor],
    config: IKConfig,
    initial_q: torch.Tensor | None,
    max_iter: int,
) -> torch.Tensor:
    initial = initial_q.clone() if initial_q is not None else model._q_default.clone()
    q_rest = model._q_default.clone()

    costs = []
    for link_name, target_pose in targets.items():
        link_idx = model.link_index(link_name)
        costs.append(CostTerm(
            residual_fn=functools.partial(
                pose_residual,
                robot=model,
                target_link_index=link_idx,
                target_pose=target_pose,
                pos_weight=config.pos_weight,
                ori_weight=config.ori_weight,
            ),
            weight=config.pose_weight,
        ))

    costs += [
        CostTerm(
            residual_fn=functools.partial(limit_residual, robot=model),
            weight=config.limit_weight,
        ),
        CostTerm(
            residual_fn=functools.partial(rest_residual, q_rest=q_rest),
            weight=config.rest_weight,
        ),
    ]

    jac_fn = None
    if config.jacobian == "analytic":
        jac_fn = _build_fixed_jacobian_fn(model, targets, config, q_rest)

    problem = Problem(
        variables=initial,
        costs=costs,
        lower_bounds=model.joints.lower_limits.clone(),
        upper_bounds=model.joints.upper_limits.clone(),
        jacobian_fn=jac_fn,
    )
    return SOLVERS.get(config.solver)(**config.solver_params).solve(problem, max_iter=max_iter)


def _build_fixed_jacobian_fn(
    model: RobotModel,
    targets: dict[str, torch.Tensor],
    config: IKConfig,
    q_rest: torch.Tensor,
):
    """Build analytical Jacobian for the full IK problem (fixed base)."""
    target_specs = [
        (model.link_index(name), pose) for name, pose in targets.items()
    ]

    def jacobian_fn(q: torch.Tensor) -> torch.Tensor:
        fk = model.forward_kinematics(q)
        rows = []
        for link_idx, target_pose in target_specs:
            J = compute_jacobian(
                model, q, link_idx, target_pose,
                config.pos_weight, config.ori_weight, fk=fk,
            )
            rows.append(J * config.pose_weight)
        rows.append(limit_jacobian(q, model) * config.limit_weight)
        rows.append(rest_jacobian(q, q_rest) * config.rest_weight)
        return torch.cat(rows, dim=0)

    return jacobian_fn


# ---------------------------------------------------------------------------
# Floating-base path (unified via IKVariable)
# ---------------------------------------------------------------------------

def _base_reg_residual(
    base: torch.Tensor,
    default_base: torch.Tensor,
    base_pos_weight: float,
    base_ori_weight: float,
) -> torch.Tensor:
    """6D residual pulling the base toward default_base."""
    T_err = se3_compose(se3_inverse(default_base), base)
    log_err = se3_log(T_err)
    return torch.cat([log_err[:3] * base_pos_weight, log_err[3:] * base_ori_weight])


def _fb_residual(
    q: torch.Tensor,
    base: torch.Tensor,
    model: RobotModel,
    target_link_indices: list[int],
    target_poses: list[torch.Tensor],
    config: IKConfig,
    q_rest: torch.Tensor,
    default_base: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute (fk, residual_vector) for floating-base IK."""
    fk = model.forward_kinematics(q, base_pose=base)
    parts = []
    for link_idx, tp in zip(target_link_indices, target_poses):
        T_err = se3_compose(se3_inverse(tp), fk[link_idx])
        log_e = se3_log(T_err)
        parts.append(
            torch.cat([log_e[:3] * config.pos_weight, log_e[3:] * config.ori_weight])
            * config.pose_weight
        )
    parts.append(limit_residual(q, model) * config.limit_weight)
    parts.append(rest_residual(q, q_rest) * config.rest_weight)
    parts.append(
        _base_reg_residual(base, default_base, config.base_pos_weight, config.base_ori_weight)
    )
    return fk, torch.cat(parts)


def _solve_floating(
    model: RobotModel,
    targets: dict[str, torch.Tensor],
    config: IKConfig,
    initial_q: torch.Tensor | None,
    initial_base_pose: torch.Tensor,
    max_iter: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unified floating-base IK using IKVariable and our LM solver.

    Variable: x = [q (n,), base_tangent (6,)]
    base_tangent is a se3 offset applied to current base via retraction.
    """
    if config.jacobian == "analytic":
        return _solve_floating_analytic(model, targets, config, initial_q, initial_base_pose, max_iter)
    return _solve_floating_autodiff(model, targets, config, initial_q, initial_base_pose, max_iter)


def _solve_floating_autodiff(
    model: RobotModel,
    targets: dict[str, torch.Tensor],
    config: IKConfig,
    initial_q: torch.Tensor | None,
    initial_base_pose: torch.Tensor,
    max_iter: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Floating-base IK with autodiff Jacobian, using IKVariable retraction."""
    base = initial_base_pose.clone().float()
    q = (
        initial_q.clone().float()
        if initial_q is not None
        else model._q_default.clone().float()
    )
    q_rest = model._q_default.clone().float()
    lo = model.joints.lower_limits.float()
    hi = model.joints.upper_limits.float()

    target_link_indices = [model.link_index(name) for name in targets]
    target_poses = [tp.float() for tp in targets.values()]
    default_base = base.clone()
    n = model.joints.num_actuated_joints
    lam = 1e-4
    factor = 2.0
    reject = 16

    for _ in range(max_iter):
        _, r = _fb_residual(q, base, model, target_link_indices, target_poses, config, q_rest, default_base)

        # Build residual as function of flat [q, base_tangent]
        def total_res_flat(x: torch.Tensor) -> torch.Tensor:
            q_x = x[:n]
            delta_base = x[n:]
            new_base = se3_compose(se3_exp(delta_base), base)
            quat = new_base[3:7]
            new_base = torch.cat([new_base[:3], quat / quat.norm().clamp(min=1e-8)])
            _, res = _fb_residual(q_x, new_base, model, target_link_indices, target_poses, config, q_rest, default_base)
            return res

        x0 = torch.cat([q, torch.zeros(6, dtype=q.dtype, device=q.device)])
        J = torch.func.jacrev(total_res_flat)(x0)
        JtJ = J.T @ J
        Jtr = J.T @ r

        for _ in range(reject):
            A = JtJ + lam * torch.eye(n + 6, dtype=q.dtype, device=q.device)
            delta = torch.linalg.solve(A, -Jtr)
            q_new = (q + delta[:n]).clamp(lo, hi)
            new_base = se3_compose(se3_exp(delta[n:]), base)
            quat = new_base[3:7]
            base_new = torch.cat([new_base[:3], quat / quat.norm().clamp(min=1e-8)])
            _, r_new = _fb_residual(q_new, base_new, model, target_link_indices, target_poses, config, q_rest, default_base)
            if r_new.norm() <= r.norm():
                q, base = q_new, base_new
                lam = max(lam / factor, 1e-7)
                break
            lam = min(lam * factor, 1e7)

    return base.detach(), q.detach()


def _solve_floating_analytic(
    model: RobotModel,
    targets: dict[str, torch.Tensor],
    config: IKConfig,
    initial_q: torch.Tensor | None,
    initial_base_pose: torch.Tensor,
    max_iter: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Floating-base IK with analytic Jacobian."""
    base = initial_base_pose.clone().float()
    default_base = initial_base_pose.clone().float()
    q = (
        initial_q.clone().float()
        if initial_q is not None
        else model._q_default.clone().float()
    )
    q_rest = model._q_default.clone().float()
    lo = model.joints.lower_limits.float()
    hi = model.joints.upper_limits.float()

    target_link_indices = [model.link_index(name) for name in targets]
    target_poses = [tp.float() for tp in targets.values()]

    n = model.joints.num_actuated_joints
    lam = 1e-4
    factor = 2.0
    reject = 16
    w_vec = q.new_tensor([config.pos_weight] * 3 + [config.ori_weight] * 3)
    w_base_vec = q.new_tensor(
        [config.base_pos_weight] * 3 + [config.base_ori_weight] * 3
    )

    for _ in range(max_iter):
        fk, r = _fb_residual(
            q, base, model, target_link_indices, target_poses, config, q_rest, default_base
        )

        J_rows = []
        for link_idx, target_pose in zip(target_link_indices, target_poses):
            J_joints = compute_jacobian(
                model, q, link_idx, target_pose,
                config.pos_weight, config.ori_weight, base_pose=base, fk=fk,
            ) * config.pose_weight
            T_ee_local = se3_compose(se3_inverse(base), fk[link_idx])
            Ad = adjoint_se3(se3_inverse(T_ee_local))
            J_base = w_vec.unsqueeze(1) * Ad * config.pose_weight
            J_rows.append(torch.cat([J_joints, J_base], dim=1))

        J_lim = torch.cat([
            limit_jacobian(q, model) * config.limit_weight,
            torch.zeros(2 * n, 6, dtype=q.dtype, device=q.device),
        ], dim=1)
        J_rest = torch.cat([
            rest_jacobian(q, q_rest) * config.rest_weight,
            torch.zeros(n, 6, dtype=q.dtype, device=q.device),
        ], dim=1)
        J_base_reg = torch.cat([
            torch.zeros(6, n, dtype=q.dtype, device=q.device),
            torch.diag(w_base_vec),
        ], dim=1)
        J = torch.cat(J_rows + [J_lim, J_rest, J_base_reg], dim=0)

        JtJ = J.T @ J
        Jtr = J.T @ r

        for _ in range(reject):
            A = JtJ + lam * torch.eye(n + 6, dtype=q.dtype, device=q.device)
            delta = torch.linalg.solve(A, -Jtr)

            q_new = (q + delta[:n]).clamp(lo, hi)
            base_new = se3_compose(se3_exp(delta[n:]), base)
            base_new = torch.cat([base_new[:3], base_new[3:7] / base_new[3:7].norm()])

            _, r_new = _fb_residual(
                q_new, base_new, model, target_link_indices, target_poses, config, q_rest,
                default_base,
            )
            if r_new.norm() <= r.norm():
                q, base = q_new, base_new
                lam = max(lam / factor, 1e-7)
                break
            lam = min(lam * factor, 1e7)

    return base.detach(), q.detach()
