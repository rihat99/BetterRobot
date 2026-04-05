"""Inverse kinematics — unified fixed-base and floating-base solver."""
from __future__ import annotations

import functools

import torch
import torch.nn as nn
import pypose as pp
import pypose.optim as ppo
import pypose.optim.strategy as ppo_strategy

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


def solve_ik(
    model: RobotModel,
    targets: dict[str, torch.Tensor],
    cfg: IKConfig | None = None,
    initial_cfg: torch.Tensor | None = None,
    initial_base_pose: torch.Tensor | None = None,
    max_iter: int = 100,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Solve inverse kinematics for one or more end-effector targets.

    Args:
        model: RobotModel instance.
        targets: {link_name: (7,) SE3 target [tx, ty, tz, qx, qy, qz, qw]}.
        cfg: IK configuration (weights, position/orientation balance).
        initial_cfg: (n,) starting joint config. Defaults to model._default_cfg.
        initial_base_pose: (7,) SE3 initial base pose. When provided, the base
            is also optimized (floating-base mode). Default None = fixed base.
        max_iter: Solver iterations.

    Returns:
        Fixed base: (num_actuated_joints,) optimized joint config tensor.
        Floating base: tuple (base_pose (7,), cfg (num_actuated_joints,)).
    """
    if cfg is None:
        cfg = IKConfig()
    if initial_base_pose is not None:
        return _run_floating_base_lm(model, targets, cfg, initial_cfg, initial_base_pose, max_iter)
    return _solve_fixed(model, targets, cfg, initial_cfg, max_iter)


# ---------------------------------------------------------------------------
# Fixed-base path
# ---------------------------------------------------------------------------

def _solve_fixed(
    model: RobotModel,
    targets: dict[str, torch.Tensor],
    ik_cfg: IKConfig,
    initial_cfg: torch.Tensor | None,
    max_iter: int,
) -> torch.Tensor:
    initial = initial_cfg.clone() if initial_cfg is not None else model._default_cfg.clone()
    rest = model._default_cfg.clone()

    costs = []
    for link_name, target_pose in targets.items():
        link_idx = model.link_index(link_name)
        costs.append(CostTerm(
            residual_fn=functools.partial(
                pose_residual,
                robot=model,
                target_link_index=link_idx,
                target_pose=target_pose,
                pos_weight=ik_cfg.pos_weight,
                ori_weight=ik_cfg.ori_weight,
            ),
            weight=ik_cfg.pose_weight,
        ))

    costs += [
        CostTerm(
            residual_fn=functools.partial(limit_residual, robot=model),
            weight=ik_cfg.limit_weight,
        ),
        CostTerm(
            residual_fn=functools.partial(rest_residual, rest_pose=rest),
            weight=ik_cfg.rest_weight,
        ),
    ]

    jac_fn = None
    if ik_cfg.jacobian == "analytic":
        jac_fn = _build_fixed_jacobian_fn(model, targets, ik_cfg, rest)

    problem = Problem(
        variables=initial,
        costs=costs,
        lower_bounds=model.joints.lower_limits.clone(),
        upper_bounds=model.joints.upper_limits.clone(),
        jacobian_fn=jac_fn,
    )
    return SOLVERS.get(ik_cfg.solver)(**ik_cfg.solver_params).solve(problem, max_iter=max_iter)


def _build_fixed_jacobian_fn(
    model: RobotModel,
    targets: dict[str, torch.Tensor],
    ik_cfg: IKConfig,
    rest: torch.Tensor,
):
    """Build analytical Jacobian for the full IK problem (fixed base)."""
    target_specs = [
        (model.link_index(name), pose) for name, pose in targets.items()
    ]

    def jacobian_fn(cfg: torch.Tensor) -> torch.Tensor:
        fk = model.forward_kinematics(cfg)
        rows = []
        for link_idx, target_pose in target_specs:
            J = compute_jacobian(
                model, cfg, link_idx, target_pose,
                ik_cfg.pos_weight, ik_cfg.ori_weight, fk=fk,
            )
            rows.append(J * ik_cfg.pose_weight)
        rows.append(limit_jacobian(cfg, model) * ik_cfg.limit_weight)
        rows.append(rest_jacobian(cfg, rest) * ik_cfg.rest_weight)
        return torch.cat(rows, dim=0)

    return jacobian_fn


# ---------------------------------------------------------------------------
# Floating-base path
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


class _FloatingBaseIKModule(nn.Module):
    """LM-compatible module for whole-body IK with a floating base."""

    def __init__(
        self,
        model: RobotModel,
        target_link_indices: list[int],
        target_poses: list[torch.Tensor],
        initial_base: torch.Tensor,
        initial_cfg: torch.Tensor,
        rest_cfg: torch.Tensor,
        ik_cfg: IKConfig,
    ) -> None:
        super().__init__()
        self.base = pp.Parameter(pp.SE3(initial_base.float()))
        self.cfg = nn.Parameter(initial_cfg.float())
        self._model = model
        self._target_link_indices = target_link_indices
        self._target_poses = [tp.float() for tp in target_poses]
        self._rest_cfg = rest_cfg.float()
        self._ik_cfg = ik_cfg
        self.register_buffer("_default_base", initial_base.float().clone())

    def forward(self, _input: torch.Tensor) -> torch.Tensor:
        fk = self._model.forward_kinematics(self.cfg, base_pose=self.base.tensor())
        residuals = []

        for link_idx, target_pose in zip(self._target_link_indices, self._target_poses):
            actual_pose = fk[..., link_idx, :]
            T_err = se3_compose(se3_inverse(target_pose), actual_pose)
            log_err = se3_log(T_err)
            weighted = torch.cat([
                log_err[:3] * self._ik_cfg.pos_weight,
                log_err[3:] * self._ik_cfg.ori_weight,
            ]) * self._ik_cfg.pose_weight
            residuals.append(weighted)

        residuals.append(limit_residual(self.cfg, self._model) * self._ik_cfg.limit_weight)
        residuals.append(rest_residual(self.cfg, self._rest_cfg) * self._ik_cfg.rest_weight)
        residuals.append(
            _base_reg_residual(
                self.base.tensor(),
                self._default_base,
                self._ik_cfg.base_pos_weight,
                self._ik_cfg.base_ori_weight,
            )
        )
        return torch.cat(residuals)


def _fb_residual(
    cfg: torch.Tensor,
    base: torch.Tensor,
    model: RobotModel,
    target_link_indices: list[int],
    target_poses: list[torch.Tensor],
    ik_cfg: IKConfig,
    rest: torch.Tensor,
    default_base: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute (fk, residual_vector) for floating-base IK."""
    fk = model.forward_kinematics(cfg, base_pose=base)
    parts = []
    for link_idx, tp in zip(target_link_indices, target_poses):
        T_err = se3_compose(se3_inverse(tp), fk[link_idx])
        log_e = se3_log(T_err)
        parts.append(
            torch.cat([log_e[:3] * ik_cfg.pos_weight, log_e[3:] * ik_cfg.ori_weight])
            * ik_cfg.pose_weight
        )
    parts.append(limit_residual(cfg, model) * ik_cfg.limit_weight)
    parts.append(rest_residual(cfg, rest) * ik_cfg.rest_weight)
    parts.append(
        _base_reg_residual(base, default_base, ik_cfg.base_pos_weight, ik_cfg.base_ori_weight)
    )
    return fk, torch.cat(parts)


def _run_floating_base_lm_analytic(
    model: RobotModel,
    targets: dict[str, torch.Tensor],
    ik_cfg: IKConfig,
    initial_cfg: torch.Tensor | None,
    initial_base_pose: torch.Tensor,
    max_iter: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Floating-base LM using the analytical Jacobian."""
    base = initial_base_pose.clone().float()
    default_base = initial_base_pose.clone().float()
    cfg = (
        initial_cfg.clone().float()
        if initial_cfg is not None
        else model._default_cfg.clone().float()
    )
    rest = model._default_cfg.clone().float()
    lo = model.joints.lower_limits.float()
    hi = model.joints.upper_limits.float()

    target_link_indices = [model.link_index(name) for name in targets]
    target_poses = [tp.float() for tp in targets.values()]

    n = model.joints.num_actuated_joints
    lam = 1e-4
    factor = 2.0
    reject = 16
    w_vec = cfg.new_tensor([ik_cfg.pos_weight] * 3 + [ik_cfg.ori_weight] * 3)
    w_base_vec = cfg.new_tensor(
        [ik_cfg.base_pos_weight] * 3 + [ik_cfg.base_ori_weight] * 3
    )

    for _ in range(max_iter):
        fk, r = _fb_residual(
            cfg, base, model, target_link_indices, target_poses, ik_cfg, rest, default_base
        )

        J_rows = []
        for link_idx, target_pose in zip(target_link_indices, target_poses):
            J_joints = compute_jacobian(
                model, cfg, link_idx, target_pose,
                ik_cfg.pos_weight, ik_cfg.ori_weight, base_pose=base, fk=fk,
            ) * ik_cfg.pose_weight
            T_ee_local = se3_compose(se3_inverse(base), fk[link_idx])
            Ad = adjoint_se3(se3_inverse(T_ee_local))
            J_base = w_vec.unsqueeze(1) * Ad * ik_cfg.pose_weight
            J_rows.append(torch.cat([J_joints, J_base], dim=1))

        J_lim = torch.cat([
            limit_jacobian(cfg, model) * ik_cfg.limit_weight,
            torch.zeros(2 * n, 6, dtype=cfg.dtype, device=cfg.device),
        ], dim=1)
        J_rest = torch.cat([
            rest_jacobian(cfg, rest) * ik_cfg.rest_weight,
            torch.zeros(n, 6, dtype=cfg.dtype, device=cfg.device),
        ], dim=1)
        J_base_reg = torch.cat([
            torch.zeros(6, n, dtype=cfg.dtype, device=cfg.device),
            torch.diag(w_base_vec),
        ], dim=1)
        J = torch.cat(J_rows + [J_lim, J_rest, J_base_reg], dim=0)

        JtJ = J.T @ J
        Jtr = J.T @ r

        for _ in range(reject):
            A = JtJ + lam * torch.eye(n + 6, dtype=cfg.dtype, device=cfg.device)
            delta = torch.linalg.solve(A, -Jtr)

            cfg_new = (cfg + delta[:n]).clamp(lo, hi)
            base_new = se3_compose(se3_exp(delta[n:]), base)
            base_new = torch.cat([base_new[:3], base_new[3:7] / base_new[3:7].norm()])

            _, r_new = _fb_residual(
                cfg_new, base_new, model, target_link_indices, target_poses, ik_cfg, rest,
                default_base,
            )
            if r_new.norm() <= r.norm():
                cfg, base = cfg_new, base_new
                lam = max(lam / factor, 1e-7)
                break
            lam = min(lam * factor, 1e7)

    return base.detach(), cfg.detach()


def _run_floating_base_lm(
    model: RobotModel,
    targets: dict[str, torch.Tensor],
    ik_cfg: IKConfig,
    initial_cfg: torch.Tensor | None,
    initial_base_pose: torch.Tensor,
    max_iter: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Entry point for floating-base IK."""
    if ik_cfg.jacobian == "analytic":
        return _run_floating_base_lm_analytic(
            model, targets, ik_cfg, initial_cfg, initial_base_pose, max_iter
        )

    initial_base = initial_base_pose.clone().float()
    initial_cfg_t = (
        initial_cfg.clone().float()
        if initial_cfg is not None
        else model._default_cfg.clone().float()
    )
    rest_cfg = model._default_cfg.clone().float()

    target_link_indices = [model.link_index(name) for name in targets]
    target_poses = list(targets.values())

    lo = model.joints.lower_limits.float()
    hi = model.joints.upper_limits.float()

    module = _FloatingBaseIKModule(
        model=model,
        target_link_indices=target_link_indices,
        target_poses=target_poses,
        initial_base=initial_base,
        initial_cfg=initial_cfg_t,
        rest_cfg=rest_cfg,
        ik_cfg=ik_cfg,
    )

    strategy = ppo_strategy.Adaptive(damping=1e-4)
    optimizer = ppo.LevenbergMarquardt(module, strategy=strategy, vectorize=True)
    dummy = torch.zeros(1)

    for _ in range(max_iter):
        optimizer.step(input=dummy)
        with torch.no_grad():
            module.cfg.data.clamp_(
                lo.to(device=module.cfg.device),
                hi.to(device=module.cfg.device),
            )
            raw = module.base.tensor().clone()
            raw[3:7] = raw[3:7] / raw[3:7].norm()
            module.base.data.copy_(raw)

    return module.base.tensor().detach(), module.cfg.detach()
