"""Floating-base whole-body IK using PyPose LM."""

from __future__ import annotations

import torch
import torch.nn as nn
import pypose as pp
import pypose.optim as ppo
import pypose.optim.strategy as ppo_strategy

from ..core._robot import Robot
from ..core._lie_ops import se3_identity, se3_log, se3_compose, se3_inverse, se3_exp, adjoint_se3
from ..costs._limits import limit_residual
from ..costs._regularization import rest_residual
from ..costs._jacobian import pose_jacobian, limit_jacobian, rest_jacobian
from ._config import IKConfig


class _FloatingBaseIKModule(nn.Module):
    """LM-compatible module for whole-body IK with a floating base."""

    def __init__(
        self,
        robot: Robot,
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
        self._robot = robot
        self._target_link_indices = target_link_indices
        self._target_poses = [tp.float() for tp in target_poses]
        self._rest_cfg = rest_cfg.float()
        self._ik_cfg = ik_cfg

    def forward(self, _input: torch.Tensor) -> torch.Tensor:
        fk = self._robot.forward_kinematics(self.cfg, base_pose=self.base.tensor())

        residuals = []

        for link_idx, target_pose in zip(self._target_link_indices, self._target_poses):
            actual_pose = fk[..., link_idx, :]
            T_err = se3_compose(se3_inverse(target_pose), actual_pose)
            log_err = se3_log(T_err)  # (6,) [tx, ty, tz, rx, ry, rz]
            weighted = torch.cat([
                log_err[:3] * self._ik_cfg.pos_weight,
                log_err[3:] * self._ik_cfg.ori_weight,
            ]) * self._ik_cfg.pose_weight
            residuals.append(weighted)

        residuals.append(limit_residual(self.cfg, self._robot) * self._ik_cfg.limit_weight)
        residuals.append(rest_residual(self.cfg, self._rest_cfg) * self._ik_cfg.rest_weight)

        return torch.cat(residuals)


def _fb_residual(
    cfg: torch.Tensor,
    base: torch.Tensor,
    robot: Robot,
    target_link_indices: list[int],
    target_poses: list[torch.Tensor],
    ik_cfg: IKConfig,
    rest: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute (fk, residual_vector) for floating-base IK."""
    fk = robot.forward_kinematics(cfg, base_pose=base)
    parts = []
    for link_idx, tp in zip(target_link_indices, target_poses):
        T_err = se3_compose(se3_inverse(tp), fk[link_idx])
        log_e = se3_log(T_err)
        parts.append(
            torch.cat([log_e[:3] * ik_cfg.pos_weight, log_e[3:] * ik_cfg.ori_weight])
            * ik_cfg.pose_weight
        )
    parts.append(limit_residual(cfg, robot) * ik_cfg.limit_weight)
    parts.append(rest_residual(cfg, rest) * ik_cfg.rest_weight)
    return fk, torch.cat(parts)


def _run_floating_base_lm_analytic(
    robot: Robot,
    targets: dict[str, torch.Tensor],
    ik_cfg: IKConfig,
    initial_cfg: torch.Tensor | None,
    initial_base_pose: torch.Tensor,
    max_iter: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Floating-base LM using the analytical Jacobian.

    Variables: [cfg (n,) | base_tangent (6,)].
    Base updated via SE3 retraction: base_new = se3_exp(delta_base) @ base.
    """
    base = initial_base_pose.clone().float()
    cfg = (
        initial_cfg.clone().float()
        if initial_cfg is not None
        else robot._default_cfg.clone().float()
    )
    rest = robot._default_cfg.clone().float()
    lo = robot.joints.lower_limits.float()
    hi = robot.joints.upper_limits.float()

    target_link_indices = [robot.get_link_index(name) for name in targets]
    target_poses = [tp.float() for tp in targets.values()]

    n = robot.joints.num_actuated_joints
    lam = 1e-4
    factor = 2.0
    reject = 16
    w_vec = cfg.new_tensor([ik_cfg.pos_weight] * 3 + [ik_cfg.ori_weight] * 3)

    for _ in range(max_iter):
        fk, r = _fb_residual(cfg, base, robot, target_link_indices, target_poses, ik_cfg, rest)

        # Build Jacobian: (m, n+6) — joint cols first, then 6 base cols
        # Reuse fk already computed by _fb_residual — no extra FK call needed.
        J_rows = []
        for link_idx, target_pose in zip(target_link_indices, target_poses):
            J_joints = pose_jacobian(
                cfg, robot, link_idx, target_pose,
                ik_cfg.pos_weight, ik_cfg.ori_weight, base_pose=base, fk=fk,
            ) * ik_cfg.pose_weight                                             # (6, n)
            T_ee_local = se3_compose(se3_inverse(base), fk[link_idx])
            Ad = adjoint_se3(se3_inverse(T_ee_local))                          # (6, 6)
            J_base = w_vec.unsqueeze(1) * Ad * ik_cfg.pose_weight              # (6, 6)
            J_rows.append(torch.cat([J_joints, J_base], dim=1))                # (6, n+6)

        J_lim = torch.cat([
            limit_jacobian(cfg, robot) * ik_cfg.limit_weight,
            torch.zeros(2 * n, 6, dtype=cfg.dtype, device=cfg.device),
        ], dim=1)
        J_rest = torch.cat([
            rest_jacobian(cfg, rest) * ik_cfg.rest_weight,
            torch.zeros(n, 6, dtype=cfg.dtype, device=cfg.device),
        ], dim=1)
        J = torch.cat(J_rows + [J_lim, J_rest], dim=0)                        # (m, n+6)

        JtJ = J.T @ J
        Jtr = J.T @ r

        for _ in range(reject):
            A = JtJ + lam * torch.eye(n + 6, dtype=cfg.dtype, device=cfg.device)
            delta = torch.linalg.solve(A, -Jtr)

            cfg_new = (cfg + delta[:n]).clamp(lo, hi)
            base_new = se3_compose(se3_exp(delta[n:]), base)
            base_new = torch.cat([base_new[:3], base_new[3:7] / base_new[3:7].norm()])

            _, r_new = _fb_residual(
                cfg_new, base_new, robot, target_link_indices, target_poses, ik_cfg, rest
            )
            if r_new.norm() <= r.norm():
                cfg, base = cfg_new, base_new
                lam = max(lam / factor, 1e-7)
                break
            lam = min(lam * factor, 1e7)

    return base.detach(), cfg.detach()


def _run_floating_base_lm(
    robot: Robot,
    targets: dict[str, torch.Tensor],
    ik_cfg: IKConfig,
    initial_cfg: torch.Tensor | None,
    initial_base_pose: torch.Tensor,
    max_iter: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Entry point called by solve_ik when initial_base_pose is not None."""
    if ik_cfg.jacobian == "analytic":
        return _run_floating_base_lm_analytic(
            robot, targets, ik_cfg, initial_cfg, initial_base_pose, max_iter
        )

    initial_base = initial_base_pose.clone().float()
    initial_cfg_t = (
        initial_cfg.clone().float()
        if initial_cfg is not None
        else robot._default_cfg.clone().float()
    )
    rest_cfg = robot._default_cfg.clone().float()

    target_link_indices = [robot.get_link_index(name) for name in targets]
    target_poses = list(targets.values())

    lo = robot.joints.lower_limits.float()
    hi = robot.joints.upper_limits.float()

    module = _FloatingBaseIKModule(
        robot=robot,
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
