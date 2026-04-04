"""Floating-base whole-body IK using PyPose LM."""

from __future__ import annotations

import torch
import torch.nn as nn
import pypose as pp
import pypose.optim as ppo
import pypose.optim.strategy as ppo_strategy

from ..core._robot import Robot
from ..core._lie_ops import se3_identity, se3_log, se3_compose, se3_inverse
from ..costs._limits import limit_residual
from ..costs._regularization import rest_residual
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


def _run_floating_base_lm(
    robot: Robot,
    targets: dict[str, torch.Tensor],
    ik_cfg: IKConfig,
    initial_cfg: torch.Tensor | None,
    initial_base_pose: torch.Tensor,
    max_iter: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Entry point called by solve_ik when initial_base_pose is not None."""
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
