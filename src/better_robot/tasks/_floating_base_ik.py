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


class _FloatingBaseIKModule(nn.Module):
    """LM-compatible module for whole-body IK with a floating base.

    Optimizes two parameters simultaneously:
    - self.base: pp.Parameter(pp.SE3) — base frame SE3 pose in world
    - self.cfg:  nn.Parameter          — joint angles

    The residual is: [pose_errors..., limit_violations, rest_deviation].
    """

    def __init__(
        self,
        robot: Robot,
        target_link_indices: list[int],
        target_poses: list[torch.Tensor],
        initial_base: torch.Tensor,
        initial_cfg: torch.Tensor,
        rest_cfg: torch.Tensor,
        pose_weight: float,
        limit_weight: float,
        rest_weight: float,
        pos_weight: float,
        ori_weight: float,
    ) -> None:
        super().__init__()
        self.base = pp.Parameter(pp.SE3(initial_base.float()))
        self.cfg = nn.Parameter(initial_cfg.float())
        self._robot = robot
        self._target_link_indices = target_link_indices
        self._target_poses = [tp.float() for tp in target_poses]
        self._rest_cfg = rest_cfg.float()
        self._pose_weight = pose_weight
        self._limit_weight = limit_weight
        self._rest_weight = rest_weight
        self._pos_weight = pos_weight
        self._ori_weight = ori_weight

    def forward(self, _input: torch.Tensor) -> torch.Tensor:
        fk = self._robot.forward_kinematics(self.cfg, base_pose=self.base.tensor())

        residuals = []

        for link_idx, target_pose in zip(self._target_link_indices, self._target_poses):
            actual_pose = fk[link_idx]
            T_err = se3_compose(se3_inverse(target_pose), actual_pose)
            log_err = se3_log(T_err)  # (6,) [tx, ty, tz, rx, ry, rz]
            weighted = torch.cat([
                log_err[:3] * self._pos_weight,
                log_err[3:] * self._ori_weight,
            ]) * self._pose_weight
            residuals.append(weighted)

        residuals.append(limit_residual(self.cfg, self._robot) * self._limit_weight)
        residuals.append(rest_residual(self.cfg, self._rest_cfg) * self._rest_weight)

        return torch.cat(residuals)


def solve_ik_floating_base(
    robot: Robot,
    targets: dict[str, torch.Tensor],
    initial_base_pose: torch.Tensor | None = None,
    initial_cfg: torch.Tensor | None = None,
    weights: dict[str, float] | None = None,
    max_iter: int = 100,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve whole-body IK with a floating base.

    Both joint angles and the robot base pose are optimized. Use this for
    humanoids or mobile manipulators where the root can move freely.

    Args:
        robot: Robot instance.
        targets: {link_name: (7,) SE3 target [tx, ty, tz, qx, qy, qz, qw]}.
        initial_base_pose: (7,) SE3 initial base pose. Defaults to identity
            (robot root at world origin).
        initial_cfg: (n,) initial joint config. Defaults to robot._default_cfg.
        weights: Cost weights. Keys: 'pose', 'limits', 'rest'.
            Defaults: {'pose': 1.0, 'limits': 0.1, 'rest': 0.01}.
        max_iter: Solver iterations.

    Returns:
        Tuple (base_pose, cfg):
            base_pose: (7,) optimized SE3 base pose.
            cfg: (num_actuated_joints,) optimized joint configuration.
    """
    w = {"pose": 1.0, "limits": 0.1, "rest": 0.01}
    if weights:
        w.update(weights)

    initial_base = (
        initial_base_pose.clone().float()
        if initial_base_pose is not None
        else se3_identity()
    )
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
        pose_weight=w["pose"],
        limit_weight=w["limits"],
        rest_weight=w["rest"],
        pos_weight=1.0,
        ori_weight=0.1,
    )

    strategy = ppo_strategy.Adaptive(damping=1e-4)
    optimizer = ppo.LevenbergMarquardt(module, strategy=strategy, vectorize=True)
    dummy = torch.zeros(1)

    for _ in range(max_iter):
        optimizer.step(input=dummy)
        with torch.no_grad():
            # Hard joint limit enforcement
            module.cfg.data.clamp_(
                lo.to(device=module.cfg.device),
                hi.to(device=module.cfg.device),
            )
            # Keep base quaternion unit-norm (indices 3:7 are qx,qy,qz,qw)
            q = module.base.data[3:7]
            module.base.data[3:7] = q / q.norm()

    return module.base.tensor().detach(), module.cfg.detach()
