"""Manipulator trajectory optimisation — Panda start-to-goal with smoothness.

Pyroki-style trajopt demo (see ``reference_optim/pyroki/examples/07_trajopt.py``):
a Panda arm moves from a home pose to a shifted goal with
* pose constraints at the first and last timesteps (via ``TimeIndexedResidual``),
* acceleration smoothness in tangent space,
* joint-position limit penalties at every step.

Usage::

    uv run python examples/05_panda_trajopt.py               # solve + viewer playback
    uv run python examples/05_panda_trajopt.py --no-viewer   # just solve

For floating-base robots, replace the initial-trajectory linear
interpolation with ``model.integrate(q_start, alpha * model.difference(q_start, q_goal))``.
"""

from __future__ import annotations

import argparse
import math
import time

import torch

import better_robot as br
from better_robot.costs.stack import CostStack
from better_robot.optim.optimizers.levenberg_marquardt import LevenbergMarquardt
from better_robot.residuals import (
    AccelerationResidual,
    JointPositionLimit,
    PoseResidual,
    TimeIndexedResidual,
)
from better_robot.tasks.trajopt import solve_trajopt

PANDA_READY = [
    0.0, -math.pi / 4, 0.0, -3 * math.pi / 4,
    0.0, math.pi / 2, math.pi / 4, 0.04, 0.04,
]
EE_FRAME = "body_panda_hand"

T = 30          # timesteps
DT = 0.05       # 1.5 s total
GOAL_OFFSET = torch.tensor([0.10, 0.15, 0.15])   # move EE by this in world frame


def build_cost_stack(
    model, *, T_start: torch.Tensor, T_goal: torch.Tensor, frame_id: int,
) -> CostStack:
    stack = CostStack()
    stack.add(
        "start_pose",
        TimeIndexedResidual(PoseResidual(frame_id=frame_id, target=T_start), t_idx=0),
        weight=1000.0,
    )
    stack.add(
        "goal_pose",
        TimeIndexedResidual(PoseResidual(frame_id=frame_id, target=T_goal), t_idx=T - 1),
        weight=100.0,
    )
    stack.add("accel", AccelerationResidual(model, dt=DT), weight=0.1)
    for t in range(T):
        stack.add(
            f"limits_t{t}",
            TimeIndexedResidual(JointPositionLimit(model), t_idx=t, name=f"limits_t{t}"),
            weight=10.0,
        )
    return stack


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-viewer", action="store_true")
    ap.add_argument("--fps", type=float, default=1.0 / DT)
    args = ap.parse_args()

    from robot_descriptions import panda_description

    model = br.load(panda_description.URDF_PATH, dtype=torch.float64)

    # --- Define start / goal configurations ----------------------------------
    q_start = torch.tensor(PANDA_READY, dtype=torch.float64).clamp(
        model.lower_pos_limit, model.upper_pos_limit
    )
    data_start = br.forward_kinematics(model, q_start, compute_frames=True)
    frame_id = model.frame_id(EE_FRAME)
    T_start = data_start.frame_pose_world[frame_id].clone()

    # Goal pose: same orientation, translated by GOAL_OFFSET.
    T_goal = T_start.clone()
    T_goal[:3] += GOAL_OFFSET.to(T_goal.dtype)

    # IK for the goal configuration.
    from better_robot.tasks.ik import IKCostConfig, OptimizerConfig, solve_ik
    goal_res = solve_ik(
        model,
        targets={EE_FRAME: T_goal},
        initial_q=q_start,
        cost_cfg=IKCostConfig(limit_weight=0.1, rest_weight=0.001),
        optimizer_cfg=OptimizerConfig(max_iter=50),
    )
    if not goal_res.converged:
        raise RuntimeError("IK for trajopt goal did not converge")
    q_goal = goal_res.q.to(torch.float64)

    # --- Initial trajectory: linear interpolation in config space ------------
    alpha = torch.linspace(0.0, 1.0, T, dtype=torch.float64).unsqueeze(1)
    q_init = q_start * (1.0 - alpha) + q_goal * alpha   # (T, nq)

    # --- Cost stack ----------------------------------------------------------
    stack = build_cost_stack(
        model, T_start=T_start, T_goal=T_goal, frame_id=frame_id
    )
    print(f"Cost stack dim: {stack.total_dim()}  (vars: {T * model.nq})")

    # --- Solve ---------------------------------------------------------------
    t0 = time.perf_counter()
    result = solve_trajopt(
        model,
        horizon=T,
        dt=DT,
        initial_q_traj=q_init,
        cost_stack=stack,
        optimizer=LevenbergMarquardt(tol=1e-7),
        max_iter=50,
    )
    solve_time = time.perf_counter() - t0
    print(
        f"Trajopt: iters={result.iters}  converged={result.converged}  "
        f"residual_norm={float(result.residual.norm()):.3e}  "
        f"time={solve_time * 1000:.1f} ms"
    )

    # --- Quality metrics ------------------------------------------------------
    q_opt = result.trajectory.q[0]    # (T, nq)
    ee_start = br.forward_kinematics(model, q_opt[0], compute_frames=True).frame_pose_world[frame_id]
    ee_end = br.forward_kinematics(model, q_opt[-1], compute_frames=True).frame_pose_world[frame_id]
    print(f"EE start pos error: {(ee_start[:3] - T_start[:3]).norm():.3e} m")
    print(f"EE goal  pos error: {(ee_end[:3] - T_goal[:3]).norm():.3e} m")

    if args.no_viewer:
        return

    # --- Viewer playback ------------------------------------------------------
    from better_robot.viewer import Visualizer

    # Viewer needs fp32 at the moment; downcast the trajectory for playback.
    model_f32 = br.load(panda_description.URDF_PATH)
    traj_f32 = type(result.trajectory)(
        t=result.trajectory.t.float(),
        q=result.trajectory.q.float(),
        v=None, a=None, u=None,
        model_id=getattr(model_f32, "id", -1),
    )

    viewer = Visualizer(model_f32, port=8080)
    player = viewer.add_trajectory(traj_f32)
    viewer.show(block=False)
    print(f"Viewer at http://localhost:8080 — playing {T} frames at {args.fps} fps. Ctrl-C to exit.")
    try:
        while True:
            player.play(fps=args.fps)
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nExit.")


if __name__ == "__main__":
    main()
