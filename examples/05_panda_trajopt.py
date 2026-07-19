"""Manipulator trajectory optimisation — Panda start-to-goal with smoothness.

Pyroki-style trajopt demo (see ``reference_optim/pyroki/examples/07_trajopt.py``):
a Panda arm moves from a home pose to a shifted goal with
* pose constraints at the first and last timesteps,
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
from functools import partial
import math
import time

import torch

import better_robot as br
from better_robot.optim import LevenbergMarquardt
from better_robot.residuals import (
    AccelerationResidual,
    JointPositionLimit,
    PoseResidual,
)
from better_robot.tasks.trajopt import ResidualFactory, solve_trajopt

PANDA_READY = [
    0.0,
    -math.pi / 4,
    0.0,
    -3 * math.pi / 4,
    0.0,
    math.pi / 2,
    math.pi / 4,
    0.04,
]
EE_FRAME = "body_panda_hand"

T = 30  # timesteps
DT = 0.05  # 1.5 s total
GOAL_OFFSET = torch.tensor([0.10, 0.15, 0.15])  # move EE by this in world frame


def build_residuals(
    *,
    T_start: torch.Tensor,
    T_goal: torch.Tensor,
    frame_id: int,
) -> list[ResidualFactory]:
    residuals = [
        partial(
            PoseResidual,
            frame_id=frame_id,
            target=T_start,
            knot=0,
            weight=1000.0,
            name="start_pose",
        ),
        partial(
            PoseResidual,
            frame_id=frame_id,
            target=T_goal,
            knot=T - 1,
            weight=100.0,
            name="goal_pose",
        ),
        partial(
            AccelerationResidual,
            dt=DT,
            weight=0.1,
            name="accel",
        ),
    ]
    for t in range(T):
        name = f"limits_t{t}"
        residuals.append(
            partial(
                JointPositionLimit,
                knot=t,
                weight=10.0,
                name=name,
            )
        )
    return residuals


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-viewer", action="store_true")
    ap.add_argument("--fps", type=float, default=1.0 / DT)
    args = ap.parse_args()

    from robot_descriptions import panda_description  # noqa: PLC0415

    model = br.load(panda_description.URDF_PATH, dtype=torch.float64)

    # --- Define start / goal configurations ----------------------------------
    q_start = torch.tensor(PANDA_READY, dtype=torch.float64).clamp(model.lower_pos_limit, model.upper_pos_limit)
    data_start = br.forward_kinematics(model, q_start, compute_frames=True)
    frame_id = model.frame_id(EE_FRAME)
    T_start = data_start.frame_pose_world[frame_id].clone()

    # Goal pose: same orientation, translated by GOAL_OFFSET.
    T_goal = T_start.clone()
    T_goal[:3] += GOAL_OFFSET.to(T_goal.dtype)

    # IK for the goal configuration.
    from better_robot.tasks.ik import IKCostConfig, OptimizerConfig, solve_ik  # noqa: PLC0415

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
    q_init = q_start * (1.0 - alpha) + q_goal * alpha  # (T, nq)

    # --- Residuals -----------------------------------------------------------
    residuals = build_residuals(T_start=T_start, T_goal=T_goal, frame_id=frame_id)
    residual_dim = 2 * 6 + (T - 2) * model.nv + T * 2 * model.nq
    print(f"Residual dim: {residual_dim}  (vars: {T * model.nq})")

    # --- Solve ---------------------------------------------------------------
    t0 = time.perf_counter()
    result = solve_trajopt(
        model,
        dt=DT,
        initial_q_traj=q_init,
        residuals=residuals,
        optimizer=partial(LevenbergMarquardt, max_iterations=50, tolerance=1e-7),
    )
    solve_time = time.perf_counter() - t0
    print(
        f"Trajopt: iters={result.iters}  converged={result.converged}  "
        f"residual_norm={float(result.residual.norm()):.3e}  "
        f"time={solve_time * 1000:.1f} ms"
    )

    # --- Quality metrics ------------------------------------------------------
    q_opt = result.trajectory.q[0]  # (T, nq)
    ee_start = br.forward_kinematics(model, q_opt[0], compute_frames=True).frame_pose_world[frame_id]
    ee_end = br.forward_kinematics(model, q_opt[-1], compute_frames=True).frame_pose_world[frame_id]
    print(f"EE start pos error: {(ee_start[:3] - T_start[:3]).norm():.3e} m")
    print(f"EE goal  pos error: {(ee_end[:3] - T_goal[:3]).norm():.3e} m")

    if args.no_viewer:
        return

    # --- Viewer playback ------------------------------------------------------
    from better_robot.viewer import Visualizer  # noqa: PLC0415

    viewer = Visualizer(model, port=8080)
    player = viewer.add_trajectory(result.trajectory)
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
