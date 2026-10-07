"""Trajectory optimization: reach a goal pose smoothly, inside the joint limits.

The variable has a time axis; the terms are time-local, so the solver works on a
block-tridiagonal system and the cost grows linearly with the number of knots.

    uv run python examples/02_trajectory.py
"""

from __future__ import annotations

import math
import time

import torch
from robot_descriptions import panda_description

import better_robot as br
from better_robot import optim, terms
from better_robot.kinematics import forward_kinematics_state

FRAME, KNOTS, DT = "body_panda_hand", 40, 0.05
READY = [0.0, -math.pi / 4, 0.0, -3 * math.pi / 4, 0.0, math.pi / 2, math.pi / 4, 0.04]


def main() -> None:
    model = br.load(panda_description.URDF_PATH, dtype=torch.float64)
    frame_id = model.frame_id(FRAME)
    q_start = torch.tensor(READY, dtype=torch.float64).clamp(model.lower_pos_limit, model.upper_pos_limit)
    pose_start = forward_kinematics_state(model, q_start[None]).frame_pose[0, frame_id]
    pose_goal = pose_start.clone()
    pose_goal[:3] += torch.tensor([0.10, 0.15, 0.15], dtype=torch.float64)

    # Pose targets only at the first and the last knot: a weight per knot selects them.
    target = pose_start.repeat(1, KNOTS, 1)
    target[:, -1] = pose_goal
    end_weight = torch.zeros(KNOTS, 1, dtype=torch.float64)
    end_weight[0], end_weight[-1] = 1e6, 1e4

    problem = optim.Problem(
        {"q": q_start.repeat(1, KNOTS, 1)},
        manifolds={"q": terms.RobotManifold(model)},
        bounds={"q": (model.lower_pos_limit, model.upper_pos_limit)},
        data={"target": target},
        time=("q",),
    )
    problem.cached(terms.kinematic_state(model), name="state")
    problem.residual("ends", terms.frame_pose(model, FRAME, target="target"), weight=end_weight, window=1)
    problem.residual("velocity", terms.finite_difference(model, order=1, dt=DT), weight=1e-3)
    problem.residual("acceleration", terms.finite_difference(model, order=2, dt=DT), weight=1e-4)

    start = time.perf_counter()
    result = optim.solve(problem, optim.LevenbergMarquardt(), iters=200)
    seconds = time.perf_counter() - start
    q = result.x["q"][0]
    reached = forward_kinematics_state(model, q).frame_pose[:, frame_id]
    print(f"{result.iters} LM steps in {seconds * 1e3:.0f} ms, converged: {bool(result.done[0])}")
    print(f"start error {float((reached[0, :3] - pose_start[:3]).norm()):.1e} m, "
          f"goal error {float((reached[-1, :3] - pose_goal[:3]).norm()):.1e} m")
    print(f"peak joint speed {float(((q[1:] - q[:-1]) / DT).abs().max()):.2f} rad/s, "
          f"inside limits: {bool(((q >= model.lower_pos_limit) & (q <= model.upper_pos_limit)).all())}")


if __name__ == "__main__":
    main()
