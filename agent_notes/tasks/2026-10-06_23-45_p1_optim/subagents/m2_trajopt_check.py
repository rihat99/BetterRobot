"""M2 check: the Panda trajectory example on the new optim vs the stored result of the old solver."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from robot_descriptions import panda_description

import better_robot as br
from better_robot import optim2, terms
from better_robot.kinematics import forward_kinematics_state

FRAME, DT = "body_panda_hand", 0.05
GOLDEN = Path(__file__).parents[2] / "2026-10-06_21-40_p0_measure_and_fix/golden/trajopt_panda.npz"


def build(model, q_init, pose_start, pose_goal, analytic: bool = True) -> optim2.Problem:
    """Start and goal pose at the end knots, smooth acceleration in between, hard joint limits."""
    knots = q_init.shape[1]
    target = pose_start.expand(1, knots, 7).clone()
    target[:, -1] = pose_goal
    end_weight = torch.zeros(knots, 1, dtype=q_init.dtype)
    end_weight[0], end_weight[-1] = 1e6, 1e4
    problem = optim2.Problem(
        {"q": q_init},
        manifolds={"q": terms.RobotManifold(model)},
        bounds={"q": (model.lower_pos_limit, model.upper_pos_limit)},
        data={"target": target},
        time=("q",),
    )
    problem.cached(terms.kinematic_state(model), name="state")
    pose = terms.frame_pose(model, FRAME, target="target")
    accel = terms.finite_difference(model, order=2, dt=DT)
    if not analytic:
        pose = optim2.TermFn(fn=pose.fn, reads=pose.reads)
        accel = optim2.TermFn(fn=accel.fn, reads=accel.reads, window=3)
    problem.residual("ends", pose, weight=end_weight, window=1)
    problem.residual("accel", accel, weight=0.01)
    return problem


def describe(model, q, pose_start, pose_goal) -> dict:
    frames = forward_kinematics_state(model, q).frame_pose[:, model.frame_id(FRAME)]
    accel = (q[2:] - 2 * q[1:-1] + q[:-2]) / DT**2
    inside = bool(((q >= model.lower_pos_limit - 1e-9) & (q <= model.upper_pos_limit + 1e-9)).all())
    return {"start_error_m": float((frames[0, :3] - pose_start[:3]).norm()),
            "goal_error_m": float((frames[-1, :3] - pose_goal[:3]).norm()),
            "accel_cost": float(0.5 * 0.01 * accel.square().sum()), "inside_limits": inside}


def main() -> None:
    torch.set_num_threads(8)
    golden = np.load(GOLDEN)
    model = br.load(panda_description.URDF_PATH, dtype=torch.float64)
    q_init = torch.as_tensor(golden["q_init"]).unsqueeze(0)
    pose_start, pose_goal = torch.as_tensor(golden["pose_start"]), torch.as_tensor(golden["pose_goal"])
    out = {"old_solver": describe(model, torch.as_tensor(golden["q_opt"]), pose_start, pose_goal)}
    for label, analytic in (("new_lm_analytic", True), ("new_lm_autodiff", False)):
        problem = build(model, q_init, pose_start, pose_goal, analytic)
        if analytic:
            out["jacobian_check"] = optim2.check_jacobian(problem)
        start = time.perf_counter()
        result = optim2.solve(problem, optim2.LevenbergMarquardt(), iters=50)
        seconds = time.perf_counter() - start
        out[label] = {**describe(model, result.x["q"][0], pose_start, pose_goal), "iters": result.iters,
                      "seconds": round(seconds, 3), "loss": float(result.loss[0]), "done": bool(result.done[0])}
    batch = 64
    problem = build(model, q_init.expand(batch, -1, -1).clone(), pose_start, pose_goal)
    start = time.perf_counter()
    result = optim2.solve(problem, optim2.LevenbergMarquardt(), iters=50)
    out["new_lm_batch64"] = {"seconds": round(time.perf_counter() - start, 3), "iters": result.iters,
                             "loss_spread": float(result.loss.max() - result.loss.min())}
    Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
    for key, value in out.items():
        print(key, value)


if __name__ == "__main__":
    main()
