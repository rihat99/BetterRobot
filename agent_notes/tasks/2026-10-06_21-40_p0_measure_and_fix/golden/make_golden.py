"""Record inputs and outputs of today's IK and trajectory optimization (commit in meta) for later comparison.

Problems are stored with their inputs so a new optimizer can solve the same ones; compare
achieved errors and costs, not the exact q.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
from robot_descriptions import panda_description

import better_robot as br
from better_robot.optim import LevenbergMarquardt
from better_robot.tasks import solve_ik
from better_robot.tasks.ik import IKCostConfig, OptimizerConfig
from better_robot.tasks.trajopt import solve_trajopt

FRAME = "body_panda_hand"


def pose_errors(reached: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Position error (m) and rotation angle error (rad) between ``(..., 7)`` poses."""
    position = (reached[..., :3] - target[..., :3]).norm(dim=-1)
    dot = (reached[..., 3:] * target[..., 3:]).sum(-1).abs().clamp(max=1.0)
    return position, 2.0 * torch.acos(dot)


def golden_ik(out: Path) -> dict:
    model = br.load(panda_description.URDF_PATH, dtype=torch.float64)
    torch.manual_seed(0)
    spread = torch.tensor([0.3, 1.0]).repeat_interleave(128)[:, None]
    q_goal = model.q_neutral + spread * torch.randn(256, model.nq, dtype=torch.float64)
    q_goal = torch.minimum(torch.maximum(q_goal, model.lower_pos_limit), model.upper_pos_limit)
    frame_id = model.frame_id(FRAME)
    target = br.forward_kinematics(model, q_goal, compute_frames=True).frame_pose_world[:, frame_id]
    summary = {}
    arrays = {"q_goal": q_goal.numpy(), "target": target.numpy(), "q_start": model.q_neutral.numpy()}
    configs = {
        "default": IKCostConfig(),
        "no_regularizer": IKCostConfig(limit_weight=0.0, rest_weight=0.0),
    }
    for name, cost_cfg in configs.items():
        start = time.perf_counter()
        result = solve_ik(model, {FRAME: target}, cost_cfg=cost_cfg, optimizer_cfg=OptimizerConfig(max_iter=100))
        seconds = time.perf_counter() - start
        position, rotation = pose_errors(result.frame_pose(FRAME), target)
        arrays[f"q_{name}"] = result.q.numpy()
        arrays[f"position_error_{name}"] = position.numpy()
        arrays[f"rotation_error_{name}"] = rotation.numpy()
        arrays[f"iters_{name}"] = result.iters.numpy()
        for label, rows in (("spread0.3", slice(0, 128)), ("spread1.0", slice(128, 256))):
            summary[f"{name}_{label}"] = {
                "converged_flag": float(result.converged[rows].float().mean()),
                "position_error_m_median": float(position[rows].median()),
                "position_error_m_max": float(position[rows].max()),
                "frac_below_1e-4_m": float((position[rows] < 1e-4).float().mean()),
                "rotation_error_rad_median": float(rotation[rows].median()),
                "iters_median": float(result.iters[rows].float().median()),
            }
        summary[f"{name}_seconds_cpu"] = round(seconds, 2)
    np.savez_compressed(out / "ik_panda.npz", **arrays)
    return summary


def golden_trajopt(out: Path, repo: Path) -> dict:
    spec = importlib.util.spec_from_file_location("example_trajopt", repo / "examples/05_panda_trajopt.py")
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    model = br.load(panda_description.URDF_PATH, dtype=torch.float64)
    q_start = torch.tensor(example.PANDA_READY, dtype=torch.float64).clamp(model.lower_pos_limit, model.upper_pos_limit)
    frame_id = model.frame_id(FRAME)
    pose_start = br.forward_kinematics(model, q_start, compute_frames=True).frame_pose_world[frame_id].clone()
    pose_goal = pose_start.clone()
    pose_goal[:3] += example.GOAL_OFFSET.to(pose_goal.dtype)
    goal = solve_ik(
        model,
        targets={FRAME: pose_goal},
        initial_q=q_start,
        cost_cfg=IKCostConfig(limit_weight=0.1, rest_weight=0.001),
        optimizer_cfg=OptimizerConfig(max_iter=50),
    )
    alpha = torch.linspace(0.0, 1.0, example.T, dtype=torch.float64).unsqueeze(1)
    q_init = q_start * (1.0 - alpha) + goal.q * alpha
    residuals = example.build_residuals(T_start=pose_start, T_goal=pose_goal, frame_id=frame_id)
    start = time.perf_counter()
    result = solve_trajopt(
        model,
        dt=example.DT,
        initial_q_traj=q_init,
        residuals=residuals,
        optimizer=partial(LevenbergMarquardt, max_iterations=50, tolerance=1e-7),
    )
    seconds = time.perf_counter() - start
    q_opt = result.trajectory.q.reshape(-1, model.nq)[-example.T :]
    frames = br.forward_kinematics(model, q_opt, compute_frames=True).frame_pose_world[:, frame_id]
    acceleration = (q_opt[2:] - 2 * q_opt[1:-1] + q_opt[:-2]) / example.DT**2
    np.savez_compressed(
        out / "trajopt_panda.npz",
        q_init=q_init.numpy(), q_opt=q_opt.numpy(), pose_start=pose_start.numpy(), pose_goal=pose_goal.numpy(),
        residual=result.residual.numpy(),
    )
    return {
        "iters": int(result.iters),
        "converged": bool(result.converged),
        "linearization_used": result.linearization_used,
        "whitened_residual_norm": float(result.residual.norm()),
        "start_position_error_m": float((frames[0, :3] - pose_start[:3]).norm()),
        "goal_position_error_m": float((frames[-1, :3] - pose_goal[:3]).norm()),
        "acceleration_rms": float(acceleration.pow(2).mean().sqrt()),
        "seconds_cpu": round(seconds, 2),
    }


if __name__ == "__main__":
    out = Path(__file__).parent
    repo = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())
    torch.set_num_threads(8)
    summary = {
        "meta": {
            "commit": subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip(),
            "torch": torch.__version__, "dtype": "float64", "device": "cpu", "threads": 8,
        },
        "ik_panda": golden_ik(out),
        "trajopt_panda": golden_trajopt(out, repo),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    json.dump(summary, sys.stdout, indent=1)
