"""Solve about one thousand independent IK targets in one batched call.

The example builds a tiny one-joint arm so the batching mechanics stay clear.
It defaults to CPU because a model this small is GPU launch-bound. CUDA and
automatic selection remain available for comparison, and timings are
synchronized on CUDA.

Usage::

    uv run python examples/03_batched_ik.py
    uv run python examples/03_batched_ik.py --batch-size 32 --device cpu
"""

from __future__ import annotations

import argparse
import math
import time

import torch

import better_robot as br
from better_robot.io import ModelBuilder, build_model
from better_robot.tasks import IKCostConfig, OptimizerConfig, solve_ik


def _arm(device: torch.device):
    builder = ModelBuilder("batched_ik_arm")
    base = builder.add_body("base", mass=0.5)
    link = builder.add_body("link", mass=1.0)
    identity = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    builder.add_revolute_z("joint", parent=base, child=link, origin=identity, lower=-math.pi, upper=math.pi)
    builder.add_frame("tip", parent_body=link, placement=identity + torch.tensor([0.5, 0.0, 0.0, 0, 0, 0, 0]))
    return build_model(builder.finalize(), device=device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="cpu",
        help="execution device; CPU is fastest for this tiny launch-bound model",
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda requested, but CUDA is unavailable")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    model = _arm(device)
    goal_q = torch.linspace(-0.6, 0.6, args.batch_size, device=device).unsqueeze(-1)
    target = br.forward_kinematics(model, goal_q, compute_frames=True).frame_pose_world[:, model.frame_id("tip")]

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    result = solve_ik(
        model,
        {"tip": target},
        initial_q=torch.zeros_like(goal_q),
        cost_cfg=IKCostConfig(limit_weight=0.0, rest_weight=0.0),
        optimizer_cfg=OptimizerConfig(max_iter=args.max_iterations),
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started

    if device.type == "cuda":
        print("Note: this one-joint example is launch-bound; CPU is usually faster.")
    print(f"Device: {device.type}")
    print(f"Problems: {args.batch_size}")
    print(f"Convergence rate: {float(result.converged.float().mean()):.1%}")
    print(f"Wall time: {elapsed * 1000.0:.1f} ms")


if __name__ == "__main__":
    main()
