"""Differentiate an IK solution back to its target pose.

The target pose is a graph-carrying static variable. ``differentiable=True``
asks the LM-backed task facade to attach its implicit backward, so a loss on
the solved configuration reaches target coordinates ``[x, y, z, qx, qy, qz,
qw]`` without unrolling optimizer iterations.

Usage::

    uv run python examples/06_differentiable_ik.py
"""

from __future__ import annotations

import argparse
import math

import torch

import better_robot as br
from better_robot.io import ModelBuilder, build_model
from better_robot.tasks import IKCostConfig, OptimizerConfig, solve_ik


def _arm():
    builder = ModelBuilder("differentiable_ik_arm")
    base = builder.add_body("base", mass=0.5)
    link = builder.add_body("link", mass=1.0)
    identity = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    builder.add_revolute_z("joint", parent=base, child=link, origin=identity, lower=-math.pi, upper=math.pi)
    builder.add_frame("tip", parent_body=link, placement=identity + torch.tensor([0.5, 0.0, 0.0, 0, 0, 0, 0]))
    return build_model(builder.finalize(), dtype=torch.float64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-iterations", type=int, default=50)
    args = parser.parse_args()

    model = _arm()
    initial_q = torch.tensor([-0.15], dtype=torch.float64)
    target_q = torch.tensor([0.10], dtype=torch.float64)
    target = (
        br.forward_kinematics(model, target_q, compute_frames=True)
        .frame_pose_world[model.frame_id("tip")]
        .detach()
        .requires_grad_()
    )
    result = solve_ik(
        model,
        {"tip": target},
        initial_q=initial_q,
        cost_cfg=IKCostConfig(limit_weight=0.0, rest_weight=0.01, q_rest=initial_q),
        optimizer_cfg=OptimizerConfig(max_iter=args.max_iterations, tol=1e-10),
        differentiable=True,
    )
    result.q.square().sum().backward()
    if target.grad is None:
        raise RuntimeError("the differentiable IK solve did not reach its target")

    print(f"Converged: {result.converged}")
    print(f"Solved configuration: {result.q.detach().tolist()}")
    print(f"Target-pose gradient: {target.grad.tolist()}")


if __name__ == "__main__":
    main()
