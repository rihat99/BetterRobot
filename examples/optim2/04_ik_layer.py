"""Inverse kinematics as a differentiable layer inside network training.

A small network maps a task code to a hand target. The target is solved by IK and the
loss is taken on the joint configuration, so gradients flow through the solver.

    uv run python examples/optim2/04_ik_layer.py
"""

from __future__ import annotations

import torch
from robot_descriptions import panda_description

import better_robot as br
from better_robot import optim2, terms
from better_robot.kinematics import forward_kinematics_state

FRAME = "body_panda_hand"


def main() -> None:
    torch.manual_seed(0)
    model = br.load(panda_description.URDF_PATH, dtype=torch.float64)
    frame_id = model.frame_id(FRAME)
    q_home = torch.tensor([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.02], dtype=torch.float64)
    home_pose = forward_kinematics_state(model, q_home[None]).frame_pose[0, frame_id]

    # Ground truth: each task code moves the hand by a fixed linear map of the code.
    codes = torch.randn(64, 4, dtype=torch.float64)
    true_map = 0.05 * torch.randn(4, 3, dtype=torch.float64)

    def solve(offset: torch.Tensor) -> torch.Tensor:
        """Joint configuration ``(B, nq)`` that puts the hand at ``home + offset``; differentiable in ``offset``."""
        target = torch.cat([home_pose[:3] + offset, home_pose[3:].expand(offset.shape[0], 4)], dim=-1)
        problem = terms.ik_problem(model, {FRAME: target}, q_home.expand(offset.shape[0], -1))
        return optim2.solve_layer(problem, optim2.LevenbergMarquardt(), iters=30).x["q"]

    q_teacher = solve(codes @ true_map).detach()
    network = torch.nn.Linear(4, 3, bias=False, dtype=torch.float64)
    torch.nn.init.zeros_(network.weight)
    optimizer = torch.optim.Adam(network.parameters(), lr=0.01)
    for step in range(61):
        optimizer.zero_grad()
        loss = (solve(network(codes)) - q_teacher).square().mean()
        loss.backward()
        optimizer.step()
        if step % 20 == 0:
            gap = float((network.weight.T - true_map).abs().max().detach())
            print(f"step {step:3d}: joint loss {float(loss):.2e}, largest error of the learned map {gap:.4f}")


if __name__ == "__main__":
    main()
