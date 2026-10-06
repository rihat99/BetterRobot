"""Batched inverse kinematics with the new optimization layer.

Shows the three levels: the one-line recipe, the open problem with a custom term added,
and the same problem under a ``torch.optim`` optimizer.

    uv run python examples/optim2/01_ik.py
"""

from __future__ import annotations

import torch
from robot_descriptions import panda_description

import better_robot as br
from better_robot import optim2, terms
from better_robot.kinematics import forward_kinematics_state

FRAME = "body_panda_hand"


def position_error(model: br.Model, q: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Distance ``(B,)`` between the hand and the target position."""
    pose = forward_kinematics_state(model, q).frame_pose[:, model.frame_id(FRAME)]
    return (pose[:, :3] - target[:, :3]).norm(dim=-1)


def main() -> None:
    model = br.load(panda_description.URDF_PATH, dtype=torch.float64)
    generator = torch.Generator().manual_seed(0)
    q0 = model.q_neutral.clamp(model.lower_pos_limit, model.upper_pos_limit).expand(256, -1)
    q_goal = terms.random_start(model, q0, generator)
    target = forward_kinematics_state(model, q_goal).frame_pose[:, model.frame_id(FRAME)]

    # 1. Recipe: LM, hard joint limits, a few random restarts for the local minima.
    result = terms.solve_ik(model, {FRAME: target}, q0, restarts=8, generator=generator)
    error = position_error(model, result.x["q"], target)
    print(f"recipe      : reached (<0.1 mm) {float((error < 1e-4).float().mean()):.3f}, median {float(error.median()):.1e} m")

    # 2. Open problem: add a term that is just a function of named tensors. Its Jacobian is automatic.
    problem = terms.ik_problem(model, {FRAME: target}, q0)
    problem.data["preferred_elbow"] = torch.tensor(-1.8, dtype=torch.float64)
    problem.residual("elbow", lambda q, preferred_elbow: q[:, 3:4] - preferred_elbow, weight=1e-3)
    result = optim2.solve(problem, optim2.LevenbergMarquardt(), iters=100)
    print(f"custom term : per-term loss {{{', '.join(f'{k}: {float(v.mean()):.2e}' for k, v in problem.losses(result.x).items())}}}")

    # 3. The same problem under any torch.optim optimizer.
    result = optim2.solve(problem, optim2.TorchOptim(torch.optim.Adam, lr=0.05), iters=1500)
    error = position_error(model, result.x["q"], target)
    print(f"Adam        : median position error {float(error.median()):.1e} m after 1500 steps")


if __name__ == "__main__":
    main()
