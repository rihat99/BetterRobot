"""Hard constraints: reach a point exactly while a link stays under a ceiling.

The cost is "stay close to the start pose". The hand position is an equality constraint and
the elbow height an inequality; both are ordinary terms, named when calling the solver.

    uv run python examples/05_hard_constraints.py
"""

from __future__ import annotations

import torch
from robot_descriptions import panda_description

import better_robot as br
from better_robot import optim, terms
from better_robot.kinematics import forward_kinematics_state

HAND, ELBOW = "body_panda_hand", "body_panda_link4"


def main() -> None:
    model = br.load(panda_description.URDF_PATH, dtype=torch.float64)
    hand, elbow = model.frame_id(HAND), model.frame_id(ELBOW)
    q_start = torch.tensor([[0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.02, 0.02]], dtype=torch.float64)
    q_start = q_start[:, : model.nq].expand(64, -1)
    start = forward_kinematics_state(model, q_start).frame_pose
    generator = torch.Generator().manual_seed(0)
    goal = start[:, hand, :3] + 0.15 * (torch.rand(64, 3, generator=generator, dtype=torch.float64) - 0.5)
    ceiling = start[:, elbow, 2:3] - 0.03  # the elbow has to come down by 3 cm

    problem = optim.Problem(
        {"q": q_start},
        manifolds={"q": terms.RobotManifold(model)},
        bounds={"q": (model.lower_pos_limit, model.upper_pos_limit)},
        data={"goal": goal, "ceiling": ceiling, "q_start": q_start},
    )
    problem.cached(terms.kinematic_state(model), name="state")
    problem.residual("stay", lambda q, q_start: q - q_start)
    problem.residual("reach", terms.frame_position(model, HAND, target="goal"))
    problem.residual("under", lambda state, ceiling: state.frame_pose[:, elbow, 2:3] - ceiling)

    soft = optim.solve(problem.with_(weights={"reach": 100.0, "under": 0.0}), optim.LevenbergMarquardt(), iters=50)
    hard = optim.solve_constrained(
        problem, optim.LevenbergMarquardt(), equalities=("reach",), inequalities=("under",), tolerance=1e-8
    )
    for label, result in (("penalty only, no ceiling", soft), ("hard constraints", hard)):
        pose = forward_kinematics_state(model, result.x["q"]).frame_pose
        reach = (pose[:, hand, :3] - goal).norm(dim=-1).max()
        over = (pose[:, elbow, 2:3] - ceiling).max()
        moved = (result.x["q"] - q_start).norm(dim=-1).mean()
        print(f"{label:25s}: worst reach error {float(reach):.1e} m, elbow above ceiling by at most "
              f"{float(over):+.1e} m, mean joint motion {float(moved):.3f} rad")
    print(f"rounds: {hard.history['violation'].shape[0]}, LM steps: {hard.iters}")


if __name__ == "__main__":
    main()
