"""Play an optimized trajectory beside a plot of its joint torques.

The trajectory comes from the problem of ``examples/02_trajectory.py``; torques from
inverse dynamics on finite-difference velocities and accelerations. Each joint's drum takes the colour of its load
(``|tau|`` over the effort limit), and a see-through ghost of the robot stands where the motion ends.

    uv run python examples/viewer/02_play_trajectory.py
    uv run python examples/viewer/02_play_trajectory.py --check
"""

from __future__ import annotations

import argparse
import math
import torch
from robot_descriptions import panda_description

import better_robot as br
from better_robot import Viewer, forward_kinematics, inverse_dynamics, optim, terms
from better_robot.testing import FakeServer
from better_robot.viewer import RobotVisual

URDF = panda_description.URDF_PATH
FRAME, KNOTS, DT = "panda_hand", 60, 1.0 / 30.0
READY = [0.0, -math.pi / 4, 0.0, -3 * math.pi / 4, 0.0, math.pi / 2, math.pi / 4, 0.04]


def optimize_trajectory(model: br.Model) -> torch.Tensor:
    """``(KNOTS, nq)`` smooth motion from the ready pose to a shifted hand pose."""
    frame_id = model.frame_id(FRAME)
    q_start = torch.tensor(READY, dtype=torch.float64)
    pose_start = forward_kinematics(model, q_start[None]).frame_poses[0, frame_id]
    pose_goal = pose_start.clone()
    pose_goal[:3] += torch.tensor([0.10, 0.25, 0.20], dtype=torch.float64)
    target = pose_start.repeat(1, KNOTS, 1)
    target[:, -1] = pose_goal
    end_weight = torch.zeros(KNOTS, 1, dtype=torch.float64)
    end_weight[0], end_weight[-1] = 1e6, 1e4
    problem = optim.Problem(
        {"q": q_start.repeat(1, KNOTS, 1)},
        manifolds={"q": terms.RobotManifold(model)},
        bounds={"q": (model.lower_position_limit, model.upper_position_limit)},
        data={"target": target},
        time=("q",),
    )
    problem.cached(terms.kinematic_state(model), name="state")
    problem.residual("ends", terms.frame_pose(model, frame=FRAME, target="target"), weight=end_weight, window=1)
    problem.residual("velocity", terms.finite_difference(model, order=1, dt=DT), weight=1e-3)
    problem.residual("acceleration", terms.finite_difference(model, order=2, dt=DT), weight=1e-4)
    return optim.solve(problem, optim.LevenbergMarquardt(), iterations=200).x["q"][0]


def joint_torques(model: br.Model, q: torch.Tensor) -> torch.Tensor:
    """``(T, nv)`` inverse-dynamics torques along the clip (central differences, gravity on)."""
    v = torch.zeros(q.shape[0], model.nv, dtype=q.dtype)
    v[1:-1] = (q[2:] - q[:-2]) / (2 * DT)
    a = torch.zeros_like(v)
    a[1:-1] = (q[2:] - 2 * q[1:-1] + q[:-2]) / DT**2
    return inverse_dynamics(model, forward_kinematics(model, q), v=v, a=a)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check", action="store_true", help="build the scene on the stand-in server, push every frame and exit"
    )
    args = parser.parse_args()

    model = br.load(URDF, dtype=torch.float64)
    q = optimize_trajectory(model)
    tau = joint_torques(model, q)
    print(f"{KNOTS} frames, peak torque {float(tau.abs().max()):.1f} N m")

    load = torch.full((KNOTS, model.njoints), float("nan"), dtype=q.dtype)  # one value per joint: |tau| / effort
    joints = [j for j, nv in enumerate(model.nvs) if nv == 1]
    columns = [model.idx_vs[j] for j in joints]
    load[:, joints] = tau[:, columns].abs() / model.effort_limit[columns]

    viewer = Viewer(fps=1.0 / DT, server=FakeServer() if args.check else None)
    geometry = br.io.load_geometry(URDF, model)
    robot = viewer.add(
        RobotVisual(model, q, geometry=geometry, root=URDF, style="maximal", frames=[FRAME], joint_values=load,
                    value_range=(0.0, 0.3))  # orange at 30 % of the effort limit
    )  # fmt: skip
    viewer.add(RobotVisual(model, q[-1], name="goal", style="maximal", hue=robot.hue, opacity=0.35))  # a ghost
    moving = [name for name, nv in zip(model.joint_names, model.nvs) if nv]
    viewer.add_plot("torque", tau.numpy(), labels=moving)
    if args.check:
        for step in range(KNOTS):
            viewer.show(step)
        return
    print(f"playing at {viewer.url}")
    viewer.serve()


if __name__ == "__main__":
    main()
