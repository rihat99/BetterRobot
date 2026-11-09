"""Play an optimized trajectory with joints coloured by torque and a torque plot.

The trajectory comes from the problem of ``examples/02_trajectory.py``; torques from
inverse dynamics on finite-difference velocities and accelerations.

    uv run python examples/viewer/02_play_trajectory.py
    uv run python examples/viewer/02_play_trajectory.py --check
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from robot_descriptions import panda_description

import better_robot as br
from better_robot import Viewer, forward_kinematics, inverse_dynamics, optim, terms
from better_robot.testing import FakeServer
from better_robot.viewer import FramesLayer, GroundLayer, RigidMeshLayer, robot_meshes, robot_skeleton

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

    model = br.load(panda_description.URDF_PATH, dtype=torch.float64)
    q = optimize_trajectory(model)
    tau = joint_torques(model, q)
    print(f"{KNOTS} frames, peak torque {float(tau.abs().max()):.1f} N m")

    geometry = br.io.load_geometry(panda_description.URDF_PATH, model)
    state = forward_kinematics(model, q)
    poses = state.joint_poses.numpy()
    viewer = Viewer(fps=1.0 / DT, server=FakeServer() if args.check else None)
    viewer.add([GroundLayer()])
    meshes = robot_meshes(geometry, root=Path(panda_description.REPOSITORY_PATH))
    hand = FramesLayer("hand", state.frame_poses[:, [model.frame_id(FRAME)]].numpy())
    viewer.add(
        [RigidMeshLayer("panda", meshes, body_poses=poses), robot_skeleton(model, poses, tau=tau), hand], group="panda"
    )
    moving = [name for name, nv in zip(model.joint_names, model.nvs) if nv]
    viewer.add_plot("torque", tau.numpy(), labels=moving)
    if args.check:
        for step in range(KNOTS):
            viewer.show(step)
        return
    print(f"playing at {viewer.url}")
    viewer.play()


if __name__ == "__main__":
    main()
