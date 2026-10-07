"""Docs screenshots with the offline rasterizer (no viser server)."""

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from robot_descriptions import g1_description, panda_description

import better_robot as br
from better_robot import optim, terms
from better_robot.kinematics import forward_kinematics_state
from better_robot.viewer import GroundLayer  # noqa: F401
from better_robot.viewer.render import frame_camera, render_frame
from better_robot.viewer.robot import robot_layers

OUT = Path(sys.argv[1])
W, H = 1200, 800
HAND = "body_panda_hand"


def shot(layers, size=(W, H), camera=None):
    camera = camera or frame_camera(layers)
    return render_frame(layers, 0, camera, size=size)


def save(image, name):
    Image.fromarray(np.asarray(image)).save(OUT / name)
    print("wrote", name, np.asarray(image).shape)


panda = br.load(panda_description.URDF_PATH, dtype=torch.float64)
q_rest = panda.q_neutral.clamp(panda.lower_pos_limit, panda.upper_pos_limit)
save(shot(robot_layers(panda, q_rest, frames=(HAND,))), "panda_rest.png")

# IK: before (rest) and after (target reached), same camera, side by side.
generator = torch.Generator().manual_seed(3)
q_goal = terms.random_start(panda, q_rest[None], generator)
target = forward_kinematics_state(panda, q_goal).frame_pose[:, panda.frame_id(HAND)]
result = terms.solve_ik(panda, {HAND: target}, q_rest[None], restarts=8, generator=generator)
q_ik = result.x["q"][0]
err = (forward_kinematics_state(panda, q_ik[None]).frame_pose[0, panda.frame_id(HAND)][:3] - target[0, :3]).norm()
print("ik position error", float(err))
before = robot_layers(panda, q_rest, frames=(HAND,))
after = robot_layers(panda, q_ik, frames=(HAND,))
camera = frame_camera(before + after)
save(np.concatenate([shot(before, (W // 2, H), camera), shot(after, (W // 2, H), camera)], axis=1), "panda_ik.png")

# Trajectory: the examples/02_trajectory.py problem, 5 knots side by side with one shared camera.
KNOTS, DT = 20, 0.05
READY = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04][: panda.nq]
fid = panda.frame_id(HAND)
q_start = torch.tensor(READY, dtype=torch.float64).clamp(panda.lower_pos_limit, panda.upper_pos_limit)
pose_start = forward_kinematics_state(panda, q_start[None]).frame_pose[0, fid]
pose_goal = pose_start.clone()
pose_goal[:3] += torch.tensor([0.10, 0.15, 0.15], dtype=torch.float64)
tgt = pose_start.repeat(1, KNOTS, 1)
tgt[:, -1] = pose_goal
end_weight = torch.zeros(KNOTS, 1, dtype=torch.float64)
end_weight[0], end_weight[-1] = 1e6, 1e4
problem = optim.Problem(
    {"q": q_start.repeat(1, KNOTS, 1)},
    manifolds={"q": terms.RobotManifold(panda)},
    bounds={"q": (panda.lower_pos_limit, panda.upper_pos_limit)},
    data={"target": tgt},
    time=("q",),
)
problem.cached(terms.kinematic_state(panda), name="state")
problem.residual("ends", terms.frame_pose(panda, HAND, target="target"), weight=end_weight, window=1)
problem.residual("velocity", terms.finite_difference(panda, order=1, dt=DT), weight=1e-3)
traj = optim.solve(problem, optim.LevenbergMarquardt(), iters=200).x["q"][0]
picks = [round(k) for k in np.linspace(0, KNOTS - 1, 5)]
views = [robot_layers(panda, traj[k], frames=(HAND,), name=f"k{k}") for k in picks]
camera = frame_camera([layer for view in views for layer in view])
tiles = [Image.fromarray(np.asarray(shot(view, (W, H), camera))[:, 360:840]).resize((240, 400)) for view in views]
save(np.concatenate([np.asarray(t) for t in tiles], axis=1), "trajectory_strip.png")

g1 = br.load(g1_description.URDF_PATH, free_flyer=True, dtype=torch.float64)
q_g1 = g1.q_neutral.clone()
q_g1[2] = 0.79
save(shot(robot_layers(g1, q_g1)), "g1_floating.png")
