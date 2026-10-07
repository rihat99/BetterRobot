import sys
from pathlib import Path
import numpy as np, torch
from PIL import Image
from robot_descriptions import panda_description
import better_robot as br
from better_robot import terms
from better_robot.kinematics import forward_kinematics_state
from better_robot.viewer.render import frame_camera, render_frame
from better_robot.viewer.robot import robot_layers
OUT = Path(sys.argv[1]); W, H = 1200, 800; HAND = "body_panda_hand"
panda = br.load(panda_description.URDF_PATH, dtype=torch.float64)
READY = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04][: panda.nq]
q_rest = torch.tensor(READY, dtype=torch.float64).clamp(panda.lower_pos_limit, panda.upper_pos_limit)
target = forward_kinematics_state(panda, q_rest[None]).frame_pose[:, panda.frame_id(HAND)].clone()
target[0, :3] += torch.tensor([0.20, 0.25, -0.10], dtype=torch.float64)
result = terms.solve_ik(panda, {HAND: target}, q_rest[None], restarts=8, generator=torch.Generator().manual_seed(3))
q_ik = result.x["q"][0]
err = (forward_kinematics_state(panda, q_ik[None]).frame_pose[0, panda.frame_id(HAND)][:3] - target[0, :3]).norm()
print("ik position error", float(err))
before = robot_layers(panda, q_rest, frames=(HAND,))
after = robot_layers(panda, q_ik, frames=(HAND,))
camera = frame_camera(before + after, margin=float(sys.argv[2]), azimuth=float(sys.argv[3]))
tiles = [render_frame(layers, 0, camera, size=(W // 2, H)) for layers in (before, after)]
Image.fromarray(np.concatenate(tiles, axis=1)).save(OUT / "panda_ik.png")
