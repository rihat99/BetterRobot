"""M1: analytic frame-pose Jacobian against autodiff on a float64 Panda (was ~1e-7 before the parser fix)."""

import torch
from robot_descriptions import panda_description

import better_robot as br
from better_robot import optim2, terms
from better_robot.kinematics import forward_kinematics_state

model = br.load(panda_description.URDF_PATH, dtype=torch.float64)
generator = torch.Generator().manual_seed(0)
q0 = model.q_neutral.clamp(model.lower_pos_limit, model.upper_pos_limit).expand(64, -1)
q = terms.random_start(model, q0, generator)
target = forward_kinematics_state(model, terms.random_start(model, q0, generator)).frame_pose[:, model.frame_id("body_panda_hand")]
problem = terms.ik_problem(model, {"body_panda_hand": target}, q)
print(optim2.check_jacobian(problem))
