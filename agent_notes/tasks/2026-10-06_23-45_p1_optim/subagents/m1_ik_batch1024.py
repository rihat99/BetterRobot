"""M1: Panda IK, batch 1024, float32, CPU: new optim (LM, with and without restarts) vs the old solve_ik."""

from __future__ import annotations

import json
import sys
import time

import torch
from robot_descriptions import panda_description

import better_robot as br
from better_robot import terms
from better_robot.kinematics import forward_kinematics_state
from better_robot.tasks import solve_ik as old_solve_ik
from better_robot.tasks.ik import IKCostConfig

FRAME = "body_panda_hand"


def main() -> None:
    torch.set_num_threads(int(sys.argv[2]))
    model = br.load(panda_description.URDF_PATH)
    frame_id = model.frame_id(FRAME)
    out = {}
    for spread in (0.3, 1.0):
        torch.manual_seed(1)
        q_goal = model.q_neutral + spread * torch.randn(1024, model.nq)
        q_goal = torch.minimum(torch.maximum(q_goal, model.lower_pos_limit), model.upper_pos_limit)
        target = forward_kinematics_state(model, q_goal).frame_pose[:, frame_id]
        q0 = torch.minimum(torch.maximum(model.q_neutral, model.lower_pos_limit), model.upper_pos_limit).expand(1024, -1)

        def measure(label, solve):
            start = time.perf_counter()
            q = solve()
            seconds = time.perf_counter() - start
            pose = forward_kinematics_state(model, q).frame_pose[:, frame_id]
            position = (pose[:, :3] - target[:, :3]).norm(dim=-1)
            rotation = 2.0 * torch.acos((pose[:, 3:] * target[:, 3:]).sum(-1).abs().clamp(max=1.0))
            inside = ((q >= model.lower_pos_limit - 1e-6) & (q <= model.upper_pos_limit + 1e-6)).all(-1)
            out[f"{label}_spread{spread}"] = {
                "seconds": round(seconds, 2),
                "reached_1e-4m_1e-3rad": float(((position < 1e-4) & (rotation < 1e-3)).float().mean()),
                "inside_limits": float(inside.float().mean()),
                "position_median_m": float(position.median()),
            }
            print(label, spread, out[f"{label}_spread{spread}"], flush=True)

        measure("new_lm", lambda: terms.solve_ik(model, {FRAME: target}, q0).x["q"])
        measure("new_lm_4_restarts", lambda: terms.solve_ik(
            model, {FRAME: target}, q0, restarts=4, generator=torch.Generator().manual_seed(0)).x["q"])
        measure("old_default", lambda: old_solve_ik(model, {FRAME: target}).q)
        measure("old_no_regularizer", lambda: old_solve_ik(
            model, {FRAME: target}, cost_cfg=IKCostConfig(limit_weight=0.0, rest_weight=0.0)).q)
    json.dump(out, open(sys.argv[1], "w"), indent=1)


if __name__ == "__main__":
    main()
