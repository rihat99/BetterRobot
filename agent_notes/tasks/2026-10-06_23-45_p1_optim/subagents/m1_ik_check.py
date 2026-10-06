"""M1 check (c): new optim on the stored Panda IK problems vs the old solver's stored results."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from robot_descriptions import panda_description

import better_robot as br
from better_robot import optim2, terms
from better_robot.kinematics import forward_kinematics_state

FRAME = "body_panda_hand"
GOLDEN = Path(__file__).parents[2] / "2026-10-06_21-40_p0_measure_and_fix/golden/ik_panda.npz"


def errors(model, q, target):
    pose = forward_kinematics_state(model, q).frame_pose[:, model.frame_id(FRAME)]
    position = (pose[:, :3] - target[:, :3]).norm(dim=-1)
    rotation = 2.0 * torch.acos((pose[:, 3:] * target[:, 3:]).sum(-1).abs().clamp(max=1.0))
    return position, rotation


def summarize(position, rotation, rows):
    reached = (position[rows] < 1e-4) & (rotation[rows] < 1e-3)
    return {"reached_1e-4m_1e-3rad": float(reached.float().mean()),
            "position_median_m": float(position[rows].median()), "position_max_m": float(position[rows].max())}


def main() -> None:
    golden = np.load(GOLDEN)
    out = {}
    for dtype in (torch.float64, torch.float32):
        model = br.load(panda_description.URDF_PATH, dtype=dtype)
        target = torch.as_tensor(golden["target"], dtype=dtype)
        q0 = model.q_neutral.expand(target.shape[0], -1).clone()
        q0 = torch.minimum(torch.maximum(q0, model.lower_pos_limit), model.upper_pos_limit)
        torch.set_num_threads(8)
        for label, solver, restarts in (
            ("lm", optim2.LevenbergMarquardt(), 0),
            ("lm_8_restarts", optim2.LevenbergMarquardt(), 8),
            ("gauss_newton_damped", optim2.GaussNewton(damping=1e-3), 0),
        ):
            start = time.perf_counter()
            result = terms.solve_ik(model, {FRAME: target}, q0, solver=solver, iters=100, restarts=restarts,
                                    generator=torch.Generator().manual_seed(0))
            seconds = time.perf_counter() - start
            position, rotation = errors(model, result.x["q"], target)
            entry = {"seconds": round(seconds, 2), "iters": result.iters, "done": float(result.done.float().mean())}
            for name, rows in (("spread0.3", slice(0, 128)), ("spread1.0", slice(128, 256))):
                entry[name] = summarize(position, rotation, rows)
            out[f"new_{label}_{str(dtype).split('.')[-1]}"] = entry
    for name in ("default", "no_regularizer"):
        position = torch.as_tensor(golden[f"position_error_{name}"])
        rotation = torch.as_tensor(golden[f"rotation_error_{name}"])
        out[f"old_{name}_float64"] = {k: summarize(position, rotation, rows)
                                      for k, rows in (("spread0.3", slice(0, 128)), ("spread1.0", slice(128, 256)))}
    Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
    for key, value in out.items():
        print(key, json.dumps(value))


if __name__ == "__main__":
    main()
