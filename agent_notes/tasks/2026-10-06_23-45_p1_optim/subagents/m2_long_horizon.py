"""M2 scale check: G1 (floating base) follows noisy hand/foot/pelvis position tracks over a long clip.

Compares block-tridiagonal and dense normal equations: time per LM step, final loss.
"""

from __future__ import annotations

import json
import sys
import time

import torch
from robot_descriptions import g1_description

import better_robot as br
from better_robot import optim2, terms
from better_robot.kinematics import forward_kinematics_state


def build(model, q_init, tracks, frame_ids, dense: bool) -> optim2.Problem:
    problem = optim2.Problem({"q": q_init}, manifolds={"q": terms.RobotManifold(model)},
                             data={f"track{i}": track for i, track in enumerate(tracks)}, time=("q",), dense=dense)
    problem.cached(terms.kinematic_state(model), name="state")
    for i, frame_id in enumerate(frame_ids):
        problem.residual(f"track{i}", terms.frame_position(model, frame_id, target=f"track{i}"), window=1)
    problem.residual("accel", terms.finite_difference(model, order=2, dt=1 / 30), weight=1e-4)
    problem.residual("rest", lambda q: 0.03 * (q[..., 7:] - model.q_neutral[7:]), window=1)
    return problem


def main() -> None:
    torch.set_num_threads(8)
    model = br.load(g1_description.URDF_PATH, free_flyer=True, dtype=torch.float64)
    names = [n for n in model.frame_names if any(k in n for k in ("pelvis", "ankle_roll", "wrist_yaw", "head"))][:6]
    frame_ids = [model.frame_id(n) for n in names]
    out = {"frames": names, "nq": model.nq, "nv": model.nv}
    for knots in [int(k) for k in sys.argv[2].split(",")]:
        generator = torch.Generator().manual_seed(0)
        phase = torch.linspace(0, 6.28, knots, dtype=torch.float64)[:, None]
        q_ref = model.q_neutral.repeat(knots, 1)
        q_ref[:, 7:] += 0.4 * torch.sin(phase + torch.arange(model.nq - 7, dtype=torch.float64))
        q_ref[:, 0] += 0.5 * phase[:, 0] / 6.28
        state = forward_kinematics_state(model, q_ref.unsqueeze(0))
        tracks = [state.frame_pose[:, :, f, :3] + 0.005 * torch.randn(1, knots, 3, generator=generator, dtype=torch.float64)
                  for f in frame_ids]
        q_init = model.q_neutral.repeat(1, knots, 1)
        for label, dense in (("banded", False), ("dense", True)):
            if dense and knots * model.nv > 6000:
                continue
            problem = build(model, q_init, tracks, frame_ids, dense)
            start = time.perf_counter()
            normal, _ = problem.linearize()
            linearize_s = time.perf_counter() - start
            start = time.perf_counter()
            normal.solve(1e-3)
            solve_s = time.perf_counter() - start
            start = time.perf_counter()
            result = optim2.solve(problem, optim2.LevenbergMarquardt(), iters=30)
            total_s = time.perf_counter() - start
            track_error = torch.stack([(forward_kinematics_state(model, result.x["q"]).frame_pose[:, :, f, :3] - t).norm(dim=-1)
                                       for f, t in zip(frame_ids, tracks)]).mean()
            out[f"T{knots}_{label}"] = {"unknowns": knots * model.nv, "linearize_s": round(linearize_s, 3),
                                        "solve_s": round(solve_s, 4), "lm_total_s": round(total_s, 2),
                                        "iters": result.iters, "loss": float(result.loss[0]),
                                        "mean_track_error_m": float(track_error)}
            print(knots, label, out[f"T{knots}_{label}"], flush=True)
    json.dump(out, open(sys.argv[1], "w"), indent=1)


if __name__ == "__main__":
    main()
