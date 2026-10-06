"""M3 scale check: G1 tracking as in m2_long_horizon.py plus variables shared over the clip.

Shared: one 3D offset per tracked frame (a marker placed off the link origin), 18 unknowns.
Compares bordered-band and dense paths (loss, time). `offset_error_m` is printed but is not a
pass criterion: six tracked points do not pin down the offsets of a 35-DOF robot, and both
paths drift to the same wrong offsets, so this setup says nothing about recovery.
Usage: m3_shared_variable.py out.json 30,100,300
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
    offsets = q_init.new_zeros(1, len(frame_ids), 3)
    problem = optim2.Problem({"q": q_init, "offsets": offsets}, manifolds={"q": terms.RobotManifold(model)},
                             data={"tracks": tracks}, time=("q",), dense=dense)
    problem.cached(terms.kinematic_state(model), name="state")
    index = torch.tensor(frame_ids)

    def markers(state, offsets, tracks):
        pose = state.frame_pose[:, :, index]  # (B, T, F, 7)
        rotated = br.lie.so3.act(pose[..., 3:], offsets[:, None].expand(-1, pose.shape[1], -1, -1))
        return (pose[..., :3] + rotated - tracks).flatten(2)

    problem.residual("markers", markers, window=1)
    problem.residual("accel", terms.finite_difference(model, order=2, dt=1 / 30), weight=1e-4)
    problem.residual("rest", lambda q: 0.03 * (q[..., 7:] - model.q_neutral[7:]), window=1)
    problem.residual("offset_prior", lambda offsets: offsets.flatten(1), weight=1e-2)
    return problem


def main() -> None:
    torch.set_num_threads(8)
    model = br.load(g1_description.URDF_PATH, free_flyer=True, dtype=torch.float64)
    names = [n for n in model.frame_names if any(k in n for k in ("pelvis", "ankle_roll", "wrist_yaw", "head"))][:6]
    frame_ids = [model.frame_id(n) for n in names]
    generator = torch.Generator().manual_seed(0)
    true_offsets = 0.05 * torch.randn(len(frame_ids), 3, generator=generator, dtype=torch.float64)
    out = {"frames": names, "shared_unknowns": true_offsets.numel()}
    for knots in [int(k) for k in sys.argv[2].split(",")]:
        phase = torch.linspace(0, 6.28, knots, dtype=torch.float64)[:, None]
        q_ref = model.q_neutral.repeat(knots, 1)
        q_ref[:, 7:] += 0.4 * torch.sin(phase + torch.arange(model.nq - 7, dtype=torch.float64))
        q_ref[:, 0] += 0.5 * phase[:, 0] / 6.28
        pose = forward_kinematics_state(model, q_ref.unsqueeze(0)).frame_pose[:, :, frame_ids]
        tracks = pose[..., :3] + br.lie.so3.act(pose[..., 3:], true_offsets.expand(1, knots, -1, -1))
        tracks = tracks + 0.005 * torch.randn(tracks.shape, generator=generator, dtype=torch.float64)
        q_init = model.q_neutral.repeat(1, knots, 1)
        for label, dense in (("bordered", False), ("dense", True)):
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
            out[f"T{knots}_{label}"] = {
                "normal": type(normal).__name__, "unknowns": problem.tangent_dim, "linearize_s": round(linearize_s, 3),
                "solve_s": round(solve_s, 4), "lm_total_s": round(total_s, 2), "iters": result.iters,
                "loss": float(result.loss[0]),
                "offset_error_m": float((result.x["offsets"][0] - true_offsets).norm(dim=-1).mean()),
            }
            print(knots, label, out[f"T{knots}_{label}"], flush=True)
    json.dump(out, open(sys.argv[1], "w"), indent=1)


if __name__ == "__main__":
    main()
