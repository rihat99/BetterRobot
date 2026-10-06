"""Status-page benchmark: FK, frame Jacobian, RNEA, CRBA, ABA, solve_ik (BetterRobot) + pinocchio reference."""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

import better_robot as br
from better_robot.dynamics.aba import aba
from better_robot.dynamics.crba import crba
from better_robot.dynamics.rnea import rnea
from better_robot.kinematics.forward import forward_kinematics
from better_robot.kinematics.jacobian import compute_joint_jacobians, get_frame_jacobian
from better_robot.tasks import solve_ik


def load_model(name: str, device: str):
    # robot_descriptions is not installed in the project env; use its on-disk cache directly.
    from pathlib import Path

    cache = Path.home() / ".cache/robot_descriptions"
    path = (cache / "example-robot-data/robots/panda_description/urdf/panda.urdf" if name == "panda"
            else cache / "unitree_ros/robots/g1_description/g1_29dof.urdf")
    model = br.load(path, free_flyer=(name == "g1"), device=device, dtype=torch.float32)
    return model, path


def end_frame(model, name: str) -> str:
    candidates = ("body_panda_hand", "body_panda_link8") if name == "panda" else ()
    for candidate in candidates:
        if candidate in model.frame_name_to_id:
            return candidate
    return model.frame_names[-1]


def time_fn(fn, device: str, warmup: int, samples: int, budget_s: float) -> dict:
    for _ in range(warmup):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
    times = []
    t_start = time.perf_counter()
    while len(times) < samples and (time.perf_counter() - t_start) < budget_s or len(times) < 3:
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter_ns()
        fn()
        if device == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter_ns() - t0) / 1e6)
    return {"median_ms": statistics.median(times), "min_ms": min(times), "n": len(times)}


def random_q(model, batch: int, device: str) -> torch.Tensor:
    q = model.q_neutral.to(device=device, dtype=torch.float32).expand(batch, model.nq).clone()
    # Perturb only the joint-angle tail; free-flyer quaternion (if any) stays neutral.
    start = 7 if model.nq != model.nv else 0
    q[:, start:] += 0.3 * torch.randn(batch, model.nq - start, device=device)
    return q


def bench_algorithms(name: str, device: str, batches: list[int], args) -> list[dict]:
    model, _ = load_model(name, device)
    frame = end_frame(model, name)
    fid = model.frame_id(frame)
    rows = []
    for batch in batches:
        torch.manual_seed(0)
        q = random_q(model, batch, device)
        v = torch.randn(batch, model.nv, device=device)
        a = torch.randn(batch, model.nv, device=device)

        def jac():
            data = forward_kinematics(model, q, compute_frames=True)
            compute_joint_jacobians(model, data)
            return get_frame_jacobian(model, data, fid)

        algos = {
            "fk": lambda: forward_kinematics(model, q),
            "fk+frames": lambda: forward_kinematics(model, q, compute_frames=True),
            "frame_jacobian(fk+jj+get)": jac,
            "rnea": lambda: rnea(model, q, v, a),
            "crba": lambda: crba(model, q),
            "aba": lambda: aba(model, q, v, a),
        }
        for algo, fn in algos.items():
            row = {"lib": "better_robot", "model": name, "nq": model.nq, "nv": model.nv, "algo": algo,
                   "batch": batch, "device": device, "variant": "eager", "threads": torch.get_num_threads()}
            try:
                with torch.no_grad():
                    row.update(time_fn(fn, device, args.warmup, args.samples, args.budget))
                row["evals_per_s"] = batch / (row["median_ms"] / 1e3)
            except Exception as exc:  # report failures as rows, do not hide them
                row["error"] = f"{type(exc).__name__}: {exc}"[:300]
            print(json.dumps(row), flush=True)
            rows.append(row)
    return rows


def bench_ik(device: str, batches: list[int], args) -> list[dict]:
    model, _ = load_model("panda", device)
    frame = end_frame(model, "panda")
    rows = []
    for batch in batches:
        torch.manual_seed(1)
        q_goal = random_q(model, batch, device)
        data = forward_kinematics(model, q_goal, compute_frames=True)
        target = data.frame_pose_world[..., model.frame_id(frame), :] if hasattr(data, "frame_pose_world") else None
        if target is None:
            raise RuntimeError("no frame_pose_world on Data")
        target = target.detach()

        def solve():
            return solve_ik(model, {frame: target})

        row = {"lib": "better_robot", "model": "panda", "algo": "solve_ik(default LM, max_iter=100)",
               "batch": batch, "device": device, "variant": "eager", "threads": torch.get_num_threads()}
        try:
            row.update(time_fn(solve, device, 2, args.ik_samples, args.budget * 2))
            result = solve()
            iters = result.iters
            conv = result.converged
            row["iters_median"] = float(torch.as_tensor(iters).float().median())
            row["iters_max"] = float(torch.as_tensor(iters).float().max())
            row["converged_frac"] = float(torch.as_tensor(conv).float().mean())
            row["solves_per_s"] = batch / (row["median_ms"] / 1e3)
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"[:300]
        print(json.dumps(row), flush=True)
        rows.append(row)
    return rows


def bench_pinocchio(args) -> list[dict]:
    rows = []
    try:
        import numpy as np
        import pinocchio as pin
    except ImportError as exc:
        return [{"lib": "pinocchio", "error": f"not importable: {exc}"}]
    for name in ("panda", "g1"):
        _, path = load_model(name, "cpu")
        if name == "g1":
            pmodel = pin.buildModelFromUrdf(str(path), pin.JointModelFreeFlyer())
        else:
            pmodel = pin.buildModelFromUrdf(str(path))
        pdata = pmodel.createData()
        q = pin.randomConfiguration(pmodel, -np.ones(pmodel.nq), np.ones(pmodel.nq)) if name == "panda" else pin.neutral(pmodel)
        v = np.random.randn(pmodel.nv)
        a = np.random.randn(pmodel.nv)
        fid = pmodel.nframes - 1
        algos = {
            "fk": lambda: pin.forwardKinematics(pmodel, pdata, q),
            "fk+frames": lambda: pin.framesForwardKinematics(pmodel, pdata, q),
            "frame_jacobian(fk+jj+get)": lambda: pin.computeFrameJacobian(pmodel, pdata, q, fid, pin.LOCAL_WORLD_ALIGNED),
            "rnea": lambda: pin.rnea(pmodel, pdata, q, v, a),
            "crba": lambda: pin.crba(pmodel, pdata, q),
            "aba": lambda: pin.aba(pmodel, pdata, q, v, a),
        }
        for algo, fn in algos.items():
            for _ in range(200):
                fn()
            n = 5000
            t0 = time.perf_counter_ns()
            for _ in range(n):
                fn()
            per_call_ms = (time.perf_counter_ns() - t0) / 1e6 / n
            row = {"lib": "pinocchio " + pin.__version__, "model": name, "nq": pmodel.nq, "nv": pmodel.nv, "algo": algo,
                   "batch": 1, "device": "cpu", "variant": "python loop, mean of 5000 calls", "median_ms": per_call_ms,
                   "evals_per_s": 1e3 / per_call_ms}
            print(json.dumps(row), flush=True)
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 64, 4096])
    parser.add_argument("--ik-batches", type=int, nargs="+", default=[1, 1024])
    parser.add_argument("--models", nargs="+", default=["panda", "g1"])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--ik-samples", type=int, default=5)
    parser.add_argument("--budget", type=float, default=4.0)
    parser.add_argument("--pinocchio", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    env = {"torch": torch.__version__, "threads": torch.get_num_threads(), "device": args.device}
    try:
        import warp  # noqa: F401
        env["warp"] = warp.__version__
    except ImportError as exc:
        env["warp"] = f"not installed ({exc})"
    print(json.dumps(env), flush=True)
    rows = []
    for name in args.models:
        rows += bench_algorithms(name, args.device, args.batches, args)
    if args.ik_batches:
        rows += bench_ik(args.device, args.ik_batches, args)
    if args.pinocchio:
        rows += bench_pinocchio(args)
    with open(args.output, "w") as handle:
        json.dump({"env": env, "rows": rows}, handle, indent=1)


if __name__ == "__main__":
    main()
