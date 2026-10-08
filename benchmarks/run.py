"""Benchmark runner: wall time and torch operator counts for FK, Jacobian, RNEA, CRBA, ABA.

Usage (from the repo root)::

    CUDA_VISIBLE_DEVICES="" uv run python benchmarks/run.py --device cpu
    uv run python benchmarks/run.py --device cuda --batches 1,4096 --models panda,tree200 --algos fk,rnea

Writes ``benchmarks/results/<git short sha>_<device>.json`` (see ``--out``) and prints a text table.
Pinocchio CPU reference rows (batch 1, single thread) are added for Panda and G1 when pinocchio imports.
"""

from __future__ import annotations

import argparse
import datetime
import json
import platform
import statistics
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

import better_robot as br
from better_robot.kinematics import forward_kinematics_state, frame_jacobian
from better_robot.io.build_model import build_model
from better_robot.io.parsers.programmatic import ModelBuilder

REPO_ROOT = Path(__file__).resolve().parent.parent
DTYPE = torch.float32
ROW_BUDGET_S = 2.0
WARMUP = 1  # one warm-up call (plan caches); the profiled call is a second warm-up before timing
# Library Jacobian benchmarked: forward_kinematics_state + frame_jacobian(last frame, "local_world_aligned").
JACOBIAN_FN = "frame_jacobian(model, forward_kinematics_state(model, q), frame=nframes-1)"


def urdf_path(name: str) -> str:
    """Return the robot_descriptions URDF path for ``panda`` or ``g1``."""
    if name == "panda":
        from robot_descriptions import panda_description

        return panda_description.URDF_PATH
    from robot_descriptions import g1_description

    return g1_description.URDF_PATH


def random_tree(n_moving: int, seed: int = 0) -> br.Model:
    """Seeded fixed-base random tree of ``n_moving`` revolute joints; each parent is uniform over earlier bodies."""
    builder = ModelBuilder(f"tree{n_moving}")
    gen = torch.Generator().manual_seed(seed)

    def origin() -> torch.Tensor:
        return torch.cat([0.1 * torch.randn(3, generator=gen, dtype=torch.float64), torch.tensor([0.0, 0.0, 0.0, 1.0])])

    builder.add_body("b0", mass=1.0, inertia=0.01 * torch.eye(3, dtype=torch.float64))
    for i in range(1, n_moving + 1):
        parent = int(torch.randint(0, i, (1,), generator=gen))
        axis = torch.nn.functional.normalize(torch.randn(3, generator=gen, dtype=torch.float64), dim=0)
        builder.add_body(f"b{i}", mass=1.0, inertia=0.01 * torch.eye(3, dtype=torch.float64))
        builder.add_revolute(f"j{i}", parent=f"b{parent}", child=f"b{i}", axis=axis, origin=origin(), lower=-3.0, upper=3.0)
    return build_model(builder.finalize())


def load_model(name: str, device: str) -> br.Model:
    """Load ``panda`` / ``g1`` (free flyer) from URDF or build ``tree<N>``; float32 on ``device``."""
    if name in ("panda", "g1"):
        model = br.load(urdf_path(name), free_flyer=(name == "g1"), dtype=DTYPE)
    elif name.startswith("tree"):
        model = random_tree(int(name[4:]))
    else:
        raise ValueError(f"unknown model {name!r}")
    return model.to(dtype=DTYPE, device=device)


def make_inputs(model: br.Model, batch: int, device: str) -> dict[str, torch.Tensor]:
    """Seeded inputs: q (B, nq) tiled from 16 random configurations, v / a / tau (B, nv)."""
    gen = torch.Generator().manual_seed(1)
    base = torch.stack([model.to(device="cpu").random_configuration(gen) for _ in range(16)])
    q = base[torch.arange(batch) % 16].to(device=device, dtype=DTYPE)
    v, a, tau = (torch.randn(batch, model.nv, generator=gen).to(device=device, dtype=DTYPE) for _ in range(3))
    return {"q": q, "v": v, "a": a, "tau": tau}


def algo_fn(algo: str, model: br.Model, inputs: dict[str, torch.Tensor]) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return ``f(q) -> tensor`` for one library algorithm; v / a / tau are held fixed."""
    v, a, tau = inputs["v"], inputs["a"], inputs["tau"]
    if algo == "fk":
        return lambda q: forward_kinematics_state(model, q).joint_pose
    if algo == "jacobian":
        frame_id = model.nframes - 1
        return lambda q: frame_jacobian(model, forward_kinematics_state(model, q), frame=frame_id)
    if algo == "fk_state":
        return lambda q: forward_kinematics_state(model, q).joint_transform
    if algo == "jacobian_state":
        frame_id = model.nframes - 1
        return lambda q: frame_jacobian(model, forward_kinematics_state(model, q), frame=frame_id)
    if algo == "rnea":
        return lambda q: br.inverse_dynamics(model, forward_kinematics_state(model, q), v=v, a=a)
    if algo == "crba":
        return lambda q: br.mass_matrix(model, forward_kinematics_state(model, q))
    if algo == "aba":
        return lambda q: br.forward_dynamics(model, forward_kinematics_state(model, q), v=v, tau=tau)
    raise ValueError(f"unknown algo {algo!r}")


def run_once(fn: Callable[[torch.Tensor], torch.Tensor], q: torch.Tensor, backward: bool) -> None:
    """One call: forward under no_grad, or forward + backward of ``sum(output)`` w.r.t. ``q``."""
    if backward:
        qq = q.detach().requires_grad_(True)
        fn(qq).sum().backward()
    else:
        with torch.no_grad():
            fn(q)


def count_ops(fn: Callable[[torch.Tensor], torch.Tensor], q: torch.Tensor, backward: bool) -> int:
    """Count top-level ``aten::`` operator calls in one call (forward, plus backward when requested).

    Counted: every profiler CPU event named ``aten::*`` that has no ``aten::*`` ancestor, i.e. each operator
    dispatched from Python or from an autograd backward node. Ops nested inside another aten op (for example
    ``aten::bmm`` inside ``aten::matmul``) are not counted again. Same profiler-event method as
    proto_level_fk.py, but counting CPU-side operator dispatches instead of CUDA kernels.
    """

    def is_top_level(event) -> bool:
        parent = event.cpu_parent
        while parent is not None:
            if parent.name.startswith("aten::"):
                return False
            parent = parent.cpu_parent
        return True

    with profile(activities=[ProfilerActivity.CPU]) as prof:
        run_once(fn, q, backward)
    return sum(1 for e in prof.events() if e.name.startswith("aten::") and is_top_level(e))


def time_calls(call: Callable[[], None], max_repeats: int, sync: Callable[[], None], inner: int = 1) -> tuple[float, float]:
    """Return (median_ms, min_ms) per call; repeats shrink so the row takes about ``ROW_BUDGET_S``."""
    samples: list[float] = []
    budget_start = time.perf_counter()
    while len(samples) < max_repeats:
        sync()
        start = time.perf_counter()
        for _ in range(inner):
            call()
        sync()
        samples.append((time.perf_counter() - start) * 1e3 / inner)
        # at least 3 samples, except when one call alone exceeds the row budget (slow batch-4096 rows)
        if (len(samples) >= 3 or samples[0] > ROW_BUDGET_S * 1e3) and time.perf_counter() - budget_start > ROW_BUDGET_S:
            break
    return statistics.median(samples), min(samples)


def bench_torch_row(algo: str, model: br.Model, batch: int, mode: str, args: argparse.Namespace) -> dict:
    """Benchmark one (algo, model, batch, mode) row of the library."""
    inputs = make_inputs(model, batch, args.device)
    fn = algo_fn(algo, model, inputs)
    backward = mode == "forward_backward"
    sync = torch.cuda.synchronize if args.device == "cuda" else (lambda: None)
    for _ in range(WARMUP):
        run_once(fn, inputs["q"], backward)
    ops = count_ops(fn, inputs["q"], backward)
    median_ms, min_ms = time_calls(lambda: run_once(fn, inputs["q"], backward), args.repeats, sync)
    return {"median_ms": median_ms, "min_ms": min_ms, "ops": ops}


def pinocchio_rows(model_names: list[str], args: argparse.Namespace) -> list[dict]:
    """Pinocchio float64 CPU reference rows: batch 1, forward only, same URDFs."""
    import numpy as np
    import pinocchio as pin

    rows = []
    for name in (m for m in model_names if m in ("panda", "g1")):
        if name == "g1":
            pmodel = pin.buildModelFromUrdf(urdf_path(name), pin.JointModelFreeFlyer())
        else:
            pmodel = pin.buildModelFromUrdf(urdf_path(name))
        pdata = pmodel.createData()
        rng = np.random.default_rng(1)
        q = pin.neutral(pmodel)
        v, a, tau = (rng.standard_normal(pmodel.nv) for _ in range(3))
        calls = {
            "fk": lambda: pin.forwardKinematics(pmodel, pdata, q),
            "jacobian": lambda: pin.computeJointJacobians(pmodel, pdata, q),
            "rnea": lambda: pin.rnea(pmodel, pdata, q, v, a),
            "crba": lambda: pin.crba(pmodel, pdata, q),
            "aba": lambda: pin.aba(pmodel, pdata, q, v, tau),
        }
        for algo in args.algos:
            row = {"algo": algo, "model": name, "njoints": pmodel.njoints, "nq": pmodel.nq, "batch": 1,
                   "mode": "forward", "impl": "pinocchio", "ops": None}
            if algo == "jacobian":
                row["fn"] = "pin.computeJointJacobians"
            try:
                calls[algo]()
                row["median_ms"], row["min_ms"] = time_calls(calls[algo], max(args.repeats, 50), lambda: None, 200)
            except Exception as exc:  # noqa: BLE001 - record the failure in the row, keep benchmarking
                row["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
            print_row(row)
    return rows


def git_info() -> tuple[str, bool]:
    """Return (short sha, dirty flag) of the repo."""
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True)
    status = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True)
    return sha.stdout.strip() or "nogit", bool(status.stdout.strip())


def device_name(device: str) -> str:
    """Return a human-readable device name."""
    if device == "cuda":
        return torch.cuda.get_device_name()
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor()


def print_row(row: dict) -> None:
    """Print one compact table line."""
    if "error" in row:
        timing = f"ERROR {row['error'][:80]}"
    else:
        timing = f"{row['median_ms']:10.4f} {row['min_ms']:10.4f} {row['ops'] if row['ops'] is not None else '-':>7}"
    print(f"{row['algo']:9} {row['model']:8} {row['batch']:>5} {row['mode']:16} {row['impl']:10} {timing}", flush=True)


def main() -> None:
    """Parse arguments, run the grid, write JSON."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batches", default="1,64,256,4096")
    parser.add_argument("--models", default="panda,g1,tree25,tree200")
    parser.add_argument("--algos", default="fk,jacobian,rnea,crba,aba")
    parser.add_argument("--repeats", type=int, default=20, help="max timed repeats per row (fewer if row > 2 s)")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--out", default=None)
    parser.add_argument("--max-call-s", type=float, default=20.0,
                        help="skip larger batches of an (algo, model, mode) once one call exceeds this; row gets an error")
    args = parser.parse_args()
    args.batches = [int(b) for b in args.batches.split(",")]
    args.models = args.models.split(",")
    args.algos = args.algos.split(",")
    torch.set_num_threads(args.threads)

    sha, dirty = git_info()
    out = Path(args.out) if args.out else REPO_ROOT / "benchmarks" / "results" / f"{sha}_{args.device}.json"
    meta = {"commit": sha, "dirty": dirty, "torch": torch.__version__, "device": args.device,
            "device_name": device_name(args.device), "threads": args.threads, "dtype": str(DTYPE),
            "date": datetime.datetime.now().isoformat(timespec="seconds"), "jacobian_fn": JACOBIAN_FN}
    print(f"{'algo':9} {'model':8} {'batch':>5} {'mode':16} {'impl':10} {'median_ms':>10} {'min_ms':>10} {'ops':>7}")
    rows: list[dict] = []
    started = time.perf_counter()
    for model_name in args.models:
        try:
            model = load_model(model_name, args.device)
        except Exception as exc:  # noqa: BLE001 - record the failure in the row, keep benchmarking
            rows.append({"algo": "load", "model": model_name, "error": f"{type(exc).__name__}: {exc}"})
            print(f"load {model_name} ERROR {exc}")
            continue
        last_ms: dict[tuple[str, str], tuple[int, float]] = {}  # (algo, mode) -> (batch, median_ms) of this model
        for algo in args.algos:
            for batch in args.batches:
                for mode in ("forward", "forward_backward"):
                    row = {"algo": algo, "model": model_name, "njoints": model.njoints, "nq": model.nq,
                           "batch": batch, "mode": mode, "impl": "better_robot"}
                    if algo == "jacobian":
                        row["fn"] = "frame_jacobian"
                    previous = last_ms.get((algo, mode))
                    if previous is not None and previous[1] > args.max_call_s * 1e3:
                        row["error"] = f"skipped: batch {previous[0]} took {previous[1] / 1e3:.1f} s per call (> --max-call-s)"
                    else:
                        try:
                            row.update(bench_torch_row(algo, model, batch, mode, args))
                            last_ms[(algo, mode)] = (batch, row["median_ms"])
                        except Exception as exc:  # noqa: BLE001 - record the failure in the row, keep benchmarking
                            row["error"] = f"{type(exc).__name__}: {exc}"
                    rows.append(row)
                    print_row(row)
    if 1 in args.batches:
        try:
            rows.extend(pinocchio_rows(args.models, args))
        except ImportError as exc:
            print(f"pinocchio rows skipped: {exc}")
    meta["total_s"] = round(time.perf_counter() - started, 1)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"meta": meta, "rows": rows}, indent=1))
    print(f"wrote {out} ({meta['total_s']} s)")


if __name__ == "__main__":
    main()
