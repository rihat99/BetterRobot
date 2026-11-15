"""The speed suite: wall time of each function per robot, device, compile mode and batch, with reference columns.

Usage (from the repository root; on CPU pin one idle core)::

    taskset -c 5 uv run python benchmarks/speed.py run --device cpu --out cpu.json
    CUDA_VISIBLE_DEVICES=4 uv run python benchmarks/speed.py run --device cuda --out cuda.json
    uv run python benchmarks/speed.py table after.json --before before.json --reference cpu.json frax_cuda.json

``run`` times each of ``FUNCTIONS`` on the tracked Panda (fixed base) and G1 (free-flyer root, for every function)
of ``tests/fixtures/robots``, at each ``--batches`` size, in each ``--modes`` entry: ``eager``, ``compile``
(``torch.compile(fn)``) and ``reduce-overhead`` (``torch.compile(fn, mode="reduce-overhead")``, on CUDA by default).
``MODES`` is the one adapter from a mode to a callable; a compiled library entry point replaces it there. On CPU it
adds Pinocchio float64 rows at batch 1 (group ``test-oracles``), one call per sample.

A call includes what a user's call includes: ``forward_kinematics`` is ``forward_kinematics(model, q).joint_transforms``
(the state alone), ``joint_poses`` its ``joint_poses``, ``frame_jacobian`` the Jacobian of the first ``IK_FRAMES``
frame, ``joint_jacobians`` the state's ``jacobian_columns`` (every joint's Jacobian is a mask of it, as in
Pinocchio's ``computeJointJacobians``), ``inverse_dynamics``, ``mass_matrix``, ``forward_dynamics`` and
``center_of_mass`` that function of a fresh state; ``ik_step`` is one ``LevenbergMarquardt.step`` on ``ik_problem`` from ``q_neutral``; ``solve_ik`` the
default call (100 iterations, one start) and ``solve_ik_multistart`` five starts (``restarts=4``), each to the
``IK_FRAMES`` poses of seeded random configurations. IK rows record ``success_rate`` (every frame within 1 mm and
0.01 rad) and ``iterations``. Every call runs under ``torch.no_grad`` on seeded inputs (a pool of 1024 samples,
tiled); a compiled row starts from ``torch.compiler.reset()``, records its first call as ``compile_seconds``, then
warms up twice. Wall time is the median over at most ``--repeats`` samples (each sample one or more calls, at least
``SAMPLE_S`` long, between ``torch.cuda.synchronize`` calls), stopping once a row has ``ROW_BUDGET_S``. A failure
is recorded as ``error``; the larger batches of that function and mode are then recorded as ``skipped``, as are
those after a call above ``--max-call-s`` or a compile above ``--max-compile-s``.

The run re-executes itself with glibc's allocator thresholds fixed (``FIXED_MALLOC``; see the benchmark guide):
otherwise a large-batch time can double from page faults alone, depending on what the process freed before.

Row schema, shared by every row of a run and of a reference JSON (``{"meta": {...}, "rows": [...]}``):

    library           "better_robot", "pinocchio", "frax", ...
    function          one of FUNCTIONS
    robot             "panda" or "g1"
    device            "cpu" or "cuda"
    dtype             "float32" or "float64"
    batch             samples per call
    mode              "eager", "compile", "reduce-overhead"; a reference names its own ("jit", "vmap", "c++")
    seconds_per_call  median wall time of one call; absent when the row has "error" or "skipped" (text)
    compile_seconds   the first call of a compiled row (optional)

A run's rows add ``seconds_min``, ``seconds_spread`` (interquartile range over the median), ``samples``,
``calls_per_sample``, ``page_faults`` per call, ``loadavg``, on CUDA ``gpu_other_pids`` (other processes on the
GPU at the row's start or end: the row was timed on a shared GPU), on compiled rows ``graph_breaks``, ``graphs``,
``recompiles`` (frames compiled after the warm-up), ``cudagraph_skips`` and ``max_abs_diff_vs_eager``, on IK rows
``success_rate``, ``iterations``, ``median_position_error_m`` and ``median_rotation_error_rad``.

``table`` prints one markdown line per row of a run: microseconds per sample, compile seconds, the same row of
``--before`` and the speedup ``before / after``, and for each library of the ``--reference`` files its
microseconds per sample and the ratio ``ours / reference``. A reference row matches on function, robot, device and
batch; a Pinocchio row (a serial loop) matches every device and batch at its batch-1 time per sample.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import resource
import shlex
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import torch
from torch._dynamo.utils import counters as dynamo_counters

import better_robot as br
from better_robot.optim import LevenbergMarquardt, State

REPO_ROOT = Path(__file__).resolve().parent.parent
ROBOT_DIR = REPO_ROOT / "tests" / "fixtures" / "robots"
URDFS = {"panda": ROBOT_DIR / "panda" / "panda.urdf", "g1": ROBOT_DIR / "g1" / "g1_29dof.urdf"}
IK_FRAMES = {
    "panda": ("panda_hand",),
    "g1": ("left_rubber_hand", "right_rubber_hand", "left_ankle_roll_link", "right_ankle_roll_link"),
}
BATCHES = "1,64,1024,4096,16384,65536"
POOL = 1024
SAMPLE_S = 0.01
ROW_BUDGET_S = 1.0
IK_POSITION_M, IK_ROTATION_RAD = 1e-3, 1e-2
FIXED_MALLOC = {"MALLOC_MMAP_THRESHOLD_": "134217728", "MALLOC_TRIM_THRESHOLD_": "268435456"}
MODES: dict[str, Callable[[Callable[..., Any]], Callable[..., Any]]] = {
    "eager": lambda fn: fn,
    "compile": torch.compile,
    "reduce-overhead": partial(torch.compile, mode="reduce-overhead"),
}
KINEMATICS: dict[str, Callable[..., torch.Tensor]] = {
    "forward_kinematics": lambda model, frame, q, v, a: br.forward_kinematics(model, q).joint_transforms,
    "joint_poses": lambda model, frame, q, v, a: br.forward_kinematics(model, q).joint_poses,
    "frame_jacobian": lambda model, frame, q, v, a: br.frame_jacobian(
        model, br.forward_kinematics(model, q), frame=frame
    ),
    "joint_jacobians": lambda model, frame, q, v, a: br.forward_kinematics(model, q).jacobian_columns,
    "inverse_dynamics": lambda model, frame, q, v, a: br.inverse_dynamics(
        model, br.forward_kinematics(model, q), v=v, a=a
    ),
    "mass_matrix": lambda model, frame, q, v, a: br.mass_matrix(model, br.forward_kinematics(model, q)),
    "forward_dynamics": lambda model, frame, q, v, a: br.forward_dynamics(
        model, br.forward_kinematics(model, q), v=v, tau=a
    ),
    "center_of_mass": lambda model, frame, q, v, a: br.center_of_mass(model, br.forward_kinematics(model, q)),
}
FUNCTIONS = (*KINEMATICS, "ik_step", "solve_ik", "solve_ik_multistart")
SERIAL_REFERENCES = {"pinocchio"}  # one call per sample: its batch-1 time per sample holds at every batch


def load_model(robot: str, dtype: torch.dtype, device: torch.device) -> br.Model:
    """``panda`` (fixed base) or ``g1`` (free-flyer root) from the tracked URDFs."""
    root_joint = br.JointFreeFlyer() if robot == "g1" else None
    return br.load(URDFS[robot], root_joint=root_joint, dtype=dtype).to(device=device)


def ik_case(
    function: str, robot: str, model: br.Model, q_target: torch.Tensor
) -> tuple[Callable[..., Any], tuple[torch.Tensor, ...], Callable[[Any], dict[str, Any]] | None]:
    """``(fn, args, report)`` of an IK row toward the ``IK_FRAMES`` poses at ``q_target: (B, nq)``, from ``q_neutral``.

    ``fn(*args)`` returns ``(q, ...)``; ``report(output)`` gives a solve's accuracy and iterations.
    """
    frames, frame_ids = IK_FRAMES[robot], [model.frame_id(name) for name in IK_FRAMES[robot]]
    with torch.no_grad():
        reached = br.forward_kinematics(model, q_target)
        targets = reached.frame_transforms[:, frame_ids]  # (B, F, 3, 4)
        poses = tuple(reached.frame_poses[:, frame_ids].unbind(1))
    q0 = model.q_neutral.expand(q_target.shape[0], model.nq).clone()
    if function == "ik_step":
        problem = br.terms.ik_problem(model, dict(zip(frames, poses, strict=True)), q0=q0)
        solver = LevenbergMarquardt()
        start = solver.init(problem, {"q": q0})
        names = tuple(start.extra["terms"])

        def step(q: torch.Tensor, loss: torch.Tensor, done: torch.Tensor, *extra: torch.Tensor) -> tuple:
            damping, growth, *terms = extra
            state = State(
                x={"q": q},
                loss=loss,
                done=done,
                solver=LevenbergMarquardt,
                extra={"damping": damping, "growth": growth, "terms": dict(zip(names, terms, strict=True))},
            )
            new = solver.step(problem, state)
            return new.x["q"], new.loss, new.done, new.extra["damping"]

        extra = (start.extra["damping"], start.extra["growth"], *start.extra["terms"].values())
        return step, (q0, start.loss, start.done, *extra), None
    restarts = 4 if function == "solve_ik_multistart" else 0

    def solve(q0: torch.Tensor, *poses: torch.Tensor) -> tuple[torch.Tensor, int]:
        generator = torch.Generator(device=q0.device).manual_seed(1) if restarts else None
        targets = dict(zip(frames, poses, strict=True))
        result = br.solve_ik(model, targets, q0=q0, restarts=restarts, generator=generator)
        return result.x["q"], result.iterations

    def report(output: tuple[torch.Tensor, int]) -> dict[str, Any]:
        return ik_accuracy(robot, model, output[0], targets) | {"iterations": output[1]}

    return solve, (q0, *poses), report


def ik_accuracy(robot: str, model: br.Model, q: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    """Success rate and median worst-frame errors of ``q: (B, nq)`` against the IK frames' ``targets: (B, F, 3, 4)``."""
    frames = [model.frame_id(name) for name in IK_FRAMES[robot]]
    reached = br.forward_kinematics(model, q).frame_transforms[:, frames].double()
    targets = targets.double()
    position = (reached[..., 3] - targets[..., 3]).norm(dim=-1)  # (B, F)
    relative = targets[..., :3].transpose(-1, -2) @ reached[..., :3]
    skew = relative - relative.transpose(-1, -2)  # 2 sin(angle) [axis]x
    sine = torch.stack([skew[..., 2, 1], skew[..., 0, 2], skew[..., 1, 0]], dim=-1).norm(dim=-1) / 2
    cosine = (relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
    angle = torch.atan2(sine, cosine)
    success = ((position < IK_POSITION_M) & (angle < IK_ROTATION_RAD)).all(dim=-1)
    return {
        "success_rate": success.double().mean().item(),
        "median_position_error_m": position.amax(dim=-1).median().item(),
        "median_rotation_error_rad": angle.amax(dim=-1).median().item(),
    }


def time_call(call: Callable[[], object], sync: Callable[[], None], repeats: int) -> dict[str, Any]:
    """Median seconds per call over samples of ``calls_per_sample`` calls each (module docstring); call warmed up."""
    faults = resource.getrusage(resource.RUSAGE_SELF).ru_minflt
    sync()
    began = time.perf_counter()
    call()
    sync()
    probe = time.perf_counter() - began
    inner = max(1, int(SAMPLE_S / max(probe, 1e-7)))
    samples = [probe] if probe > ROW_BUDGET_S else []  # a slow call is its own one sample
    while len(samples) < repeats and not (
        samples and (len(samples) >= 3 or samples[0] > ROW_BUDGET_S) and time.perf_counter() - began > ROW_BUDGET_S
    ):
        sync()
        start = time.perf_counter()
        for _ in range(inner):
            call()
        sync()
        samples.append((time.perf_counter() - start) / inner)
    calls = 1 + (inner * len(samples) if probe <= ROW_BUDGET_S else 0)
    median = statistics.median(samples)
    spread = None
    if len(samples) >= 2:
        quartiles = statistics.quantiles(samples, n=4)
        spread = (quartiles[2] - quartiles[0]) / median
    page_faults = (resource.getrusage(resource.RUSAGE_SELF).ru_minflt - faults) / calls if calls else None
    return {
        "seconds_per_call": median,
        "seconds_min": min(samples),
        "seconds_spread": spread,
        "samples": len(samples),
        "calls_per_sample": inner,
        "page_faults": page_faults,
    }


def tensors(output: object) -> list[torch.Tensor]:
    """The tensors of a call's output, a tensor or a tuple."""
    items = output if isinstance(output, tuple) else (output,)
    return [item for item in items if isinstance(item, torch.Tensor)]


def measure(
    case: Callable[[], tuple],
    mode: str,
    device: torch.device,
    repeats: int,
    eager_output: list[torch.Tensor] | None,
) -> tuple[dict[str, Any], list[torch.Tensor]]:
    """Time ``MODES[mode](fn)(*args)`` of ``fn, args, report = case()`` (module docstring): the row's fields and the
    tensors of its output."""
    sync = (lambda: torch.cuda.synchronize(device)) if device.type == "cuda" else (lambda: None)
    fn, args, report = case()
    row: dict[str, Any] = {}
    compiled = mode != "eager"
    if compiled:
        torch.compiler.reset()
        dynamo_counters.clear()
    call = partial(MODES[mode](fn), *args)
    with torch.no_grad():
        sync()
        start = time.perf_counter()
        output = call()
        sync()
        if compiled:
            row["compile_seconds"] = time.perf_counter() - start
            if eager_output is not None:
                pairs = zip(tensors(output), eager_output, strict=True)
                row["max_abs_diff_vs_eager"] = max(float((a.double() - b.double()).abs().max()) for a, b in pairs)
            call()
            call()
            frames = dynamo_counters["frames"]["ok"]
        others = gpu_other_pids(device) if device.type == "cuda" else []
        row |= time_call(call, sync, repeats)
        if compiled:
            output = call()  # the last call's output: a CUDA graph overwrites its earlier outputs
    if compiled:
        row["graph_breaks"] = sum(dynamo_counters["graph_break"].values())
        row["graphs"] = dynamo_counters["stats"]["unique_graphs"]
        row["recompiles"] = dynamo_counters["frames"]["ok"] - frames
        row["cudagraph_skips"] = dynamo_counters["inductor"]["cudagraph_skips"]
    if device.type == "cuda":
        row["gpu_other_pids"] = sorted(set(others) | set(gpu_other_pids(device)))
    if report is not None:
        row |= report(output)
    return row, tensors(output)


def stop_reason(row: dict[str, Any], args: argparse.Namespace) -> str | None:
    """Why the larger batches of a row's function and mode are skipped, if they are."""
    if "error" in row:
        return f"batch {row['batch']} failed"
    if row["seconds_per_call"] > args.max_call_s:
        return f"batch {row['batch']} took {row['seconds_per_call']:.1f} s per call > --max-call-s"
    if row.get("compile_seconds", 0.0) > args.max_compile_s:
        return f"batch {row['batch']} compiled in {row['compile_seconds']:.0f} s > --max-compile-s"
    return None


def gpu_other_pids(device: torch.device) -> list[int]:
    """Processes other than this one with a CUDA context on ``device``, from ``nvidia-smi``."""
    uuid = f"GPU-{torch.cuda.get_device_properties(device).uuid}"
    command = ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"]
    listing = subprocess.run(command, capture_output=True, text=True, check=True).stdout
    apps = [line.split(", ") for line in listing.splitlines() if line.strip()]
    return sorted({int(pid) for gpu, pid in apps if gpu == uuid and int(pid) != os.getpid()})


def gpu_state(device: torch.device) -> dict[str, Any]:
    """Utilization, memory used and other processes of ``device``, from ``nvidia-smi``."""
    uuid = f"GPU-{torch.cuda.get_device_properties(device).uuid}"
    command = ["nvidia-smi", "--query-gpu=uuid,utilization.gpu,memory.used", "--format=csv,noheader,nounits"]
    listing = subprocess.run(command, capture_output=True, text=True, check=True).stdout
    (utilization, memory), *_ = [line.split(", ")[1:] for line in listing.splitlines() if line.startswith(uuid)]
    return {
        "utilization_percent": int(utilization),
        "memory_used_mib": int(memory),
        "other_pids": gpu_other_pids(device),
    }


def kinematics_case(
    function: str, robot: str, model: br.Model, q: torch.Tensor, v: torch.Tensor, a: torch.Tensor
) -> tuple[Callable[..., torch.Tensor], tuple[torch.Tensor, ...], None]:
    """``(fn, args, None)`` of a ``KINEMATICS`` row at ``q, v, a: (B, nq), (B, nv), (B, nv)``."""
    return partial(KINEMATICS[function], model, model.frame_id(IK_FRAMES[robot][0])), (q, v, a), None


def better_robot_rows(
    args: argparse.Namespace, device: torch.device, rows: list[dict[str, Any]], save: Callable[[], None]
) -> None:
    """Append the rows of ``--robots`` x ``--functions`` x ``--batches`` x ``--modes`` (module docstring), saving
    after each."""
    for robot in args.robots:
        model = load_model(robot, args.dtype, torch.device("cpu"))
        generator = torch.Generator().manual_seed(0)
        pool_q = torch.stack([model.random_configuration(generator=generator) for _ in range(POOL)])
        pool_v, pool_a = (torch.randn(POOL, model.nv, generator=generator, dtype=args.dtype) for _ in range(2))
        model = model.to(device=device)
        stopped: dict[tuple[str, str], str] = {}  # (function, mode) -> why its larger batches are skipped
        for function in args.functions:
            for batch in args.batches:
                index = torch.arange(batch) % POOL
                q, v, a = (pool[index].to(device) for pool in (pool_q, pool_v, pool_a))
                if function in KINEMATICS:
                    case = partial(kinematics_case, function, robot, model, q, v, a)
                else:
                    case = partial(ik_case, function, robot, model, q)
                eager_output = None
                for mode in args.modes:
                    row: dict[str, Any] = {"library": "better_robot", "function": function, "robot": robot}
                    row |= {"device": device.type, "dtype": str(args.dtype).removeprefix("torch."), "batch": batch}
                    row |= {"mode": mode, "loadavg": loadavg()}
                    if (function, mode) in stopped:
                        row["skipped"] = stopped[function, mode]
                    else:
                        try:
                            fields, output = measure(case, mode, device, args.repeats, eager_output)
                        except Exception as exc:  # noqa: BLE001 - a failure (out of memory, a compile error) is a row
                            fields, output = {"error": f"{type(exc).__name__}: {str(exc)[:400]}"}, None
                        row |= fields
                        eager_output = output if mode == "eager" else eager_output
                        reason = stop_reason(row, args)
                        if reason is not None:
                            stopped[function, mode] = reason
                    rows.append(row)
                    save()
                    print_row(row)
                eager_output = output = None  # free the batch's tensors before the next batch
                if device.type == "cuda":
                    torch.cuda.empty_cache()


def pinocchio_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Pinocchio float64 rows at batch 1 on the same URDFs: the reference cost per sample of a serial loop."""
    import numpy as np  # noqa: PLC0415 - with the optional Pinocchio oracle below
    import pinocchio as pin  # noqa: PLC0415 - the oracle of the test-oracles group, imported only for its rows

    rows = []
    for robot in args.robots:
        root = (pin.JointModelFreeFlyer(),) if robot == "g1" else ()
        model = pin.buildModelFromUrdf(str(URDFS[robot]), *root)
        data = model.createData()
        rng = np.random.default_rng(1)
        q = pin.neutral(model)
        v, a = (rng.standard_normal(model.nv) for _ in range(2))
        frame = model.getFrameId(IK_FRAMES[robot][0])
        calls = {
            "forward_kinematics": lambda: pin.forwardKinematics(model, data, q),
            "frame_jacobian": lambda: pin.computeFrameJacobian(model, data, q, frame, pin.LOCAL_WORLD_ALIGNED),
            "joint_jacobians": lambda: pin.computeJointJacobians(model, data, q),
            "inverse_dynamics": lambda: pin.rnea(model, data, q, v, a),
            "mass_matrix": lambda: pin.crba(model, data, q),
            "forward_dynamics": lambda: pin.aba(model, data, q, v, a),
            "center_of_mass": lambda: pin.centerOfMass(model, data, q),
        }
        for function in (name for name in args.functions if name in calls):
            row = {"library": "pinocchio", "function": function, "robot": robot, "device": "cpu", "dtype": "float64"}
            row |= {"batch": 1, "mode": "c++", "loadavg": loadavg()}
            calls[function]()
            rows.append(row | time_call(calls[function], lambda: None, args.repeats))
            print_row(rows[-1])
    return rows


def git(*arguments: str) -> str:
    """Stdout of one git command in the runner's checkout."""
    command = ["git", "-C", str(REPO_ROOT), *arguments]
    return subprocess.run(command, capture_output=True, text=True, check=True).stdout.strip()


def loadavg() -> str:
    """``/proc/loadavg``: the 1, 5 and 15 minute load averages, running/total tasks, last pid."""
    return Path("/proc/loadavg").read_text().strip()


def device_name(device: torch.device) -> str:
    """The GPU's name, or the CPU model from ``/proc/cpuinfo``."""
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor()


def print_row(row: dict[str, Any]) -> None:
    """One progress line: the row's key, microseconds per sample, compile seconds, IK success or the failure."""
    key = f"{row['library']}/{row['function']}/{row['robot']}/{row['device']}/B{row['batch']}/{row['mode']}"
    if "seconds_per_call" not in row:
        print(f"{key:58} {row.get('error') or row.get('skipped')}", flush=True)
        return
    line = f"{key:58} {1e6 * row['seconds_per_call'] / row['batch']:12.3f} us/sample"
    if "compile_seconds" in row:
        line += f"  compile {row['compile_seconds']:6.1f} s"
    if "success_rate" in row:
        line += f"  success {row['success_rate']:.3f}"
    print(line, flush=True)


def run(args: argparse.Namespace) -> None:
    """Measure the rows and write the JSON (module docstring)."""
    if any(os.environ.get(name) != value for name, value in FIXED_MALLOC.items()):
        os.execve(sys.executable, [sys.executable, *sys.argv], {**os.environ, **FIXED_MALLOC})
    if os.environ.get("BETTER_ROBOT_INTERNAL_CHECKS") == "1":
        sys.exit("BETTER_ROBOT_INTERNAL_CHECKS=1 adds host syncs and operators: unset it to benchmark")
    torch.set_num_threads(args.threads)  # noqa: TID251 - a benchmark pins its threads so rows compare; C-DEP-3 binds the library
    device = torch.device(args.device)
    meta: dict[str, Any] = {
        "commit": git("rev-parse", "HEAD"),
        "src_tree": git("rev-parse", "HEAD:src"),  # equal trees: the same library, whatever the benchmark's commit
        "dirty": bool(git("status", "--porcelain", "--", "src", "benchmarks")),
        "command": shlex.join(sys.argv),
        "package_file": os.path.relpath(br.__file__, REPO_ROOT),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "host": platform.node(),
        "device": args.device,
        "device_name": device_name(device),
        "dtype": str(args.dtype).removeprefix("torch."),
        "threads": torch.get_num_threads(),
        "affinity": sorted(os.sched_getaffinity(0)),
        "malloc_env": {name: os.environ.get(name) for name in FIXED_MALLOC},
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "loadavg_start": loadavg(),
    }
    if device.type == "cuda":
        meta["gpu_start"] = gpu_state(device)
    out = args.out or REPO_ROOT / "benchmarks" / "results" / f"speed_{meta['commit'][:7]}_{args.device}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    def save() -> None:
        out.write_text(json.dumps({"meta": meta, "rows": rows}, indent=1) + "\n")

    started = time.perf_counter()
    if device.type == "cpu" and 1 in args.batches:
        try:
            rows += pinocchio_rows(args)
        except ImportError as exc:
            meta["pinocchio"] = f"not measured: {exc}"
    better_robot_rows(args, device, rows, save)
    meta["loadavg_end"] = loadavg()
    if device.type == "cuda":
        meta["gpu_end"] = gpu_state(device)
    meta["total_s"] = round(time.perf_counter() - started, 1)
    save()
    print(f"wrote {out} ({meta['total_s']} s)")


def per_sample_us(row: dict[str, Any]) -> float | None:
    """Microseconds per sample of a timed row."""
    return 1e6 * row["seconds_per_call"] / row["batch"] if "seconds_per_call" in row else None


def table(args: argparse.Namespace) -> None:
    """Print the markdown table of a run against ``--before`` and the ``--reference`` libraries (module docstring)."""

    def key(row: dict[str, Any]) -> tuple:
        return row["function"], row["robot"], row["device"], row["dtype"], row["batch"], row["mode"]

    ours = [
        row
        for row in json.loads(args.run.read_text())["rows"]
        if row["library"] == "better_robot"
        and row["function"] in args.functions
        and row["robot"] in args.robots
        and row["batch"] in args.batches
        and row["mode"] in args.modes
    ]
    ours.sort(
        key=lambda row: (FUNCTIONS.index(row["function"]), row["robot"], list(MODES).index(row["mode"]), row["batch"])
    )
    before = {key(row): row for row in json.loads(args.before.read_text())["rows"]} if args.before else {}
    references: dict[str, list[dict[str, Any]]] = {}
    for path in args.reference:
        for row in json.loads(path.read_text())["rows"]:
            if row["library"] != "better_robot" and "seconds_per_call" in row:
                references.setdefault(row["library"], []).append(row)

    def reference(library: str, row: dict[str, Any]) -> dict[str, Any] | None:
        serial = library in SERIAL_REFERENCES
        matches = [
            candidate
            for candidate in references[library]
            if (candidate["function"], candidate["robot"]) == (row["function"], row["robot"])
            and (serial or (candidate["device"], candidate["batch"]) == (row["device"], row["batch"]))
        ]
        return min(matches, key=lambda candidate: candidate["dtype"] != row["dtype"], default=None)

    header = ["function", "robot", "device", "batch", "mode", "us/sample", "compile s"]
    header += ["before us/sample", "before/after"] if args.before else []
    for library in references:
        header += [f"{library} us/sample", f"ours/{library}"]
    header += ["note"]
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for row in ours:
        us = per_sample_us(row)
        cells = [row["function"], row["robot"], row["device"], str(row["batch"]), row["mode"], number(us)]
        cells.append(number(row.get("compile_seconds")))
        notes = [f"success {row['success_rate']:.3f}"] if "success_rate" in row else []
        notes += [str(row.get("error") or row.get("skipped") or "")[:60]]
        if row.get("gpu_other_pids"):
            notes.append("GPU shared")
        if args.before:
            old = before.get(key(row), {})
            old_us = per_sample_us(old) if old else None
            cells += [number(old_us), number(old_us / us if old_us and us else None, "x")]
            if "success_rate" in old:
                notes.append(f"before {old['success_rate']:.3f}")
        for library in references:
            match = reference(library, row)
            ref_us = 1e6 * match["seconds_per_call"] / match["batch"] if match else None
            cells += [number(ref_us), number(us / ref_us if ref_us and us else None, "x")]
        cells.append("; ".join(note for note in notes if note))
        print("| " + " | ".join(cells) + " |")


def number(value: float | None, suffix: str = "") -> str:
    """Three significant digits, or ``-``."""
    return "-" if value is None else f"{value:.3g}{suffix}"


def parse_args() -> argparse.Namespace:
    """The command line, with the comma lists split."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "table"):
        command = commands.add_parser(name)
        command.add_argument("--functions", default=",".join(FUNCTIONS))
        command.add_argument("--robots", default="panda,g1")
        command.add_argument("--batches", default=BATCHES)
        command.add_argument("--modes", default=None, help="default: eager,compile on cpu, and reduce-overhead on cuda")
    run_parser, table_parser = commands.choices["run"], commands.choices["table"]
    run_parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    run_parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    run_parser.add_argument("--repeats", type=int, default=15, help="most samples per row")
    run_parser.add_argument("--threads", type=int, default=1)
    run_parser.add_argument("--max-call-s", type=float, default=5.0, help="skip larger batches past this call time")
    run_parser.add_argument("--max-compile-s", type=float, default=300.0, help="skip larger batches past this compile")
    run_parser.add_argument("--out", type=Path, default=None)
    table_parser.add_argument("run", type=Path)
    table_parser.add_argument("--before", type=Path, default=None)
    table_parser.add_argument("--reference", type=Path, nargs="*", default=[])
    args = parser.parse_args()
    args.functions = args.functions.split(",")
    args.robots = args.robots.split(",")
    args.batches = [int(batch) for batch in args.batches.split(",")]
    unknown = (set(args.functions) - set(FUNCTIONS)) | (set(args.robots) - URDFS.keys())
    if unknown:
        parser.error(f"unknown functions or robots {sorted(unknown)}: use {','.join(FUNCTIONS)} and panda,g1")
    if args.command == "run":
        args.dtype = getattr(torch, args.dtype)
        default_modes = "eager,compile,reduce-overhead" if args.device == "cuda" else "eager,compile"
        args.modes = (args.modes or default_modes).split(",")
    else:
        args.modes = (args.modes or ",".join(MODES)).split(",")
    if set(args.modes) - MODES.keys():
        parser.error(f"unknown modes {sorted(set(args.modes) - MODES.keys())}: use {','.join(MODES)}")
    return args


def main() -> None:
    """Dispatch ``run`` or ``table``."""
    args = parse_args()
    if args.command == "run":
        run(args)
    else:
        table(args)


if __name__ == "__main__":
    main()
