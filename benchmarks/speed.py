"""The speed suite: wall time of each function per robot, device, compile mode and batch, with reference columns.

Usage (from the repository root; on CPU pin one core whose SMT sibling is idle; never time CPU and CUDA at once)::

    taskset -c 5 uv run python benchmarks/speed.py run --device cpu --grad --out cpu.json
    taskset -c 5 uv run python benchmarks/speed.py run --device cpu --dtype float64 --robots panda7,g1 --out cpu64.json
    CUDA_VISIBLE_DEVICES=4 uv run python benchmarks/speed.py run --device cuda --grad --out cuda.json
    uv run python benchmarks/speed.py table cpu.json cpu64.json --before old_cpu.json --reference frax_cpu.json
    PYTHONPATH=/another/checkout/src uv run python benchmarks/speed.py run ...   # time another version of the library

Robots, from ``tests/fixtures/robots``: ``panda`` (fixed base, nq = nv = 8: seven arm joints and one finger coordinate
the other finger mimics), ``panda7`` (the same with both fingers fixed: the 7-dof arm frax ships) and ``g1``
(free-flyer root, nq 36, nv 35, for every function). Modes: ``eager``, ``compile`` and ``reduce-overhead`` (CUDA
default); ``adapt`` is the one adapter from a mode to a callable.

Functions. A call does what a user's call does and returns what it computes, so compiled code cannot drop it:
``forward_kinematics`` returns ``forward_kinematics(model, q).joint_transforms``, ``forward_kinematics_frames`` the
joint and the frame transforms (an eager state may compute the frames for both; compiled code only for the second),
``joint_poses``, ``frame_jacobian`` (first ``IK_FRAMES`` frame), ``joint_jacobians`` (the state's
``jacobian_columns``, the column table each joint's Jacobian masks, as Pinocchio's ``computeJointJacobians``),
``inverse_dynamics``, ``mass_matrix``, ``forward_dynamics`` and ``center_of_mass``, each of a fresh state. ``--grad``
adds for each of these a row of forward and backward: ``torch.autograd.grad(sum(output.sum()), q)``. On CPU at batch
1 the run adds Pinocchio float64 rows (group ``test-oracles``) on the same models, one call per sample.

IK, aligned with the frax reference run as far as the two algorithms allow:

- start ``start_configuration``: frax's home pose for the Panda (fingers at ``q_neutral``); G1 standing at
  (0, 0, 0.793) with every joint at zero (frax's ``default_configuration``);
- targets: the ``IK_FRAMES`` poses at seeded ``terms.random_start`` configurations of the start (every limited joint
  uniform within its limits; G1's base stays at the start pose); batch ``B`` takes the first ``B``, so a batch above
  ``POOL`` draws fresh ones;
- success: every target frame within 1 mm and 0.01 rad, over at least ``POOL`` targets (``success_targets``): a row
  with ``B < POOL`` reports an eager solve of the whole pool, since a solve treats each sample on its own (LM freezes
  a done element; the restarts draw per sample);
- ``ik_step``: one ``LevenbergMarquardt.step`` from the start (frax: one QP step from ``init(start)``, which is zeros);
  ``solve_ik``: the default call, at most 100 LM iterations, stopping once every sample is done (frax: at most 200
  steps, stopping at 1e-4 on max |v|); ``solve_ik_multistart``: five starts, ``restarts=4`` (frax: five samples);
- ``iterations``: ``Result.iterations``, summed over the starts, which every sample of the batch runs (frax: the median
  over samples of its own count, summed over the starts).

What still differs: LM with hard bounds against frax's QP with a posture task and limit rows; the frames (``panda_hand``,
G1's ``rubber_hand`` and ``ankle_roll`` links against frax's end-effector, hand and foot offsets); frax's G1 URDF
revision and Panda inertias; frax.ik runs in float64 only, so compare it with a ``--dtype float64`` run.

Timing. A row is warmed up, then timed as the median over samples, each one or more calls at least ``SAMPLE_S`` long
between ``torch.cuda.synchronize`` calls: at least ``MIN_SAMPLES``, at most ``--repeats``, none started after
``ROW_BUDGET_S``. Compiles are cold: the run first compiles a throwaway function (the compiler's one-time start-up)
and calls each function once eagerly at batch 1 (filling the model's memo, which a compiled first call would trace),
then each compiled row runs after ``torch.compiler.reset()`` in an empty compile cache of its own; its first call is
``compile_seconds``. A compiled row's counted calls (the timed ones and one more) run with the garbage collector off,
as in the counter gate, so ``recompiles`` counts only graphs compiled after the warm-up; that last call (a CUDA graph
replay under reduce-overhead) is compared with eager as ``max_rel_diff_vs_eager``. A failure is a row with ``error``;
the larger batches of that function, pass and mode are then ``skipped``, as are those after a call above
``--max-call-s`` or a compile above ``--max-compile-s``. The run re-executes itself with glibc's allocator thresholds
fixed (``FIXED_MALLOC``; see the benchmark guide): otherwise a large-batch time can double from page faults alone.
``meta`` names the library measured (``src_root``, ``src_commit``, ``src_tree``: the git tree of its ``src/``,
``src_dirty``), which ``PYTHONPATH`` may take from another checkout than the harness's, and records the cores and
how busy their SMT siblings were at start and end.

Row schema, shared by every row of a run and of a reference JSON (``{"meta": {...}, "rows": [...]}``):

    library           "better_robot", "pinocchio", "frax", ...
    function          one of FUNCTIONS
    robot             one of ROBOTS; REFERENCE_ROBOTS maps a reference's own name
    device            "cpu" or "cuda"
    dtype             "float32" or "float64"
    batch             samples per call
    mode              one of MODES; a reference names its own ("jit", "jit(vmap)", "c++")
    seconds_per_call  median wall time of one call; absent when the row has "error" or "skipped" (text)
    compile_seconds   the first call of a compiled row (optional)

A run's rows add ``grad``, ``nq``, ``nv``, ``seconds_min``, ``seconds_spread`` (interquartile range over the
median), ``samples``, ``calls_per_sample``, ``page_faults`` per call, ``loadavg``, on CUDA ``gpu_other_pids`` (other
processes on the GPU at the row's start or end: the row was timed on a shared GPU), on compiled rows
``graph_breaks``, ``graphs``, ``recompiles``, ``cudagraph_skips`` and ``max_rel_diff_vs_eager`` (largest
``max |compiled - eager| / max |eager|`` over the outputs), on IK rows ``iterations``, ``success_rate``,
``success_targets``, ``median_position_error_m`` and ``median_rotation_error_rad``.

``table`` prints the rows of one or more runs as two markdown tables, kinematics and dynamics, then IK: microseconds
per sample, spread, samples, page faults per call and compile seconds; IK rows add iterations, microseconds per
iteration (per sample over ``iterations``) and success. ``--before`` adds the same row of older runs and the speedup
``before / after``; each library of ``--reference`` (and the runs' own Pinocchio rows) adds its microseconds per
sample, its dtype and mode (and nv when it differs), on IK rows its iterations, microseconds per iteration and
success, and the ratio ``ours / reference``. A ratio within the two rows' spreads (``|ratio - 1|`` at most their sum)
is marked ``~``, one without a spread ``?``, one across dtypes ``*``; a cell of ``NOT_COMPARABLE`` reads ``n/c``.
A reference row matches on function, robot, device and batch, preferring the same dtype; a Pinocchio row (a serial
loop) matches every device and batch at its batch-1 time per sample. CPU rows above ``CPU_HEADLINE_BATCH`` are
flagged: page faults dominate them.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime
import gc
import json
import math
import os
import platform
import resource
import shlex
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from functools import cache, partial
from pathlib import Path
from typing import Any

import torch
from torch._dynamo.utils import counters as dynamo_counters
from torch._inductor.utils import fresh_cache

import better_robot as br
from better_robot.model import JointFixed
from better_robot.optim import LevenbergMarquardt, State

REPO_ROOT = Path(__file__).resolve().parent.parent
ROBOT_DIR = REPO_ROOT / "tests" / "fixtures" / "robots"
URDFS = {
    "panda": ROBOT_DIR / "panda" / "panda.urdf",
    "panda7": ROBOT_DIR / "panda" / "panda.urdf",
    "g1": ROBOT_DIR / "g1" / "g1_29dof.urdf",
}
ROBOTS = tuple(URDFS)
FIXED_FINGERS = ("panda_finger_joint1", "panda_finger_joint2")  # fixed in panda7
PANDA_HOME = (0.0, -math.pi / 6, 0.0, -3 * math.pi / 4, 0.0, 5 * math.pi / 9, 0.0)  # frax's Panda home pose
G1_BASE_POSITION = (0.0, 0.0, 0.793)  # frax's G1 default: identity orientation, every joint at zero
IK_FRAMES = {
    "panda": ("panda_hand",),
    "panda7": ("panda_hand",),
    "g1": ("left_rubber_hand", "right_rubber_hand", "left_ankle_roll_link", "right_ankle_roll_link"),
}
REFERENCE_ROBOTS = {("frax", "panda"): "panda7"}  # frax's Panda has its fingers fixed
NOT_COMPARABLE = {("frax", "joint_jacobians"): "frax returns every joint's full Jacobian, 2 x (nv, 3, nv)"}
BATCHES = "1,64,1024,4096,16384,65536"
CPU_HEADLINE_BATCH = 4096
POOL = 1024
SAMPLE_S = 0.01
MIN_SAMPLES = 3
ROW_BUDGET_S = 1.0
IK_POSITION_M, IK_ROTATION_RAD = 1e-3, 1e-2
FIXED_MALLOC = {"MALLOC_MMAP_THRESHOLD_": "134217728", "MALLOC_TRIM_THRESHOLD_": "268435456"}
MODES = ("eager", "compile", "reduce-overhead")


def adapt(fn: Callable[..., Any], mode: str) -> Callable[..., Any]:
    """The callable of ``fn`` in ``mode``: itself, ``torch.compile(fn)`` or ``torch.compile(fn, mode=mode)``."""
    return fn if mode == "eager" else torch.compile(fn, mode=None if mode == "compile" else mode)


def fk_frames(model: br.Model, frame: int, q: torch.Tensor, v: torch.Tensor, a: torch.Tensor) -> tuple:
    """The joint and the frame transforms of ``q``."""
    state = br.forward_kinematics(model, q)
    return state.joint_transforms, state.frame_transforms


KINEMATICS: dict[str, Callable[..., torch.Tensor | tuple]] = {
    "forward_kinematics": lambda model, frame, q, v, a: br.forward_kinematics(model, q).joint_transforms,
    "forward_kinematics_frames": fk_frames,
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
    """One of ``ROBOTS`` from the tracked URDFs (module docstring)."""
    ir = br.io.parse_urdf(URDFS[robot])
    if robot == "panda7":
        fixed = {"joint": JointFixed(), "lower": None, "upper": None, "velocity_limit": None, "effort_limit": None}
        fixed |= {"mimic_source": None}
        ir.joints = [dataclasses.replace(j, **fixed) if j.name in FIXED_FINGERS else j for j in ir.joints]
    root_joint = br.JointFreeFlyer() if robot == "g1" else None
    return br.io.build_model(ir, root_joint=root_joint, dtype=dtype).to(device=device)


def start_configuration(robot: str, model: br.Model) -> torch.Tensor:
    """The IK start ``(nq,)`` (module docstring)."""
    start = model.q_neutral.clone()
    if robot == "g1":
        start[:3] = torch.tensor(G1_BASE_POSITION, dtype=model.dtype)
        start[7:] = 0.0
    else:
        start[:7] = torch.tensor(PANDA_HOME, dtype=model.dtype)
    return start


def ik_case(
    function: str, robot: str, model: br.Model, q_target: torch.Tensor, start: torch.Tensor
) -> tuple[Callable[..., Any], tuple[torch.Tensor, ...], Callable[[Any], dict[str, Any]]]:
    """``(fn, args, report)`` of an IK row toward the ``IK_FRAMES`` poses at ``q_target: (B, nq)``, from ``start: (nq,)``.

    ``fn(*args)`` returns ``(q, ...)``; ``report(output)`` gives the call's iterations and, for a solve, its accuracy.
    """
    frames, frame_ids = IK_FRAMES[robot], [model.frame_id(name) for name in IK_FRAMES[robot]]
    with torch.no_grad():
        reached = br.forward_kinematics(model, q_target)
        targets = reached.frame_transforms[:, frame_ids]  # (B, F, 3, 4)
        poses = tuple(reached.frame_poses[:, frame_ids].unbind(1))
    q0 = start.expand(q_target.shape[0], model.nq).clone()
    if function == "ik_step":
        problem = br.terms.ik_problem(model, dict(zip(frames, poses, strict=True)), q0=q0)
        solver = LevenbergMarquardt()
        begin = solver.init(problem, {"q": q0})
        names = tuple(begin.extra["terms"])

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

        extra = (begin.extra["damping"], begin.extra["growth"], *begin.extra["terms"].values())
        return step, (q0, begin.loss, begin.done, *extra), lambda output: {"iterations": 1}
    restarts = 4 if function == "solve_ik_multistart" else 0

    def solve(q0: torch.Tensor, *poses: torch.Tensor) -> tuple[torch.Tensor, int]:
        generator = torch.Generator(device=q0.device).manual_seed(1) if restarts else None
        targets = dict(zip(frames, poses, strict=True))
        result = br.solve_ik(model, targets, q0=q0, restarts=restarts, generator=generator)
        return result.x["q"], result.iterations

    def report(output: tuple[torch.Tensor, int]) -> dict[str, Any]:
        return {"iterations": output[1]} | ik_accuracy(robot, model, output[0], targets)

    return solve, (q0, *poses), report


def ik_accuracy(robot: str, model: br.Model, q: torch.Tensor, targets: torch.Tensor) -> dict[str, Any]:
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
        "success_targets": success.numel(),
        "median_position_error_m": position.amax(dim=-1).median().item(),
        "median_rotation_error_rad": angle.amax(dim=-1).median().item(),
    }


def pool_accuracy(function: str, robot: str, model: br.Model, q_target: torch.Tensor, start: torch.Tensor) -> dict:
    """The accuracy fields of one eager solve toward every target of ``q_target: (POOL, nq)``."""
    fn, args, report = ik_case(function, robot, model, q_target, start)
    with torch.no_grad():
        fields = report(fn(*args))
    del fields["iterations"]
    return fields


def time_call(call: Callable[[], object], sync: Callable[[], None], repeats: int) -> dict[str, Any]:
    """Median seconds per call over samples of ``calls_per_sample`` calls each (module docstring); call warmed up."""
    faults = resource.getrusage(resource.RUSAGE_SELF).ru_minflt
    sync()
    began = time.perf_counter()
    call()
    sync()
    probe = time.perf_counter() - began
    inner = max(1, int(SAMPLE_S / max(probe, 1e-7)))
    samples: list[float] = []
    while len(samples) < MIN_SAMPLES or (len(samples) < repeats and time.perf_counter() - began < ROW_BUDGET_S):
        sync()
        start = time.perf_counter()
        for _ in range(inner):
            call()
        sync()
        samples.append((time.perf_counter() - start) / inner)
    median = statistics.median(samples)
    quartiles = statistics.quantiles(samples, n=4)
    return {
        "seconds_per_call": median,
        "seconds_min": min(samples),
        "seconds_spread": (quartiles[2] - quartiles[0]) / median,
        "samples": len(samples),
        "calls_per_sample": inner,
        "page_faults": (resource.getrusage(resource.RUSAGE_SELF).ru_minflt - faults) / (1 + inner * len(samples)),
    }


def tensors(output: object) -> list[torch.Tensor]:
    """The tensors of a call's output, a tensor or a tuple."""
    items = output if isinstance(output, tuple) else (output,)
    return [item for item in items if isinstance(item, torch.Tensor)]


def max_rel_diff(outputs: list[torch.Tensor], references: list[torch.Tensor]) -> float:
    """Largest ``max |output - reference| / max |reference|`` over pairs of tensors."""
    tiny = torch.finfo(torch.float64).tiny
    pairs = zip(outputs, references, strict=True)
    return max(float((a.double() - b.double()).abs().max() / b.double().abs().max().clamp_min(tiny)) for a, b in pairs)


def forward_backward(fn: Callable[..., Any], args: tuple[torch.Tensor, ...]) -> Callable[[], tuple]:
    """A call of ``fn(q, *rest)`` and the gradient of the sum of its outputs with respect to ``q = args[0]``."""
    q = args[0].detach().requires_grad_(True)

    def call() -> tuple:
        return torch.autograd.grad(sum(output.sum() for output in tensors(fn(q, *args[1:]))), q)

    return call


def measure(
    case: Callable[[], tuple],
    mode: str,
    device: torch.device,
    repeats: int,
    eager_output: list[torch.Tensor] | None,
    *,
    grad: bool,
) -> tuple[dict[str, Any], list[torch.Tensor]]:
    """Time ``adapt(fn, mode)(*args)`` of ``fn, args, report = case()``, or its forward and backward (module
    docstring): the row's fields and the tensors of its output."""
    sync = (lambda: torch.cuda.synchronize(device)) if device.type == "cuda" else (lambda: None)
    fn, args, report = case()
    row: dict[str, Any] = {}
    compiled = mode != "eager"
    with fresh_cache() if compiled else contextlib.nullcontext(), contextlib.nullcontext() if grad else torch.no_grad():
        if compiled:
            torch.compiler.reset()
            dynamo_counters.clear()
        call = forward_backward(adapt(fn, mode), args) if grad else partial(adapt(fn, mode), *args)
        sync()
        start = time.perf_counter()
        output = call()
        sync()
        if compiled:
            row["compile_seconds"] = time.perf_counter() - start
            call()
            call()
            gc.collect()
            gc.disable()  # a collection inside a compiled call runs weakref callbacks that dynamo counts as frames
        try:
            frames = dynamo_counters["frames"]["ok"]
            others = gpu_other_pids(device) if device.type == "cuda" else []
            row |= time_call(call, sync, repeats)
            output = call()  # the replayed call: a CUDA graph overwrites the outputs of earlier ones
            if compiled and eager_output is not None:
                row["max_rel_diff_vs_eager"] = max_rel_diff(tensors(output), eager_output)
            recompiles = dynamo_counters["frames"]["ok"] - frames
        finally:
            gc.enable()
        if not grad:
            row |= report(output) if report is not None else {}
    if compiled:
        row["graph_breaks"] = sum(dynamo_counters["graph_break"].values())
        row["graphs"] = dynamo_counters["stats"]["unique_graphs"]
        row["recompiles"] = recompiles
        row["cudagraph_skips"] = dynamo_counters["inductor"]["cudagraph_skips"]
    if device.type == "cuda":
        row["gpu_other_pids"] = sorted(set(others) | set(gpu_other_pids(device)))
    return row, tensors(output)


def row_fields(
    measured: Callable[[], tuple[dict[str, Any], list[torch.Tensor]]], pool: Callable[[], dict[str, Any]] | None
) -> tuple[dict[str, Any], list[torch.Tensor] | None]:
    """The fields and output tensors of ``measured()``, or its ``error``; a solve row takes its accuracy from
    ``pool()`` when given."""
    try:
        fields, output = measured()
        if "success_rate" in fields and pool is not None:
            fields |= pool()
    except Exception as exc:  # noqa: BLE001 - a failure (out of memory, a compile error) is a row
        return {"error": f"{type(exc).__name__}: {str(exc)[:400]}"}, None
    return fields, output


def stop_reason(row: dict[str, Any], args: argparse.Namespace) -> str | None:
    """Why the larger batches of a row's function, pass and mode are skipped, if they are."""
    if "error" in row:
        return f"batch {row['batch']} failed"
    if row["seconds_per_call"] > args.max_call_s:
        return f"batch {row['batch']} took {row['seconds_per_call']:.1f} s per call > --max-call-s"
    if row.get("compile_seconds", 0.0) > args.max_compile_s:
        return f"batch {row['batch']} compiled in {row['compile_seconds']:.0f} s > --max-compile-s"
    return None


def nvidia_smi(device: torch.device, query: str) -> list[list[str]]:
    """The fields after the uuid of each line of ``nvidia-smi --query-<query>`` that names ``device``."""
    uuid = f"GPU-{torch.cuda.get_device_properties(device).uuid}"
    command = ["nvidia-smi", f"--query-{query}", "--format=csv,noheader,nounits"]
    listing = subprocess.run(command, capture_output=True, text=True, check=True).stdout
    lines = [line.split(", ") for line in listing.splitlines() if line.strip()]
    return [fields[1:] for fields in lines if fields[0] == uuid]


def gpu_other_pids(device: torch.device) -> list[int]:
    """Processes other than this one with a CUDA context on ``device``."""
    return sorted({int(pid) for (pid,) in nvidia_smi(device, "compute-apps=gpu_uuid,pid") if int(pid) != os.getpid()})


def gpu_state(device: torch.device) -> dict[str, Any]:
    """Utilization, memory used and other processes of ``device``."""
    ((utilization, memory),) = nvidia_smi(device, "gpu=uuid,utilization.gpu,memory.used")
    return {
        "utilization_percent": int(utilization),
        "memory_used_mib": int(memory),
        "other_pids": gpu_other_pids(device),
    }


def siblings_busy(cores: list[int], seconds: float = 1.0) -> dict[str, float]:
    """Busy fraction over ``seconds`` of each SMT sibling of ``cores`` outside them, from ``/proc/stat``."""
    siblings: set[int] = set()
    for core in cores:
        text = Path(f"/sys/devices/system/cpu/cpu{core}/topology/thread_siblings_list").read_text().strip()
        for part in text.split(","):
            first, _, last = part.partition("-")
            siblings |= set(range(int(first), int(last or first) + 1))

    def times() -> dict[int, tuple[int, int]]:
        out = {}
        for line in Path("/proc/stat").read_text().splitlines():
            name, *fields = line.split()
            if name[3:].isdigit() and int(name[3:]) in siblings:
                values = [int(field) for field in fields]
                out[int(name[3:])] = (sum(values), values[3] + values[4])  # all, idle + iowait
        return out

    siblings -= set(cores)
    before = times()
    time.sleep(seconds)
    after = times()
    return {str(c): round(1 - (after[c][1] - before[c][1]) / max(after[c][0] - before[c][0], 1), 3) for c in after}


def kinematics_case(
    function: str, robot: str, model: br.Model, q: torch.Tensor, v: torch.Tensor, a: torch.Tensor
) -> tuple[Callable[..., Any], tuple[torch.Tensor, ...], None]:
    """``(fn, args, None)`` of a ``KINEMATICS`` row at ``q, v, a: (B, nq), (B, nv), (B, nv)``."""
    return partial(KINEMATICS[function], model, model.frame_id(IK_FRAMES[robot][0])), (q, v, a), None


def make_case(
    function: str,
    robot: str,
    model: br.Model,
    batch: int,
    inputs: tuple[torch.Tensor, ...],
    ik_q: torch.Tensor,
    start: torch.Tensor,
) -> Callable[[], tuple]:
    """The case of a row at ``batch``: kinematics at the pool ``inputs = (q, v, a)`` tiled, IK toward the first
    ``batch`` targets ``ik_q: (N, nq)``."""
    if function in KINEMATICS:
        q, v, a = (values[torch.arange(batch) % POOL].to(model.device) for values in inputs)
        return partial(kinematics_case, function, robot, model, q, v, a)
    return partial(ik_case, function, robot, model, ik_q[:batch].to(model.device), start)


def warm_compiler(device: torch.device, modes: list[str]) -> None:
    """Compile a throwaway function in each compiled mode, forward and backward: the compiler's one-time start-up
    is then not part of the first row's ``compile_seconds``."""
    x = torch.ones(8, device=device, requires_grad=True)
    for mode in (mode for mode in modes if mode != "eager"):
        with fresh_cache():
            torch.compiler.reset()
            compiled = adapt(lambda t: t.sin() * 2, mode)
            with torch.no_grad():
                compiled(x)
            torch.autograd.grad(compiled(x).sum(), x)
    torch.compiler.reset()


def better_robot_rows(
    args: argparse.Namespace, device: torch.device, rows: list[dict[str, Any]], save: Callable[[], None]
) -> None:
    """Append the rows of ``--robots`` x ``--functions`` x passes x ``--batches`` x ``--modes`` (module docstring),
    saving after each."""
    warm_compiler(device, args.modes)
    for robot in args.robots:
        model = load_model(robot, args.dtype, torch.device("cpu"))
        generator = torch.Generator().manual_seed(0)
        pool_q = torch.stack([model.random_configuration(generator=generator) for _ in range(POOL)])
        pool_v, pool_a = (torch.randn(POOL, model.nv, generator=generator, dtype=args.dtype) for _ in range(2))
        start = start_configuration(robot, model)
        draws = start.expand(max(POOL, *args.batches), model.nq).contiguous()
        ik_q = br.terms.random_start(model, draws, generator=torch.Generator().manual_seed(2))  # restarts draw seed 1
        model, start = model.to(device=device), start.to(device)
        stopped: dict[tuple[str, bool, str], str] = {}  # (function, grad, mode) -> why its larger batches are skipped
        for function in args.functions:
            fn, warm_args, _ = make_case(function, robot, model, 1, (pool_q, pool_v, pool_a), ik_q, start)()
            with torch.no_grad():
                fn(*warm_args)  # fills the model's memo: no compiled row traces its construction
            pool_solve = cache(partial(pool_accuracy, function, robot, model, ik_q[:POOL].to(device), start))
            for grad in (False, True) if args.grad and function in KINEMATICS else (False,):
                for batch in args.batches:
                    case = make_case(function, robot, model, batch, (pool_q, pool_v, pool_a), ik_q, start)
                    eager_output = output = None
                    for mode in args.modes:
                        row: dict[str, Any] = {"library": "better_robot", "function": function, "robot": robot}
                        row |= {"device": device.type, "dtype": str(args.dtype).removeprefix("torch."), "batch": batch}
                        row |= {"mode": mode, "grad": grad, "nq": model.nq, "nv": model.nv, "loadavg": loadavg()}
                        if (function, grad, mode) in stopped:
                            row["skipped"] = stopped[function, grad, mode]
                        else:
                            measured = partial(measure, case, mode, device, args.repeats, eager_output, grad=grad)
                            fields, output = row_fields(measured, pool_solve if batch < POOL else None)
                            row |= fields
                            eager_output = output if mode == "eager" else eager_output
                            reason = stop_reason(row, args)
                            if reason is not None:
                                stopped[function, grad, mode] = reason
                        rows.append(row)
                        save()
                        print_row(row)
                    eager_output = output = None  # free the batch's tensors before the next batch
                    if device.type == "cuda":
                        torch.cuda.empty_cache()


def pinocchio_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Pinocchio float64 rows at batch 1 on the same models: the reference cost per sample of a serial loop."""
    import numpy as np  # noqa: PLC0415 - with the optional Pinocchio oracle below
    import pinocchio as pin  # noqa: PLC0415 - the oracle of the test-oracles group, imported only for its rows

    rows = []
    for robot in args.robots:
        root = (pin.JointModelFreeFlyer(),) if robot == "g1" else ()
        model = pin.buildModelFromUrdf(str(URDFS[robot]), *root)
        if robot == "panda7":
            fixed = [model.getJointId(name) for name in FIXED_FINGERS]
            model = pin.buildReducedModel(model, fixed, pin.neutral(model))
        data = model.createData()
        rng = np.random.default_rng(1)
        q = pin.neutral(model)
        v, a = (rng.standard_normal(model.nv) for _ in range(2))
        frame = model.getFrameId(IK_FRAMES[robot][0])
        calls = {
            "forward_kinematics": lambda: pin.forwardKinematics(model, data, q),
            "forward_kinematics_frames": lambda: pin.framesForwardKinematics(model, data, q),
            "frame_jacobian": lambda: pin.computeFrameJacobian(model, data, q, frame, pin.LOCAL_WORLD_ALIGNED),
            "joint_jacobians": lambda: pin.computeJointJacobians(model, data, q),
            "inverse_dynamics": lambda: pin.rnea(model, data, q, v, a),
            "mass_matrix": lambda: pin.crba(model, data, q),
            "forward_dynamics": lambda: pin.aba(model, data, q, v, a),
            "center_of_mass": lambda: pin.centerOfMass(model, data, q),
        }
        for function in (name for name in args.functions if name in calls):
            row = {"library": "pinocchio", "function": function, "robot": robot, "device": "cpu", "dtype": "float64"}
            row |= {"batch": 1, "mode": "c++", "nq": model.nq, "nv": model.nv, "loadavg": loadavg()}
            calls[function]()
            rows.append(row | time_call(calls[function], lambda: None, args.repeats))
            print_row(rows[-1])
    return rows


def git(root: Path, *arguments: str) -> str:
    """Stdout of one git command in the checkout at ``root``."""
    command = ["git", "-C", str(root), *arguments]
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
    """One progress line: the row's key, microseconds per sample, spread, compile seconds, IK success or the failure."""
    key = f"{row['library']}/{row['function']}{'+grad' if row.get('grad') else ''}/{row['robot']}/{row['device']}"
    key += f"/{row['dtype']}/B{row['batch']}/{row['mode']}"
    if "seconds_per_call" not in row:
        print(f"{key:72} {row.get('error') or row.get('skipped')}", flush=True)
        return
    line = f"{key:72} {1e6 * row['seconds_per_call'] / row['batch']:12.3f} us/sample {row['seconds_spread']:6.1%}"
    if "compile_seconds" in row:
        line += f"  compile {row['compile_seconds']:6.1f} s"
    if "success_rate" in row:
        line += f"  success {row['success_rate']:.3f} iterations {row['iterations']}"
    print(line, flush=True)


def run(args: argparse.Namespace) -> None:
    """Measure the rows and write the JSON (module docstring)."""
    if any(os.environ.get(name) != value for name, value in FIXED_MALLOC.items()):
        os.execve(sys.executable, [sys.executable, *sys.argv], {**os.environ, **FIXED_MALLOC})
    if os.environ.get("BETTER_ROBOT_INTERNAL_CHECKS") == "1":
        sys.exit("BETTER_ROBOT_INTERNAL_CHECKS=1 adds host syncs and operators: unset it to benchmark")
    torch.set_num_threads(args.threads)  # noqa: TID251 - a benchmark pins its threads so rows compare; C-DEP-3 binds the library
    device = torch.device(args.device)
    src_root = Path(br.__file__).resolve().parents[2]
    cores = sorted(os.sched_getaffinity(0))
    meta: dict[str, Any] = {
        "commit": git(REPO_ROOT, "rev-parse", "HEAD"),
        "dirty": bool(git(REPO_ROOT, "status", "--porcelain", "--", "benchmarks")),
        "src_root": str(src_root),
        "src_commit": git(src_root, "rev-parse", "HEAD"),
        "src_tree": git(src_root, "rev-parse", "HEAD:src"),  # equal trees: the same library
        "src_dirty": bool(git(src_root, "status", "--porcelain", "--", "src")),
        "command": shlex.join(sys.argv),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "host": platform.node(),
        "device": args.device,
        "device_name": device_name(device),
        "dtype": str(args.dtype).removeprefix("torch."),
        "grad": args.grad,
        "threads": torch.get_num_threads(),
        "affinity": cores,
        "siblings_busy_start": siblings_busy(cores),
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
    meta["siblings_busy_end"] = siblings_busy(cores)
    if device.type == "cuda":
        meta["gpu_end"] = gpu_state(device)
    meta["total_s"] = round(time.perf_counter() - started, 1)
    save()
    print(f"wrote {out} ({meta['total_s']} s)")


def per_sample_us(row: dict[str, Any]) -> float | None:
    """Microseconds per sample of a timed row."""
    return 1e6 * row["seconds_per_call"] / row["batch"] if "seconds_per_call" in row else None


def spread(row: dict[str, Any]) -> float | None:
    """Interquartile range over the median of a timed row (a frax row gives its quartiles)."""
    if row.get("seconds_spread") is not None:
        return row["seconds_spread"]
    if "seconds_per_call_q75" in row:
        return (row["seconds_per_call_q75"] - row["seconds_per_call_q25"]) / row["seconds_per_call"]
    return None


def ratio(numerator: dict[str, Any], denominator: dict[str, Any]) -> str:
    """``numerator / denominator`` in time per sample, marked ``~`` inside the noise, ``?`` without a spread and
    ``*`` across dtypes (module docstring)."""
    value = per_sample_us(numerator) / per_sample_us(denominator)
    spreads = [spread(numerator), spread(denominator)]
    mark = "?" if None in spreads else "~" if abs(value - 1) <= sum(spreads) else ""
    return f"{value:.3g}x{mark}{'*' if numerator['dtype'] != denominator['dtype'] else ''}"


def row_key(row: dict[str, Any]) -> tuple:
    """What makes a row of a run the same row of another run."""
    return row["function"], row["robot"], row["device"], row["dtype"], row["batch"], row["mode"], bool(row.get("grad"))


def match_reference(candidates: list[dict[str, Any]], library: str, row: dict[str, Any]) -> dict[str, Any] | None:
    """The row of ``library`` among ``candidates`` that ``row`` compares with (module docstring), if any."""
    serial = library in SERIAL_REFERENCES
    matches = [
        candidate
        for candidate in candidates
        if (candidate["function"], REFERENCE_ROBOTS.get((library, candidate["robot"]), candidate["robot"]))
        == (row["function"], row["robot"])
        and (serial or (candidate["device"], candidate["batch"]) == (row["device"], row["batch"]))
        and not row.get("grad")
    ]
    return min(matches, key=lambda candidate: candidate["dtype"] != row["dtype"], default=None)


def reference_cells(library: str, match: dict[str, Any] | None, row: dict[str, Any], *, ik: bool) -> list[str]:
    """The cells of one reference library for ``row``: time per sample, run, IK cells, ratio; then the note."""
    run_cell = f"{match['dtype']} {match['mode']}" if match else "-"
    if match and match.get("nv") not in (None, row.get("nv")):
        run_cell += f" nv {match['nv']}"
    cells = [number(per_sample_us(match) if match else None), run_cell] + (ik_cells(match) if ik else [])
    not_comparable = NOT_COMPARABLE.get((library, row["function"])) if match else None
    if not_comparable:
        return [*cells, "n/c", f"{library}: {not_comparable}"]
    return [*cells, ratio(row, match) if match and "seconds_per_call" in row else "-", ""]


def table(args: argparse.Namespace) -> None:
    """Print the markdown tables of the runs against ``--before`` and the references (module docstring)."""
    runs = [row for path in args.run for row in json.loads(path.read_text())["rows"]]
    ours = [
        row
        for row in runs
        if row["library"] == "better_robot"
        and row["function"] in args.functions
        and row["robot"] in args.robots
        and row["batch"] in args.batches
        and row["mode"] in args.modes
    ]
    ours.sort(
        key=lambda row: (
            FUNCTIONS.index(row["function"]),
            ROBOTS.index(row["robot"]),
            row["dtype"],
            bool(row.get("grad")),
            MODES.index(row["mode"]),
            row["batch"],
        )
    )
    old_rows = [row for path in args.before for row in json.loads(path.read_text())["rows"]]
    before = {row_key(row): row for row in old_rows if row["library"] == "better_robot" and "seconds_per_call" in row}
    references: dict[str, list[dict[str, Any]]] = {}
    for row in runs + [row for path in args.reference for row in json.loads(path.read_text())["rows"]]:
        if row["library"] != "better_robot" and "seconds_per_call" in row:
            references.setdefault(row["library"], []).append(row)
    for ik in (False, True):
        section = [row for row in ours if (row["function"] in KINEMATICS) != ik]
        libraries = [
            name for name, rows in references.items() if any((r["function"] in KINEMATICS) != ik for r in rows)
        ]
        if not section:
            continue
        header = ["function", "robot", "device", "dtype", "batch", "mode"] + ([] if ik else ["pass"])
        header += ["us/sample", "spread", "n", "compile s"] + (["iterations", "us/iteration", "success"] if ik else [])
        header += [] if ik else ["faults/call"]
        header += ["before us/sample", "before/after"] + (["before success"] if ik else []) if args.before else []
        for library in libraries:
            header += [f"{library} us/sample", f"{library} run"]
            header += [f"{library} iterations", f"{library} us/iteration", f"{library} success"] if ik else []
            header += [f"ours/{library}"]
        print("| " + " | ".join([*header, "note"]) + " |")
        print("|" + "---|" * (len(header) + 1))
        for row in section:
            cells = [row["function"], row["robot"], row["device"], row["dtype"], str(row["batch"]), row["mode"]]
            cells += [] if ik else ["fwd+bwd" if row.get("grad") else "fwd"]
            cells += [number(per_sample_us(row)), percent(spread(row)), str(row.get("samples", "-"))]
            cells += [number(row.get("compile_seconds"))] + (ik_cells(row) if ik else [number(row.get("page_faults"))])
            notes = [str(row.get("error") or row.get("skipped") or "")[:60]]
            notes.append("GPU shared" if row.get("gpu_other_pids") else "")
            large = row["device"] == "cpu" and row["batch"] > CPU_HEADLINE_BATCH
            notes.append("above the CPU headline batch: page faults" if large else "")
            if args.before:
                old = before.get(row_key(row))
                cells += [number(per_sample_us(old) if old else None)]
                cells += [ratio(old, row) if old and "seconds_per_call" in row else "-"]
                cells += [number(old.get("success_rate") if old else None)] if ik else []
            for library in libraries:
                match = match_reference(references[library], library, row)
                *reference, note = reference_cells(library, match, row, ik=ik)
                cells += reference
                notes.append(note)
            print("| " + " | ".join([*cells, "; ".join(note for note in notes if note)]) + " |")
        print()


def ik_cells(row: dict[str, Any] | None) -> list[str]:
    """``[iterations, us per iteration, success]`` of an IK row, or dashes."""
    if row is None:
        return ["-", "-", "-"]
    iterations = row.get("iterations")
    us = per_sample_us(row)
    per_iteration = us / iterations if us is not None and iterations else None
    return [number(iterations), number(per_iteration), number(row.get("success_rate"))]


def number(value: float | None, suffix: str = "") -> str:
    """Three significant digits, or ``-``."""
    return "-" if value is None else f"{value:.3g}{suffix}"


def percent(value: float | None) -> str:
    """A fraction as a whole percentage, or ``-``."""
    return "-" if value is None else f"{100 * value:.0f}%"


def parse_args() -> argparse.Namespace:
    """The command line, with the comma lists split."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "table"):
        command = commands.add_parser(name)
        command.add_argument("--functions", default=",".join(FUNCTIONS))
        command.add_argument("--robots", default=",".join(ROBOTS))
        command.add_argument("--batches", default=BATCHES)
        command.add_argument("--modes", default=None, help="default: eager,compile on cpu, and reduce-overhead on cuda")
    run_parser, table_parser = commands.choices["run"], commands.choices["table"]
    run_parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    run_parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    run_parser.add_argument("--grad", action="store_true", help="add forward-and-backward rows of the kinematics")
    run_parser.add_argument("--repeats", type=int, default=15, help="most samples per row")
    run_parser.add_argument("--threads", type=int, default=1)
    run_parser.add_argument("--max-call-s", type=float, default=5.0, help="skip larger batches past this call time")
    run_parser.add_argument("--max-compile-s", type=float, default=300.0, help="skip larger batches past this compile")
    run_parser.add_argument("--out", type=Path, default=None)
    table_parser.add_argument("run", type=Path, nargs="+")
    table_parser.add_argument("--before", type=Path, nargs="*", default=[])
    table_parser.add_argument("--reference", type=Path, nargs="*", default=[])
    args = parser.parse_args()
    args.functions = args.functions.split(",")
    args.robots = args.robots.split(",")
    args.batches = [int(batch) for batch in args.batches.split(",")]
    unknown = (set(args.functions) - set(FUNCTIONS)) | (set(args.robots) - set(ROBOTS))
    if unknown:
        parser.error(f"unknown functions or robots {sorted(unknown)}: use {','.join(FUNCTIONS)} and {','.join(ROBOTS)}")
    if args.command == "run":
        args.dtype = getattr(torch, args.dtype)
        default_modes = "eager,compile,reduce-overhead" if args.device == "cuda" else "eager,compile"
        args.modes = (args.modes or default_modes).split(",")
    else:
        args.modes = (args.modes or ",".join(MODES)).split(",")
    if set(args.modes) - set(MODES):
        parser.error(f"unknown modes {sorted(set(args.modes) - set(MODES))}: use {','.join(MODES)}")
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
