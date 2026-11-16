"""The counter gate: counters of BetterRobot's hot entry points, with their wall time and the context to read them.

Usage (from the repository root)::

    uv run python benchmarks/run.py --counters-only     # the counter rows, checked against the thresholds
    uv run python benchmarks/run.py                     # the counter rows with wall time
    uv run python benchmarks/run.py --device cuda --counters-only

Wall time across functions, robots, compile modes and batch sizes, with Pinocchio and other references, is
``benchmarks/speed.py``.

Counter rows (principle 5.8; record 0011) are fixed functions on the tracked Panda and G1 (``tests/fixtures/robots``):
``fk_joints/{panda,g1}/B{1,1024}`` (``forward_kinematics(model, q).joint_transforms``), ``ik/panda/B64``
(``solve_ik`` to 64 ``panda_hand`` poses, 5 seeds, fixed generators; it also records ``success_fraction`` and
``worst_position_error_m``), ``ik_constrained/panda/B64`` (20 steps of ``SequentialQP`` to the same poses with an
obstacle sphere and the arm's own sphere pairs as clearance rows; it also records ``success_fraction`` and
``worst_violation_m``, over every pair), ``track/panda/B64`` (20 calls of ``optim.track`` with the same rows and
steps of at most 5 mrad, each warm started from the last), ``{jacobian,rnea,crba,aba}/{panda,g1}/B1`` and ``qp/n7m30/B1``, ``qp/n35m150/B1024``
(one cold ``optim.solve_qp`` of seeded random programs with that many variables and rows). Each has four counters, deterministic for
one torch version and device: ``graph_breaks`` (``torch._dynamo.explain``), ``recompiles`` (frames compiled when
``torch.compile(fn, backend="eager")``, first called at the row's batch, is called at each batch of ``SWEEP``),
``ops`` (top-level ``aten::`` operators of one call) and ``host_syncs`` (calls of the operators that make the host
wait for a CUDA queue, ``SYNC_OPS``). The run exits 1 when a counter row exceeds its threshold in
``benchmarks/thresholds/torch-<version>.json``, has none, or no such file exists for the installed torch; a thresholds
file is this runner's ``--counters-only`` output from a clean commit, changed only in a commit of its own.

Wall time is the median of up to ``--repeats`` calls (at least 3; fewer once a row passes ``ROW_BUDGET_S``) after one
warm-up call, with the minor page faults per call; the counters are taken after it. It is recorded, never gated: the
load average sits beside every row.

Every number is float32, ``--threads`` Torch threads (default 1), ``BETTER_ROBOT_INTERNAL_CHECKS`` off. Output:
``{"meta": {...}, "rows": [{"row": id, ...}, ...]}`` in ``--out`` (default
``benchmarks/results/<commit>_<device>.json``, never tracked); ``meta`` records commit, dirty flag, command, torch,
Python, device, CPU, threads, affinity, the allocator variables and the load average at start and end.
"""

from __future__ import annotations

import argparse
import datetime
import gc
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
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import torch
import torch._dynamo
from torch._dynamo.utils import counters as dynamo_counters
from torch.profiler import ProfilerActivity, profile

import better_robot as br

REPO_ROOT = Path(__file__).resolve().parent.parent
ROBOTS = REPO_ROOT / "tests" / "fixtures" / "robots"  # the tracked test robots (record 0039): offline, one file each
URDFS = {"panda": ROBOTS / "panda" / "panda.urdf", "g1": ROBOTS / "g1" / "g1_29dof.urdf"}
THRESHOLDS = REPO_ROOT / "benchmarks" / "thresholds" / f"torch-{torch.__version__.split('+')[0]}.json"
DTYPE = torch.float32
ROW_BUDGET_S = 2.0
SWEEP = (1, 2, 3, 64)
COUNTERS = ("graph_breaks", "recompiles", "ops", "host_syncs")
SYNC_OPS = frozenset({"aten::_local_scalar_dense", "aten::nonzero"})  # item/bool/float/int of a tensor; mask indexing
LARGE_BATCH = 1024
IK_FRAME = "panda_hand"
IK_SUCCESS_M = 1e-3


def jacobian(model: br.Model, q: torch.Tensor, v: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """``(B, 6, nv)`` Jacobian of the last frame."""
    return br.frame_jacobian(model, br.forward_kinematics(model, q), frame=model.nframes - 1)


def rnea(model: br.Model, q: torch.Tensor, v: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """``(B, nv)`` joint torques for velocity ``v`` and acceleration ``u``."""
    return br.inverse_dynamics(model, br.forward_kinematics(model, q), v=v, a=u)


def crba(model: br.Model, q: torch.Tensor, v: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """``(B, nv, nv)`` joint-space mass matrix."""
    return br.mass_matrix(model, br.forward_kinematics(model, q))


def aba(model: br.Model, q: torch.Tensor, v: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """``(B, nv)`` joint accelerations for velocity ``v`` and torque ``u``."""
    return br.forward_dynamics(model, br.forward_kinematics(model, q), v=v, tau=u)


ALGORITHMS: dict[str, Callable[..., torch.Tensor]] = {
    "jacobian": jacobian,
    "rnea": rnea,
    "crba": crba,
    "aba": aba,
}


@dataclass(frozen=True)
class Case:
    """One row: ``call()`` is timed and profiled; a counter row also compiles ``fn`` at ``args`` and each of ``sweep``.

    ``report`` turns the warm-up call's result into extra fields of the row (the IK row's accuracy).
    """

    row: str
    batch: int
    call: Callable[[], Any]
    fn: Callable[..., Any] | None = None
    args: tuple[torch.Tensor, ...] = ()
    sweep: tuple[tuple[torch.Tensor, ...], ...] = ()
    profiled: bool = True
    report: Callable[[Any], dict[str, Any]] | None = None


def load_model(name: str) -> br.Model:
    """``panda`` or ``g1`` (free-flyer root) from the tracked URDFs; float32 on the CPU."""
    root_joint = br.JointFreeFlyer() if name == "g1" else None
    return br.load(URDFS[name], root_joint=root_joint, dtype=DTYPE)


def counter_cases(device: torch.device) -> list[Case]:
    """The counter rows (module docstring), with seeded inputs on ``device``."""
    cases = []
    for name in ("panda", "g1"):
        model = load_model(name)
        generator = torch.Generator().manual_seed(0)
        pool = torch.stack([model.random_configuration(generator=generator) for _ in range(1024)])
        v, u = (torch.randn(1024, model.nv, generator=generator) for _ in range(2))
        model, pool, v, u = model.to(device=device), pool.to(device), v.to(device), u.to(device)

        def fk_joints(q: torch.Tensor, model: br.Model = model) -> torch.Tensor:
            return br.forward_kinematics(model, q).joint_transforms

        for batch in (1, LARGE_BATCH):
            cases.append(counter_case(f"fk_joints/{name}/B{batch}", fk_joints, (pool,), batch))
        for algo in ("jacobian", "rnea", "crba", "aba"):
            cases.append(counter_case(f"{algo}/{name}/B1", partial(ALGORITHMS[algo], model), (pool, v, u), 1))
    cases.append(ik_case(device))
    cases.append(constrained_ik_case(device))
    cases.append(constrained_ik_case(device, track=True))
    for size, count, batch in ((7, 30, 1), (35, 150, LARGE_BATCH)):
        generator = torch.Generator().manual_seed(0)
        root = torch.randn(LARGE_BATCH, size, size, generator=generator)
        programs = (
            root.mT @ root + torch.eye(size),
            torch.randn(LARGE_BATCH, size, generator=generator),
            torch.randn(LARGE_BATCH, count, size, generator=generator),
        )
        programs = tuple(pool.to(device) for pool in programs)
        cases.append(counter_case(f"qp/n{size}m{count}/B{batch}", solve_qp_in_unit_bounds, programs, batch))
    return cases


def solve_qp_in_unit_bounds(
    hessian: torch.Tensor, gradient: torch.Tensor, rows: torch.Tensor
) -> tuple[torch.Tensor, ...]:
    """One cold ``solve_qp`` with every row between -1 and 1: ``x: (B, N)`` and ``dual: (B, M)``."""
    bound = torch.ones_like(rows[..., 0])
    solution = br.optim.solve_qp(hessian, gradient, rows=rows, lower=-bound, upper=bound)
    return solution.x, solution.dual


def counter_case(row: str, fn: Callable[..., Any], pools: tuple[torch.Tensor, ...], batch: int) -> Case:
    """A counter row of ``fn`` on the first ``batch`` entries of each pool, swept over ``SWEEP``."""
    args = tuple(pool[:batch] for pool in pools)
    sweep = tuple(tuple(pool[:size] for pool in pools) for size in SWEEP)
    return Case(row, batch, lambda: fn(*args), fn, args, sweep)


def ik_case(device: torch.device) -> Case:
    """``ik/panda/B64``: ``solve_ik`` to 64 reachable ``panda_hand`` poses from the neutral configuration."""
    model = load_model("panda")
    generator = torch.Generator().manual_seed(0)
    q_target = torch.stack([model.random_configuration(generator=generator) for _ in range(64)])
    model, q_target = model.to(device=device), q_target.to(device)
    frame = model.frame_id(IK_FRAME)
    targets = br.forward_kinematics(model, q_target).frame_poses[:, frame]
    q0 = model.q_neutral.expand(64, model.nq).clone()

    def ik(q0: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        generator = torch.Generator(device=q0.device).manual_seed(1)
        return br.solve_ik(model, {IK_FRAME: targets}, q0=q0, seeds=5, generator=generator).x["q"]

    def accuracy(q: torch.Tensor) -> dict[str, float]:
        error = (br.forward_kinematics(model, q).frame_poses[:, frame, :3] - targets[:, :3]).norm(dim=-1)
        return {
            "success_fraction": (error < IK_SUCCESS_M).double().mean().item(),
            "worst_position_error_m": error.max().item(),
        }

    sweep = tuple((q0[:size], targets[:size]) for size in SWEEP)
    call = partial(ik, q0, targets)
    return Case("ik/panda/B64", 64, call, ik, (q0, targets), sweep, report=accuracy)


def constrained_ik_case(device: torch.device, *, track: bool = False) -> Case:
    """``ik_constrained/panda/B64``: ``SequentialQP`` for 20 steps to the poses of ``ik/panda/B64``, clear by 2 cm of
    a sphere beside the arm and of itself (the 8 closest sphere pairs as rows); with ``track``, ``track/panda/B64``:
    20 warm calls of ``optim.track`` with steps of at most 5 mrad instead."""
    model = load_model("panda")
    generator = torch.Generator().manual_seed(0)
    q_target = torch.stack([model.random_configuration(generator=generator) for _ in range(64)])
    model, q_target = model.to(device=device), q_target.to(device)
    frame = model.frame_id(IK_FRAME)
    targets = br.forward_kinematics(model, q_target).frame_poses[:, frame]
    q0 = model.q_neutral.expand(64, model.nq).clone()
    spheres, ignore = br.collision.load_spheres(br.collision.SPHERE_SETS / "panda.json", model)
    pairs = br.collision.self_collision_pairs(model, spheres, ignore=ignore)
    centre = torch.tensor([0.45, 0.25, 0.45], dtype=DTYPE, device=device)

    def to_obstacle(centers: torch.Tensor, radii: torch.Tensor) -> torch.Tensor:
        return (centers - centre).norm(dim=-1) - radii - 0.10

    # Built outside the measured call: a clearance term reads its frame ids back to the host once, which costs an
    # `aten::to` on CUDA and none on CPU, and the release tier checks both devices against one threshold.
    avoid = br.terms.sphere_clearance(spheres, to_obstacle, margin=0.02, hinge=False)
    apart = br.terms.self_clearance(spheres, pairs=pairs, margin=0.02, hinge=False, closest=8)

    def ik(q0: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        problem = br.terms.ik_problem(model, {IK_FRAME: targets}, q0=q0)
        problem.inequality("obstacle", avoid, upper=0.0)
        problem.inequality("self", apart, upper=0.0)
        if not track:
            return br.optim.run(problem, [br.optim.Phase(br.optim.SequentialQP(), 20, stop_every=0)]).x["q"]
        x, warm = {"q": q0}, None
        for _ in range(20):
            x, warm = br.optim.track(problem, x, warm=warm, step=0.005)
        return x["q"]

    def accuracy(q: torch.Tensor) -> dict[str, float]:
        state = br.forward_kinematics(model, q)
        error = (state.frame_poses[:, frame, :3] - targets[:, :3]).norm(dim=-1)
        distances = [to_obstacle(spheres.centers(state), spheres.radii)]
        distances.append(br.collision.self_distances(spheres, state, pairs=pairs))
        return {
            "success_fraction": (error < IK_SUCCESS_M).double().mean().item(),
            "worst_violation_m": (0.02 - torch.cat(distances, dim=-1)).amax().clamp_min(0.0).item(),
        }

    sweep = tuple((q0[:size], targets[:size]) for size in SWEEP)
    row = "track/panda/B64" if track else "ik_constrained/panda/B64"
    return Case(row, 64, partial(ik, q0, targets), ik, (q0, targets), sweep, report=accuracy)


def profile_counts(call: Callable[[], object]) -> tuple[int, int]:
    """``(ops, host_syncs)`` of one call.

    ``ops`` counts the profiler's ``aten::`` events with no ``aten::`` ancestor: each operator dispatched from Python or
    from an autograd node, not those nested in another operator (``aten::bmm`` inside ``aten::matmul``). It does not
    depend on the batch size, so it tracks the Python and dispatch overhead that dominates small batches.
    ``host_syncs`` counts every ``SYNC_OPS`` event, nested or not (``linalg.cholesky`` checks its result with one).
    """

    def top_level(event: Any) -> bool:
        parent = event.cpu_parent
        while parent is not None:
            if parent.name.startswith("aten::"):
                return False
            parent = parent.cpu_parent
        return True

    with profile(activities=[ProfilerActivity.CPU]) as prof:
        call()
    events = prof.events()
    ops = sum(1 for event in events if event.name.startswith("aten::") and top_level(event))
    return ops, sum(1 for event in events if event.name in SYNC_OPS)


def compile_counts(fn: Callable[..., Any], args: tuple, sweep: tuple[tuple, ...]) -> tuple[int, int]:
    """``(graph_breaks, recompiles)`` of ``fn`` (module docstring); dynamo state is reset before and after."""
    torch._dynamo.reset()
    graph_breaks = torch._dynamo.explain(fn)(*args).graph_break_count
    torch._dynamo.reset()
    dynamo_counters.clear()
    compiled = torch.compile(fn, backend="eager")
    # A collection inside a compiled call runs weakref callbacks (WeakKeyDictionary.remove) that dynamo counts as
    # frames; when one falls depends on allocation history, so the count would vary between runs.
    gc.collect()
    gc.disable()
    try:
        compiled(*args)
        after_first = dynamo_counters["frames"]["ok"]
        for sweep_args in sweep:
            compiled(*sweep_args)
        recompiles = dynamo_counters["frames"]["ok"] - after_first
    finally:
        gc.enable()
    torch._dynamo.reset()
    return graph_breaks, recompiles


def time_ms(call: Callable[[], object], sync: Callable[[], None], repeats: int) -> dict[str, float]:
    """``median_ms`` and ``min_ms`` per call and minor ``page_faults`` per call (``getrusage``)."""
    samples: list[float] = []
    faults = resource.getrusage(resource.RUSAGE_SELF).ru_minflt
    budget_start = time.perf_counter()
    while len(samples) < repeats:
        sync()
        start = time.perf_counter()
        call()
        sync()
        samples.append((time.perf_counter() - start) * 1e3)
        # at least 3 samples, except when one call alone exceeds the row budget (slow large-batch rows)
        enough = len(samples) >= 3 or samples[0] > ROW_BUDGET_S * 1e3
        if enough and time.perf_counter() - budget_start > ROW_BUDGET_S:
            break
    faults = (resource.getrusage(resource.RUSAGE_SELF).ru_minflt - faults) / len(samples)
    return {"median_ms": statistics.median(samples), "min_ms": min(samples), "page_faults": faults}


def measure(case: Case, args: argparse.Namespace, sync: Callable[[], None]) -> dict[str, Any]:
    """One case's row: a warm-up call, wall time unless ``--counters-only``, counters unless ``--timing-only``."""
    row: dict[str, Any] = {"row": case.row, "batch": case.batch, "loadavg": loadavg()}
    result = case.call()
    if case.report is not None:
        row |= case.report(result)
    if not args.counters_only:
        row |= time_ms(case.call, sync, args.repeats)
    if not args.timing_only:
        row["ops"], row["host_syncs"] = profile_counts(case.call) if case.profiled else (None, None)
        if case.fn is not None:
            row["graph_breaks"], row["recompiles"] = compile_counts(case.fn, case.args, case.sweep)
    print_row(row)
    return row


def check_thresholds(rows: list[dict[str, Any]], selected: set[str] | None) -> list[str]:
    """Each problem of the counter ``rows`` against ``THRESHOLDS`` (module docstring), as one line naming its fix."""
    if not THRESHOLDS.exists():
        return [
            f"no thresholds for torch {torch.__version__}: on a clean commit run `uv run python benchmarks/run.py "
            f"--counters-only --out {THRESHOLDS.relative_to(REPO_ROOT)}` and commit that file alone"
        ]
    limits = {row["row"]: row for row in json.loads(THRESHOLDS.read_text())["rows"]}
    measured = {row["row"]: row for row in rows}
    problems = [
        f"{key}: has a threshold but was not measured"
        for key in sorted(limits.keys() - measured.keys())
        if selected is None or key in selected
    ]
    for key, row in sorted(measured.items()):
        if key not in limits:
            problems.append(f"{key}: no threshold; measure it on a clean commit and add it in a commit of its own")
            continue
        for counter in COUNTERS:
            limit = limits[key].get(counter)
            if limit is not None and (row.get(counter) is None or row[counter] > limit):
                problems.append(f"{key}: {counter} {row.get(counter)} > threshold {limit}")
    return problems


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


def context(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    """The ``meta`` record: what ran, on what, under which load."""
    return {
        "commit": git("rev-parse", "--short", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
        "command": shlex.join(sys.argv),
        "package_file": os.path.relpath(br.__file__, REPO_ROOT),  # relative: no home directory in a tracked file
        "torch": torch.__version__,
        "python": platform.python_version(),
        "device": args.device,
        "device_name": device_name(device),
        "threads": torch.get_num_threads(),
        "affinity": sorted(os.sched_getaffinity(0)),
        "dtype": str(DTYPE),
        "malloc_env": {name: os.environ.get(name) for name in ("MALLOC_MMAP_THRESHOLD_", "MALLOC_TRIM_THRESHOLD_")},
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "loadavg_start": loadavg(),
    }


def print_row(row: dict[str, Any]) -> None:
    """One table line: row, median ms, page faults, ops, host syncs, graph breaks, recompiles (``-``: not measured)."""

    def cell(key: str, width: int, spec: str = "") -> str:
        value = row.get(key)
        return f"{'-' if value is None else format(value, spec):>{width}}"

    line = f"{row['row']:48} {cell('median_ms', 10, '.4f')} {cell('page_faults', 8, '.0f')} {cell('ops', 7)}"
    print(f"{line} {cell('host_syncs', 6)} {cell('graph_breaks', 6)} {cell('recompiles', 10)}", flush=True)


def parse_args() -> argparse.Namespace:
    """The command line, with the comma lists split."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--repeats", type=int, default=20, help="max timed calls per row (fewer once a row passes 2 s)")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--rows", default=None, help="measure only these comma-separated row ids")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--counters-only", action="store_true", help="counter rows only, no wall time; check thresholds")
    mode.add_argument("--timing-only", action="store_true", help="wall time only, no counters")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    args.rows = None if args.rows is None else set(args.rows.split(","))
    return args


def main() -> None:
    """Measure the rows, write the JSON, check the counters; exit 1 on a counter problem."""
    args = parse_args()
    if os.environ.get("BETTER_ROBOT_INTERNAL_CHECKS") == "1":
        sys.exit("BETTER_ROBOT_INTERNAL_CHECKS=1 adds host syncs and operators: unset it to benchmark")
    torch.set_num_threads(args.threads)  # noqa: TID251 - a benchmark pins its threads so rows compare; C-DEP-3 binds the library
    device = torch.device(args.device)
    sync = (lambda: torch.cuda.synchronize(device)) if device.type == "cuda" else (lambda: None)
    meta = context(args, device)
    started = time.perf_counter()
    print(f"{'row':48} {'median_ms':>10} {'faults':>8} {'ops':>7} {'syncs':>6} {'breaks':>6} {'recompiles':>10}")

    cases = [case for case in counter_cases(device) if args.rows is None or case.row in args.rows]
    rows = [measure(case, args, sync) for case in cases]

    meta["loadavg_end"] = loadavg()
    meta["total_s"] = round(time.perf_counter() - started, 1)
    out = args.out or REPO_ROOT / "benchmarks" / "results" / f"{meta['commit']}_{args.device}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"meta": meta, "rows": rows}, indent=1) + "\n")
    print(f"wrote {out} ({meta['total_s']} s)")
    if args.timing_only:
        return
    problems = check_thresholds(rows, args.rows)
    print(f"counters against {THRESHOLDS.relative_to(REPO_ROOT)}: {len(problems)} problems")
    for problem in problems:
        print(f"  {problem}")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
