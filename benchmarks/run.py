"""The benchmark runner: counters and wall time of BetterRobot's hot entry points, with the context to read them.

Usage (from the repository root)::

    uv run python benchmarks/run.py --counters-only     # the counter rows, checked against the thresholds
    uv run python benchmarks/run.py                     # counter rows and the sweep, with wall time
    uv run python benchmarks/run.py --device cuda --batches 1,4096 --models panda,tree200 --algos fk,rnea

Counter rows (principle 5.8; record 0011) are fixed functions on the tracked Panda and G1 (``tests/fixtures/robots``):
``fk_joints/{panda,g1}/B{1,1024}`` (``forward_kinematics(model, q).joint_transforms``), ``ik/panda/B64``
(``solve_ik`` to 64 ``panda_hand`` poses, 4 restarts, fixed generators; it also records ``success_fraction`` and
``worst_position_error_m``) and ``{jacobian,rnea,crba,aba}/{panda,g1}/B1``. Each has four counters, deterministic for
one torch version and device: ``graph_breaks`` (``torch._dynamo.explain``), ``recompiles`` (frames compiled when
``torch.compile(fn, backend="eager")``, first called at the row's batch, is called at each batch of ``SWEEP``),
``ops`` (top-level ``aten::`` operators of one call) and ``host_syncs`` (calls of the operators that make the host
wait for a CUDA queue, ``SYNC_OPS``). The IK row has no ``ops`` or ``host_syncs``: one profiled solve records about
560 000 operator events and takes a minute. The run exits 1 when a counter row exceeds its threshold in
``benchmarks/thresholds/torch-<version>.json``, has none, or no such file exists for the installed torch; a thresholds
file is this runner's ``--counters-only`` output from a clean commit, changed only in a commit of its own.

Sweep rows (principle 5.7) time ``{fk,jacobian,rnea,crba,aba}`` over ``--models`` x ``--batches`` x
(``forward`` under ``no_grad``, ``forward_backward`` of ``sum(output)`` with respect to ``q``), with ``ops`` and
``host_syncs`` of one call, plus Pinocchio float64 reference rows at batch 1 when ``pinocchio`` imports.

Wall time is the median of up to ``--repeats`` calls (at least 3; fewer once a row passes ``ROW_BUDGET_S``) after one
warm-up call, with the minor page faults per call; the counters are taken after it. It is recorded, never gated: the load average
sits beside every row. For rows at batch >= ``LARGE_BATCH`` the runner re-executes itself with glibc's allocator
thresholds fixed (``FIXED_MALLOC``; they must be set before the process starts) and records that time and its page
faults beside the default ones: the default allocator's dynamic mmap threshold depends on what the process freed
before, so a large-batch default time can double from page faults alone. A "no slower" verdict at those rows reads
the fixed-threshold number.

Every number is float32, ``--threads`` Torch threads (default 1), ``BETTER_ROBOT_INTERNAL_CHECKS`` off. Output:
``{"meta": {...}, "rows": [{"row": id, ...}, ...]}`` in ``--out`` (default ``benchmarks/results/<commit>_<device>.json``,
never tracked); ``meta`` records commit, dirty flag, command, torch, Python, device, CPU, threads, affinity, the
allocator variables and the load average at start and end.
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
import tempfile
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
from better_robot.io import build_model

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
FIXED_MALLOC = {"MALLOC_MMAP_THRESHOLD_": "134217728", "MALLOC_TRIM_THRESHOLD_": "268435456"}
IK_FRAME = "panda_hand"
IK_SUCCESS_M = 1e-3


def fk(model: br.Model, q: torch.Tensor, v: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """``(B, njoints, 7)`` joint poses."""
    return br.forward_kinematics(model, q).joint_poses


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
    "fk": fk,
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
    fields: dict[str, Any]
    fn: Callable[..., Any] | None = None
    args: tuple[torch.Tensor, ...] = ()
    sweep: tuple[tuple[torch.Tensor, ...], ...] = ()
    profiled: bool = True
    report: Callable[[Any], dict[str, Any]] | None = None


def random_tree(n_moving: int, seed: int = 0) -> br.Model:
    """Seeded fixed-base random tree of ``n_moving`` revolute joints; each parent is uniform over earlier bodies."""
    builder = br.ModelBuilder(f"tree{n_moving}")
    generator = torch.Generator().manual_seed(seed)

    def origin() -> torch.Tensor:
        translation = 0.1 * torch.randn(3, generator=generator, dtype=torch.float64)
        return torch.cat([translation, torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64)])

    builder.add_body("b0", mass=1.0, inertia=0.01 * torch.eye(3, dtype=torch.float64))
    for i in range(1, n_moving + 1):
        parent = int(torch.randint(0, i, (1,), generator=generator))
        axis = torch.nn.functional.normalize(torch.randn(3, generator=generator, dtype=torch.float64), dim=0)
        builder.add_body(f"b{i}", mass=1.0, inertia=0.01 * torch.eye(3, dtype=torch.float64))
        builder.add_revolute(
            f"j{i}", parent=f"b{parent}", child=f"b{i}", axis=axis, origin=origin(), lower=-3.0, upper=3.0
        )
    return build_model(builder.finalize())


def load_model(name: str) -> br.Model:
    """``panda`` or ``g1`` (free-flyer root) from the tracked URDFs, or ``tree<N>``; float32 on the CPU."""
    if name in URDFS:
        root_joint = br.JointFreeFlyer() if name == "g1" else None
        return br.load(URDFS[name], root_joint=root_joint, dtype=DTYPE)
    if name.startswith("tree"):
        return random_tree(int(name[4:])).to(dtype=DTYPE)
    raise ValueError(f"unknown model {name!r}: use panda, g1 or tree<N>")


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
    return cases


def counter_case(row: str, fn: Callable[..., Any], pools: tuple[torch.Tensor, ...], batch: int) -> Case:
    """A counter row of ``fn`` on the first ``batch`` entries of each pool, swept over ``SWEEP``."""
    args = tuple(pool[:batch] for pool in pools)
    sweep = tuple(tuple(pool[:size] for pool in pools) for size in SWEEP)
    return Case(row, batch, lambda: fn(*args), {}, fn, args, sweep)


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
        return br.solve_ik(model, {IK_FRAME: targets}, q0=q0, restarts=4, generator=generator).x["q"]

    def accuracy(q: torch.Tensor) -> dict[str, float]:
        error = (br.forward_kinematics(model, q).frame_poses[:, frame, :3] - targets[:, :3]).norm(dim=-1)
        return {
            "success_fraction": (error < IK_SUCCESS_M).double().mean().item(),
            "worst_position_error_m": error.max().item(),
        }

    sweep = tuple((q0[:size], targets[:size]) for size in SWEEP)
    call = partial(ik, q0, targets)
    return Case("ik/panda/B64", 64, call, {}, ik, (q0, targets), sweep, profiled=False, report=accuracy)


def sweep_cases(args: argparse.Namespace, device: torch.device) -> list[Case]:
    """The sweep rows: ``--algos`` x ``--models`` x ``--batches`` x both modes, seeded inputs on ``device``."""
    cases = []
    for model_name in args.models:
        model = load_model(model_name)
        generator = torch.Generator().manual_seed(1)
        base = torch.stack([model.random_configuration(generator=generator) for _ in range(16)])
        model = model.to(device=device)
        for batch in args.batches:
            q = base[torch.arange(batch) % 16].to(device)
            v, u = (torch.randn(batch, model.nv, generator=generator).to(device) for _ in range(2))
            for algo in args.algos:
                fn = partial(ALGORITHMS[algo], model)
                for mode in ("forward", "forward_backward"):
                    fields = {"algo": algo, "model": model_name, "njoints": model.njoints, "nq": model.nq}
                    fields |= {"mode": mode, "impl": "better_robot"}
                    row = f"{algo}/{model_name}/B{batch}/{mode}/better_robot"
                    cases.append(Case(row, batch, partial(sweep_call, fn, q, v, u, backward=mode != "forward"), fields))
    return cases


def sweep_call(
    fn: Callable[..., torch.Tensor], q: torch.Tensor, v: torch.Tensor, u: torch.Tensor, *, backward: bool
) -> None:
    """One sweep call: forward under ``no_grad``, or forward and the backward of ``sum(output)`` with respect to ``q``."""
    if backward:
        fn(q.detach().requires_grad_(True), v, u).sum().backward()
        return
    with torch.no_grad():
        fn(q, v, u)


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
    compiled(*args)
    after_first = dynamo_counters["frames"]["ok"]
    for sweep_args in sweep:
        compiled(*sweep_args)
    recompiles = dynamo_counters["frames"]["ok"] - after_first
    torch._dynamo.reset()
    return graph_breaks, recompiles


def time_ms(call: Callable[[], object], sync: Callable[[], None], repeats: int, inner: int = 1) -> dict[str, float]:
    """``median_ms`` and ``min_ms`` per call and minor ``page_faults`` per call (``getrusage``)."""
    samples: list[float] = []
    faults = resource.getrusage(resource.RUSAGE_SELF).ru_minflt
    budget_start = time.perf_counter()
    while len(samples) < repeats:
        sync()
        start = time.perf_counter()
        for _ in range(inner):
            call()
        sync()
        samples.append((time.perf_counter() - start) * 1e3 / inner)
        # at least 3 samples, except when one call alone exceeds the row budget (slow large-batch rows)
        if (len(samples) >= 3 or samples[0] > ROW_BUDGET_S * 1e3) and time.perf_counter() - budget_start > ROW_BUDGET_S:
            break
    faults = (resource.getrusage(resource.RUSAGE_SELF).ru_minflt - faults) / (len(samples) * inner)
    return {"median_ms": statistics.median(samples), "min_ms": min(samples), "page_faults": faults}


def measure(case: Case, args: argparse.Namespace, sync: Callable[[], None]) -> dict[str, Any]:
    """The output row of one case: a warm-up call, wall time unless ``--counters-only``, counters unless ``--timing-only``."""
    row: dict[str, Any] = {"row": case.row, "batch": case.batch, **case.fields, "loadavg": loadavg()}
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


def measure_sweep(cases: list[Case], args: argparse.Namespace, sync: Callable[[], None]) -> list[dict[str, Any]]:
    """Measure the sweep rows; a row that raises records its error, and larger batches of a too-slow row are skipped."""
    rows: list[dict[str, Any]] = []
    slow: dict[tuple, int] = {}  # (algo, model, mode) -> batch whose call exceeded --max-call-s
    for case in cases:
        key = (case.fields["algo"], case.fields["model"], case.fields["mode"])
        if key in slow:
            error = f"skipped: batch {slow[key]} exceeded --max-call-s"
            rows.append({"row": case.row, "batch": case.batch, **case.fields, "error": error})
            continue
        try:
            rows.append(measure(case, args, sync))
        except Exception as exc:  # noqa: BLE001 - one row's failure (memory at a large batch) is recorded, not fatal
            rows.append({"row": case.row, "batch": case.batch, **case.fields, "error": f"{type(exc).__name__}: {exc}"})
            print(f"{case.row:<44} ERROR {rows[-1]['error'][:80]}")
            continue
        if rows[-1].get("median_ms", 0.0) > args.max_call_s * 1e3:
            slow[key] = case.batch
    return rows


def pinocchio_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Pinocchio float64 CPU reference rows: batch 1, forward only, the same URDFs, one call per sample of 200."""
    import numpy as np  # noqa: PLC0415 - with the optional Pinocchio oracle below
    import pinocchio as pin  # noqa: PLC0415 - the oracle of the test-oracles group, imported only for its rows

    rows = []
    for name in (m for m in args.models if m in URDFS):
        root = (pin.JointModelFreeFlyer(),) if name == "g1" else ()
        pmodel = pin.buildModelFromUrdf(str(URDFS[name]), *root)
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
            row = {"row": f"{algo}/{name}/B1/forward/pinocchio", "algo": algo, "model": name, "njoints": pmodel.njoints}
            if args.rows is not None and row["row"] not in args.rows:
                continue
            row |= {"nq": pmodel.nq, "batch": 1, "mode": "forward", "impl": "pinocchio", "loadavg": loadavg()}
            calls[algo]()
            row |= time_ms(calls[algo], lambda: None, max(args.repeats, 50), inner=200)
            rows.append(row)
            print_row(row)
    return rows


def fixed_malloc_numbers(rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    """Re-run the rows at batch >= ``LARGE_BATCH`` in a child with ``FIXED_MALLOC`` set; add its time and faults."""
    selected = [row["row"] for row in rows if row.get("batch", 0) >= LARGE_BATCH and "median_ms" in row]
    if not selected:
        return
    with tempfile.TemporaryDirectory() as directory:
        out = Path(directory) / "fixed.json"
        command = [sys.executable, str(Path(__file__).resolve()), "--timing-only", "--rows", ",".join(selected)]
        command += ["--device", args.device, "--threads", str(args.threads), "--repeats", str(args.repeats)]
        command += ["--models", ",".join(args.models), "--algos", ",".join(args.algos)]
        command += ["--batches", ",".join(map(str, args.batches)), "--out", str(out)]
        print(f"fixed allocator thresholds ({' '.join(f'{k}={v}' for k, v in FIXED_MALLOC.items())}):", flush=True)
        subprocess.run(command, env={**os.environ, **FIXED_MALLOC}, check=True)
        fixed = {row["row"]: row for row in json.loads(out.read_text())["rows"]}
    for row in rows:
        if row["row"] in fixed and "median_ms" in fixed[row["row"]]:
            row["fixed_malloc_median_ms"] = fixed[row["row"]]["median_ms"]
            row["fixed_malloc_page_faults"] = fixed[row["row"]]["page_faults"]


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
        "package_file": br.__file__,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "device": args.device,
        "device_name": device_name(device),
        "threads": torch.get_num_threads(),
        "affinity": sorted(os.sched_getaffinity(0)),
        "dtype": str(DTYPE),
        "malloc_env": {name: os.environ.get(name) for name in FIXED_MALLOC},
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "loadavg_start": loadavg(),
    }


def print_row(row: dict[str, Any]) -> None:
    """One table line: row, median ms, page faults, ops, host syncs, graph breaks, recompiles (``-``: not measured)."""

    def cell(key: str, width: int, spec: str = "") -> str:
        value = row.get(key)
        return f"{'-' if value is None else format(value, spec):>{width}}"

    line = f"{row['row']:44} {cell('median_ms', 10, '.4f')} {cell('page_faults', 8, '.0f')} {cell('ops', 7)}"
    print(f"{line} {cell('host_syncs', 6)} {cell('graph_breaks', 6)} {cell('recompiles', 10)}", flush=True)


def parse_args() -> argparse.Namespace:
    """The command line, with the comma lists split."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batches", default="1,64,256,4096", help="sweep batch sizes")
    parser.add_argument("--models", default="panda,g1,tree25,tree200", help="sweep models: panda, g1, tree<N>")
    parser.add_argument("--algos", default="fk,jacobian,rnea,crba,aba", help="sweep algorithms")
    parser.add_argument("--repeats", type=int, default=20, help="max timed calls per row (fewer once a row passes 2 s)")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--rows", default=None, help="measure only these comma-separated row ids")
    parser.add_argument("--max-call-s", type=float, default=20.0, help="skip larger sweep batches past this call time")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--counters-only", action="store_true", help="counter rows only, no wall time; check thresholds")
    mode.add_argument("--timing-only", action="store_true", help="wall time only, no counters")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    args.batches = [int(b) for b in args.batches.split(",")]
    args.models = args.models.split(",")
    args.algos = args.algos.split(",")
    unknown = set(args.algos) - ALGORITHMS.keys()
    if unknown:
        parser.error(f"unknown algos {sorted(unknown)}: use {','.join(ALGORITHMS)}")
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
    print(f"{'row':44} {'median_ms':>10} {'faults':>8} {'ops':>7} {'syncs':>6} {'breaks':>6} {'recompiles':>10}")

    cases = [case for case in counter_cases(device) if args.rows is None or case.row in args.rows]
    counter_rows = [measure(case, args, sync) for case in cases]
    rows = list(counter_rows)
    if not args.counters_only:
        cases = [case for case in sweep_cases(args, device) if args.rows is None or case.row in args.rows]
        rows += measure_sweep(cases, args, sync)
        if 1 in args.batches:
            try:
                rows += pinocchio_rows(args)
            except ImportError as exc:
                print(f"pinocchio rows not measured: {exc}")
        if not any(os.environ.get(name) for name in FIXED_MALLOC):
            fixed_malloc_numbers(rows, args)

    meta["loadavg_end"] = loadavg()
    meta["total_s"] = round(time.perf_counter() - started, 1)
    out = args.out or REPO_ROOT / "benchmarks" / "results" / f"{meta['commit']}_{args.device}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"meta": meta, "rows": rows}, indent=1) + "\n")
    print(f"wrote {out} ({meta['total_s']} s)")
    if args.timing_only:
        return
    problems = check_thresholds(counter_rows, args.rows)
    print(f"counters against {THRESHOLDS.relative_to(REPO_ROOT)}: {len(problems)} problems")
    for problem in problems:
        print(f"  {problem}")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
