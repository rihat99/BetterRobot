"""The measuring parts of a benchmark runner: operator counts, wall time, the thresholds check and the run's context.

BetterRobot's and BetterHuman's ``benchmarks/run.py`` (the counter gate, principle 5.8) each keep their rows and
call these. A counter row has four counters, deterministic for one torch version and device: ``ops`` and
``host_syncs`` come from ``profile_counts``; ``graph_breaks`` and ``recompiles`` need ``torch._dynamo`` and
``torch.compile``, which contract C-PERF-5 keeps out of the library, so each runner counts them itself.
``check_thresholds`` compares the four with a thresholds file, the runner's ``--counters-only`` output from a clean
commit. Wall time (``wall_time``) is recorded, never gated: ``load_average`` sits beside every row and ``context``
records what ran, on what, under which load.
"""

from __future__ import annotations

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
from collections.abc import Callable, Collection
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
from torch.profiler import ProfilerActivity, profile

if TYPE_CHECKING:
    from torch.autograd.profiler_util import EventList

__all__ = [
    "HEADER",
    "check_thresholds",
    "context",
    "device_name",
    "git",
    "load_average",
    "profile_counts",
    "row_line",
    "wall_time",
]

_COUNTERS = ("graph_breaks", "recompiles", "ops", "host_syncs")
_SYNC_OPS = frozenset({"aten::_local_scalar_dense", "aten::nonzero"})  # item/bool/float/int of a tensor; mask indexing
_ROW_BUDGET_S = 2.0
# The header of row_line's table.
HEADER = f"{'row':48} {'median_ms':>10} {'faults':>8} {'ops':>7} {'syncs':>6} {'breaks':>6} {'recompiles':>10}"


def profile_counts(call: Callable[[], object]) -> tuple[int, int]:
    """``(ops, host_syncs)`` of one call.

    ``ops`` counts the profiler's ``aten::`` events with no ``aten::`` ancestor: each operator dispatched from Python
    or from an autograd node, not those nested in another operator (``aten::bmm`` inside ``aten::matmul``). It does
    not depend on the batch size, so it tracks the Python and dispatch overhead that dominates small batches.
    ``host_syncs`` counts every event of an operator that makes the host wait for a CUDA queue, nested or not:
    ``aten::_local_scalar_dense`` (``item``, ``bool``, ``float`` or ``int`` of a tensor) and ``aten::nonzero`` (mask
    indexing); ``linalg.cholesky`` checks its result with one.
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
    # events() is typed optional; once the block has run it holds every event the call dispatched.
    events = cast("EventList", prof.events())
    ops = sum(1 for event in events if event.name.startswith("aten::") and top_level(event))
    return ops, sum(1 for event in events if event.name in _SYNC_OPS)


def wall_time(call: Callable[[], object], sync: Callable[[], None], *, repeats: int) -> dict[str, float]:
    """``median_ms`` and ``min_ms`` per call and minor ``page_faults`` per call (``getrusage``) of up to ``repeats``
    calls, at least 3, fewer once the row has run 2 s; ``sync`` waits for the device before and after each call."""
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
        enough = len(samples) >= 3 or samples[0] > _ROW_BUDGET_S * 1e3
        if enough and time.perf_counter() - budget_start > _ROW_BUDGET_S:
            break
    faults = (resource.getrusage(resource.RUSAGE_SELF).ru_minflt - faults) / len(samples)
    return {"median_ms": statistics.median(samples), "min_ms": min(samples), "page_faults": faults}


def check_thresholds(rows: list[dict[str, Any]], thresholds: Path, *, selected: Collection[str] | None) -> list[str]:
    """Each problem of the counter ``rows`` against the ``thresholds`` file, as one line naming its fix.

    A row above a threshold, a row without one, and a row with one that was not measured (among ``selected``, every
    row when ``None``) are problems, and so is a missing file; ``thresholds`` sits at
    ``<root>/benchmarks/thresholds/``, and its message names the command that writes it.
    """
    if not thresholds.exists():
        relative = thresholds.relative_to(thresholds.parents[2])
        return [
            f"no thresholds for torch {torch.__version__}: on a clean commit run `uv run python benchmarks/run.py "
            f"--counters-only --out {relative}` and commit that file alone"
        ]
    limits = {row["row"]: row for row in json.loads(thresholds.read_text())["rows"]}
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
        for counter in _COUNTERS:
            limit = limits[key].get(counter)
            if limit is not None and (row.get(counter) is None or row[counter] > limit):
                problems.append(f"{key}: {counter} {row.get(counter)} > threshold {limit}")
    return problems


def git(root: Path, *arguments: str) -> str:
    """Stdout of one git command in the checkout at ``root``, stripped."""
    command = ["git", "-C", str(root), *arguments]
    return subprocess.run(command, capture_output=True, text=True, check=True).stdout.strip()


def load_average() -> str:
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


def context(root: Path, device: torch.device, *, dtype: torch.dtype) -> dict[str, Any]:
    """The ``meta`` record of a run in the checkout at ``root``: commit, dirty flag, command, torch, Python, device,
    CPU or GPU, threads, affinity, dtype, date and the load average at start; a runner adds its own keys."""
    return {
        "commit": git(root, "rev-parse", "--short", "HEAD"),
        "dirty": bool(git(root, "status", "--porcelain")),
        "command": shlex.join(sys.argv),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "device": device.type,
        "device_name": device_name(device),
        "threads": torch.get_num_threads(),
        "affinity": sorted(os.sched_getaffinity(0)),
        "dtype": str(dtype),
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "loadavg_start": load_average(),
    }


def row_line(row: dict[str, Any]) -> str:
    """One line of the table ``HEADER`` heads: row, median ms, page faults, ops, host syncs, graph breaks,
    recompiles (``-``: not measured)."""

    def cell(key: str, width: int, spec: str = "") -> str:
        value = row.get(key)
        return f"{'-' if value is None else format(value, spec):>{width}}"

    line = f"{row['row']:48} {cell('median_ms', 10, '.4f')} {cell('page_faults', 8, '.0f')} {cell('ops', 7)}"
    return f"{line} {cell('host_syncs', 6)} {cell('graph_breaks', 6)} {cell('recompiles', 10)}"
