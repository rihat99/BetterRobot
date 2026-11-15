"""Smoke test of ``benchmarks/speed.py``: a run writes the row schema and its context, and ``table`` reads it back.

The counters and their thresholds are checked by ``tests/contract/test_compile_counters.py``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEED = REPO_ROOT / "benchmarks" / "speed.py"


def test_speed_run_records_rows_and_context_and_table_compares_them(tmp_path: Path) -> None:
    out = tmp_path / "run.json"
    command = [sys.executable, str(SPEED), "run", "--functions", "forward_kinematics,solve_ik", "--robots", "panda7"]
    command += ["--batches", "64", "--modes", "eager", "--grad", "--repeats", "3", "--out", str(out)]
    env = {**os.environ, "BETTER_ROBOT_INTERNAL_CHECKS": "0"}
    for name in ("MALLOC_MMAP_THRESHOLD_", "MALLOC_TRIM_THRESHOLD_"):  # the run sets them itself
        env.pop(name, None)
    completed = subprocess.run(
        command, cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=600, check=False
    )
    assert completed.returncode == 0, completed.stdout[-2000:] + completed.stderr[-2000:]
    result = json.loads(out.read_text())
    meta, (fk, fk_grad, ik) = result["meta"], result["rows"]
    context = {"commit", "src_root", "src_tree", "torch", "host", "device_name", "affinity", "siblings_busy_end"}
    assert context <= meta.keys()
    assert all(meta["malloc_env"].values()), "the run re-executes itself with the allocator thresholds fixed"
    schema = {"library", "function", "robot", "device", "dtype", "batch", "mode", "seconds_per_call"}
    assert schema | {"seconds_spread", "samples", "page_faults", "grad", "nv"} <= fk.keys()
    assert (fk["function"], fk_grad["function"], ik["function"]) == ("forward_kinematics",) * 2 + ("solve_ik",)
    assert (fk["grad"], fk_grad["grad"], fk["nv"], fk["samples"] >= 3) == (False, True, 7, True)
    assert ik["iterations"] > 0 and ik["success_targets"] == 1024, "a batch below the pool reports the pool's success"

    table = [sys.executable, str(SPEED), "table", str(out), "--before", str(out)]
    completed = subprocess.run(table, cwd=REPO_ROOT, capture_output=True, text=True, timeout=60, check=True)
    rows = [
        line for line in completed.stdout.splitlines() if line.startswith("| forward") or line.startswith("| solve")
    ]
    assert len(rows) == 3 and all(" | 1x~ | " in line for line in rows), completed.stdout
