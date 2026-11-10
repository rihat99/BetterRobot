"""Smoke test of ``benchmarks/run.py``: a large-batch row gets wall time, page faults and the fixed-allocator rerun."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ROW = "fk/panda/B1024/forward/better_robot"


def test_runner_times_a_large_batch_row_with_both_allocators(tmp_path: Path) -> None:
    out = tmp_path / "run.json"
    command = [sys.executable, str(REPO_ROOT / "benchmarks" / "run.py"), "--timing-only", "--rows", ROW]
    command += ["--models", "panda", "--algos", "fk", "--batches", "1024", "--repeats", "3", "--out", str(out)]
    env = {**os.environ, "BETTER_ROBOT_INTERNAL_CHECKS": "0"}
    for name in ("MALLOC_MMAP_THRESHOLD_", "MALLOC_TRIM_THRESHOLD_"):  # set, they would stop the rerun
        env.pop(name, None)
    completed = subprocess.run(
        command, cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=600, check=False
    )
    assert completed.returncode == 0, completed.stdout[-2000:] + completed.stderr[-2000:]
    result = json.loads(out.read_text())
    meta, (row,) = result["meta"], result["rows"]
    assert row["row"] == ROW
    assert {"median_ms", "page_faults", "fixed_malloc_median_ms", "fixed_malloc_page_faults"} <= row.keys()
    assert {"commit", "dirty", "command", "torch", "device_name", "loadavg_start", "loadavg_end"} <= meta.keys()
