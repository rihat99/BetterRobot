"""Smoke test: ``benchmarks/run.py`` imports and runs one tiny config (not a benchmark; runs in the default suite)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_benchmarks_run_py_runs_one_tiny_config(tmp_path):
    pytest.importorskip("robot_descriptions")
    out = tmp_path / "run.json"
    command = [sys.executable, str(REPO_ROOT / "benchmarks" / "run.py"), "--device", "cpu", "--models", "panda",
               "--batches", "1", "--algos", "fk", "--repeats", "1", "--out", str(out)]
    completed = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True, timeout=600, check=False)
    assert completed.returncode == 0, completed.stderr[-2000:]
    rows = json.loads(out.read_text())["rows"]
    assert any(row["impl"] == "better_robot" and "error" not in row for row in rows)
