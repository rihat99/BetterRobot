"""Normal-suite smoke for the M5 Phase-C trajectory benchmark harness."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tests.bench.bench_trajopt_sparse import _failure_status, _run_parent


_BENCHMARK = Path(__file__).with_name("bench_trajopt_sparse.py")
_BASELINE = Path(__file__).with_name("baselines") / "trajopt_sparse_cpu.json"


def test_trajopt_sparse_structured_t50_one_update(tmp_path: Path) -> None:
    """Build the real SMPL problem and complete one structured LM update."""
    output = tmp_path / "case.json"
    environment = os.environ.copy()
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "PYTHONHASHSEED": "20260717",
        }
    )
    subprocess.run(
        [
            sys.executable,
            str(_BENCHMARK),
            "--child",
            "--path",
            "structured",
            "--horizon",
            "50",
            "--updates",
            "1",
            "--warmups",
            "0",
            "--measurements",
            "1",
            "--skip-cold",
            "--case-output",
            str(output),
        ],
        check=True,
        cwd=_BENCHMARK.parents[2],
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
    )
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["status"] == "SUCCESS"
    assert result["path"] == "structured"
    assert result["route"] == "banded"
    assert result["horizon"] == 50
    assert result["updates_per_solve"] == 1
    assert result["warmup_solve_count"] == 0
    assert result["measurement_solve_count"] == 1
    assert result["active_envelope_coordinates"] == 127
    assert result["measured_solves"][0]["success"]


def test_trajopt_sparse_baseline_schema_is_pending() -> None:
    baseline = json.loads(_BASELINE.read_text(encoding="utf-8"))
    assert baseline["_schema_version"] == 1
    assert baseline["_status"] == "PENDING_MEASUREMENT"
    assert baseline["benchmark"] == "m5_sparse_trajectory_cpu"
    assert baseline["definition"]["horizons"] == [50, 125, 250, 500]
    assert baseline["gpu"]["status"] == "pending_m6"


def test_unknown_process_signal_is_not_mislabeled_as_oom() -> None:
    assert _failure_status(-9, "") == "HARNESS_ERROR"
    assert _failure_status(-11, "segmentation fault") == "HARNESS_ERROR"
    assert _failure_status(1, "MemoryError") == "DNF_OOM"


def test_duplicate_case_selectors_are_rejected_before_running() -> None:
    args = SimpleNamespace(quick=True, path=["structured", "structured"], horizon=None)
    with pytest.raises(ValueError, match="--path selectors must not contain duplicates"):
        _run_parent(args)
