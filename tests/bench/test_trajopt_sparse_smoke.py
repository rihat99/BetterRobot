"""Normal-suite smoke for the trajectory benchmark harness."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tests.bench.bench_trajopt_sparse import _definition, _failure_status, _run_parent


_BENCHMARK = Path(__file__).with_name("bench_trajopt_sparse.py")


def test_trajopt_sparse_dense_t50_one_update(tmp_path: Path) -> None:
    """Build the manifold branching-tree problem and complete one dense LM update."""
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
            "dense",
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
    assert result["path"] == "dense"
    assert result["route"] == "dense"
    assert result["route_reason"] == "forced_dense"
    assert result["route_detail"]
    assert result["horizon"] == 50
    assert result["updates_per_solve"] == 1
    assert result["warmup_solve_count"] == 0
    assert result["measurement_solve_count"] == 1
    assert result["active_envelope_coordinates"] == 127
    assert result["measured_solves"][0]["success"]


def test_trajopt_sparse_definition_is_complete() -> None:
    definition = _definition()
    assert definition["horizons"] == [50, 125, 250, 500]
    assert definition["paths"] == ["dense"]
    assert definition["measurement"]["case_timeout_seconds"] == 600
    assert definition["measurement"]["address_space_limit_bytes"] == 16 * 2**30


def test_unknown_process_signal_is_not_mislabeled_as_oom() -> None:
    assert _failure_status(-9, "") == "HARNESS_ERROR"
    assert _failure_status(-11, "segmentation fault") == "HARNESS_ERROR"
    assert _failure_status(1, "MemoryError") == "DNF_OOM"


def test_duplicate_case_selectors_are_rejected_before_running() -> None:
    args = SimpleNamespace(quick=True, path=["dense", "dense"], horizon=None)
    with pytest.raises(ValueError, match="--path selectors must not contain duplicates"):
        _run_parent(args)
