"""Headless smoke coverage for the batched and differentiable IK examples."""

from __future__ import annotations

import ast
import math
import os
from pathlib import Path
import subprocess
import sys


_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES = _ROOT / "examples"


def _run(name: str, *arguments: str) -> str:
    environment = os.environ.copy()
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "PYTHONHASHSEED": "0",
        }
    )
    completed = subprocess.run(
        [sys.executable, str(_EXAMPLES / name), *arguments],
        check=True,
        cwd=_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return completed.stdout


def test_batched_ik_example_runs_headless() -> None:
    output = _run("03_batched_ik.py", "--batch-size", "8", "--max-iterations", "5", "--device", "cpu")
    assert "Device: cpu" in output
    assert "Problems: 8" in output
    assert "Convergence rate: 100.0%" in output
    assert "Wall time:" in output


def test_differentiable_ik_example_runs_headless_and_backpropagates() -> None:
    output = _run("06_differentiable_ik.py", "--max-iterations", "20")
    assert "Converged: True" in output
    gradient_line = next(line for line in output.splitlines() if line.startswith("Target-pose gradient:"))
    gradient = ast.literal_eval(gradient_line.partition(":")[2].strip())
    assert len(gradient) == 7
    assert all(math.isfinite(value) for value in gradient)
    assert any(abs(value) > 1e-8 for value in gradient)
