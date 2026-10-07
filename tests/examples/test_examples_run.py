"""Every headless example runs to completion on CPU (the viewer example is skipped with ``--no-viewer``)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES = sorted(path.name for path in (_ROOT / "examples").glob("0*.py"))


@pytest.mark.parametrize("name", _EXAMPLES)
def test_example_runs_headless(name: str) -> None:
    arguments = ["--no-viewer"] if name.startswith("07_") else []
    environment = {**os.environ, "OMP_NUM_THREADS": "2", "CUDA_VISIBLE_DEVICES": ""}
    completed = subprocess.run(
        [sys.executable, str(_ROOT / "examples" / name), *arguments],
        cwd=_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
