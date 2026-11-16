"""Every example runs to completion on CPU without a browser: the viewer examples and example 07 with ``--check``
build their scene on the stand-in server (``better_robot.testing.FakeServer``), and example 07 also runs
``--no-viewer``. The avoidance demos that take ``--robot`` also run their scripted path on the G1."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_RUNS = [
    *((f"examples/{path.name}", []) for path in sorted((_ROOT / "examples").glob("0*.py")) if path.name[:3] != "07_"),
    *(
        (f"examples/{path.name}", [flag])
        for path in (_ROOT / "examples").glob("07_*.py")
        for flag in ("--no-viewer", "--check")
    ),
    *((f"examples/viewer/{path.name}", ["--check"]) for path in sorted((_ROOT / "examples" / "viewer").glob("0*.py"))),
    *(
        (f"examples/viewer/{name}", ["--robot", "g1", "--check"])
        for name in ("07_avoid_floor.py", "08_avoid_obstacle.py", "09_avoid_self_collision.py")
    ),
]


@pytest.mark.slow
@pytest.mark.parametrize(("example", "arguments"), _RUNS, ids=[" ".join([name, *flags]) for name, flags in _RUNS])
def test_example_runs_headless(example: str, arguments: list[str]) -> None:
    environment = {**os.environ, "OMP_NUM_THREADS": "2", "CUDA_VISIBLE_DEVICES": ""}
    completed = subprocess.run(
        [sys.executable, str(_ROOT / example), *arguments],
        cwd=_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
