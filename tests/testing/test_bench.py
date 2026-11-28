"""``testing.bench``: the thresholds check names each problem, operators and host syncs are counted, and a row prints
under the header."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from better_robot.exceptions import BetterRobotError
from better_robot.testing import bench


def test_check_thresholds_names_each_problem(tmp_path: Path) -> None:
    thresholds = tmp_path / "benchmarks" / "thresholds" / "torch-0.json"
    assert (
        f"--counters-only --out {thresholds.relative_to(tmp_path)}"
        in bench.check_thresholds([], thresholds, selected=None)[0]
    )
    thresholds.parent.mkdir(parents=True)
    limits = {"ops": 10, "host_syncs": 0, "graph_breaks": 0, "recompiles": 0}
    thresholds.write_text(json.dumps({"rows": [{"row": "fk", **limits}, {"row": "ik", **limits}]}))
    rows = [{"row": "fk", **limits, "ops": 11, "recompiles": None}, {"row": "new", **limits}]
    problems = bench.check_thresholds(rows, thresholds, selected=None)
    assert [line.split(": ", 1)[0] for line in problems] == ["ik", "fk", "fk", "new"]
    assert "not measured" in problems[0] and "no threshold" in problems[3]
    assert "recompiles None" in problems[1] and "ops 11" in problems[2] and "threshold 10" in problems[2]
    assert bench.check_thresholds(rows[:1], thresholds, selected={"fk"}) == problems[1:3]  # ik is not selected


def test_profile_counts_counts_top_level_operators_and_host_syncs() -> None:
    x = torch.ones(3)
    assert bench.profile_counts(lambda: x.add(1).sum()) == (2, 0)
    assert bench.profile_counts(lambda: x @ x)[0] == 1  # aten::dot runs inside aten::matmul: one operator
    assert bench.profile_counts(lambda: x.sum().item())[1] >= 1  # item makes the host wait
    assert bench.profile_counts(lambda: x[x > 0])[1] >= 1  # so does mask indexing


def test_a_row_line_sits_under_the_header() -> None:
    line = bench.row_line({"row": "fk/panda/B1", "median_ms": 0.5, "ops": 12})
    assert len(line) == len(bench.HEADER) and line.split() == ["fk/panda/B1", "0.5000", "-", "12", "-", "-", "-"]
    assert bench.wall_time(lambda: None, lambda: None, repeats=3).keys() == {"median_ms", "min_ms", "page_faults"}


def test_importing_the_package_needs_no_resource_module() -> None:
    """Windows has no ``resource`` module, and ``import better_robot.testing`` (``FakeServer``) must not need it."""
    code = "import sys; sys.modules['resource'] = None; import better_robot.testing"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


_PLATFORM_READERS = {
    "wall_time": lambda: bench.wall_time(lambda: None, lambda: None, repeats=3),
    "load_average": bench.load_average,
    "device_name": lambda: bench.device_name(torch.device("cpu")),
    "context": lambda: bench.context(Path.cwd(), torch.device("cpu"), dtype=torch.float64),
}


@pytest.mark.parametrize("name", _PLATFORM_READERS)
def test_the_runner_side_refuses_a_platform_without_proc(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    """Off Linux there is no ``/proc``: each reader raises and names the platform it needs, never returns a number."""
    monkeypatch.setattr(sys, "platform", "darwin")
    with pytest.raises(BetterRobotError, match=rf"bench\.{name} .*Linux"):
        _PLATFORM_READERS[name]()
