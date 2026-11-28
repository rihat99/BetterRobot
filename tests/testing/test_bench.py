"""``testing.bench``: the thresholds check names each problem, operators and host syncs are counted, and a row prints
under the header."""

from __future__ import annotations

import json
from pathlib import Path

import torch

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
    assert bench.check_thresholds(rows, thresholds, selected=None) == [
        "ik: has a threshold but was not measured",
        "fk: recompiles None > threshold 0",
        "fk: ops 11 > threshold 10",
        "new: no threshold; measure it on a clean commit and add it in a commit of its own",
    ]
    assert bench.check_thresholds(rows[:1], thresholds, selected={"fk"}) == [
        "fk: recompiles None > threshold 0",
        "fk: ops 11 > threshold 10",
    ]


def test_profile_counts_counts_top_level_operators_and_host_syncs() -> None:
    x = torch.ones(3)
    assert bench.profile_counts(lambda: x.add(1).sum()) == (2, 0)
    assert bench.profile_counts(lambda: x[x > 0].sum().item()) == (4, 2)  # gt, index (nonzero inside), sum, item


def test_a_row_line_sits_under_the_header() -> None:
    line = bench.row_line({"row": "fk/panda/B1", "median_ms": 0.5, "ops": 12})
    assert len(line) == len(bench.HEADER) and line.split() == ["fk/panda/B1", "0.5000", "-", "12", "-", "-", "-"]
    assert bench.wall_time(lambda: None, lambda: None, repeats=3).keys() == {"median_ms", "min_ms", "page_faults"}
