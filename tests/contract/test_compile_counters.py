"""The CPU counters of the hot entry points never grow (decision record 0011, principle 5.8).

``benchmarks/run.py --counters-only`` measures on CPU, for every counter row its docstring lists, the graph breaks,
recompiles, top-level operators and host-syncing operators, and compares them with
``benchmarks/thresholds/torch-<version>.json``. It exits 1 when a counter is above its threshold, a row has no
threshold, or the installed torch has no thresholds file. It runs in a child process with the internal checks off:
they add host syncs and operators, and the switch is read once, at import. Marked ``slow`` (about a minute).

Each hot path contract 09 names has counter rows in every thresholds file, so no hot path goes ungated.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
# Contract 09's hot paths, each with the counter rows (row-id prefixes in benchmarks/run.py) whose calls run it.
HOT_PATH_ROWS = {
    "lie": {"fk_joints"},  # forward kinematics composes every joint transform through `lie`
    "kinematics": {"fk_joints", "jacobian"},
    "dynamics": {"rnea", "crba", "aba"},
    "src/better_robot/optim/solvers.py": {"ik"},  # solve_ik runs the Levenberg-Marquardt step
}


@pytest.mark.slow
def test_cpu_counters_stay_within_their_thresholds(tmp_path: Path) -> None:
    """A grown counter fails: remove the new graph break, recompile, operator or host sync it names."""
    command = [sys.executable, str(REPO / "benchmarks" / "run.py"), "--counters-only", "--device", "cpu"]
    command += ["--out", str(tmp_path / "counters.json")]
    env = {**os.environ, "BETTER_ROBOT_INTERNAL_CHECKS": "0"}
    completed = subprocess.run(command, cwd=REPO, env=env, capture_output=True, text=True, timeout=900, check=False)
    assert completed.returncode == 0, (
        "a CPU counter is above its threshold (record 0011): remove what added it; a change that needs it, or that "
        "lowers one, rewrites benchmarks/thresholds/ on a clean commit in a commit of its own with a Why: line\n"
        + completed.stdout[-4000:]
        + completed.stderr[-2000:]
    )


def test_each_hot_path_of_contract_09_has_counter_rows() -> None:
    """A hot path without a counter row is ungated: add its row to ``benchmarks/run.py`` and to the thresholds."""
    page = (REPO / "docs" / "contracts" / "09-performance.md").read_text()
    sentence = re.search(r"Hot paths are (.+?);", page, re.DOTALL)
    assert sentence is not None, "contract 09 no longer defines its hot paths in one 'Hot paths are ...;' sentence"
    named = set(re.findall(r"`([^`]+)`", sentence.group(1)))
    assert named == set(HOT_PATH_ROWS), f"contract 09 names {sorted(named)}: map each in HOT_PATH_ROWS to its rows"
    thresholds = sorted((REPO / "benchmarks" / "thresholds").glob("torch-*.json"))
    assert thresholds, "benchmarks/thresholds/ holds no thresholds file"
    for path in thresholds:
        rows = {row["row"].split("/")[0] for row in json.loads(path.read_text())["rows"]}
        missing = {hot_path: sorted(names - rows) for hot_path, names in HOT_PATH_ROWS.items() if names - rows}
        assert not missing, f"{path.name}: hot paths without their counter rows: {missing}"
