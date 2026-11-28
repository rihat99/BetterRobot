"""``snapshot_differences``, its command line and ``naming_problems`` on a small package written to ``tmp_path``."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from better_robot.testing import naming_problems, public_api, snapshot_differences

PACKAGE = {
    "__init__.py": '__all__ = ["core", "IrModel", "Builder", "Maker", "scale"]\n'
    "from . import core\nfrom .core import IrModel, scale\nBuilder = Maker = IrModel\n",
    "core.py": '__all__ = ["IrModel", "scale"]\n\n\nclass IrModel:\n    pass\n\n\ndef scale(x, factor=2):\n'
    "    return x * factor\n\n\nresize = scale\n",
    "utils.py": "",
}


@pytest.fixture
def toy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """The package ``toy_api`` on ``sys.path``, removed from ``sys.modules`` afterwards."""
    for name, text in PACKAGE.items():
        (tmp_path / "toy_api" / name).parent.mkdir(exist_ok=True)
        (tmp_path / "toy_api" / name).write_text(text)
    monkeypatch.syspath_prepend(str(tmp_path))
    yield "toy_api"
    for name in [name for name in sys.modules if name.split(".")[0] == "toy_api"]:
        del sys.modules[name]


def test_differences_name_each_path_added_removed_and_changed(toy: str, tmp_path: Path) -> None:
    snapshot = {**public_api(toy), "toy_api.gone": "module", "toy_api.scale": "function(x)"}
    del snapshot["toy_api.core"]
    (tmp_path / "public_api.json").write_text(json.dumps(snapshot))
    assert snapshot_differences(toy, tmp_path / "public_api.json") == [
        "+ toy_api.core: module",
        "- toy_api.gone: module",
        "~ toy_api.scale: function(x) -> function(x, factor=2)",
    ]


def test_the_command_line_writes_checks_and_diffs(toy: str, tmp_path: Path) -> None:
    def run(*arguments: str) -> subprocess.CompletedProcess[str]:
        command = [sys.executable, "-m", "better_robot.testing.api_snapshot", toy, "public_api.json", *arguments]
        environment = {**os.environ, "PYTHONPATH": os.pathsep.join([str(tmp_path), *sys.path])}
        return subprocess.run(command, cwd=tmp_path, env=environment, capture_output=True, text=True, check=False)

    assert run("--write").returncode == 0
    assert json.loads((tmp_path / "public_api.json").read_text()) == public_api(toy)
    clean = run("--check")
    assert (clean.returncode, clean.stdout) == (0, "")
    (tmp_path / "public_api.json").write_text("{}")
    checked, diffed = run("--check"), run("--diff")
    assert (checked.returncode, diffed.returncode) == (1, 0)
    assert checked.stdout == diffed.stdout and "+ toy_api.scale: function(x, factor=2)\n" in checked.stdout


def test_naming_problems_name_each_break_and_its_rule(toy: str) -> None:
    problems = naming_problems(toy, acronyms=("IR",))
    assert [(line.split(": ", 1)[0], line[line.index("(principle") :]) for line in problems] == [
        ("Builder and IrModel and Maker", "(principle 3.4)"),
        ("toy_api.Builder = IrModel", "(principle 3.4)"),
        ("toy_api.Maker = IrModel", "(principle 3.4)"),
        ("toy_api.core.resize = scale", "(principle 3.4)"),
        ("IrModel", "(principle 3.5)"),
        ("utils.py", "(principle 3.8)"),
    ]
    assert "IR" in problems[4].split(": ", 1)[1]  # the fix spells the acronym
    assert naming_problems(toy, acronyms=(), ignore=("Builder", "Maker"))[0] == problems[1]
