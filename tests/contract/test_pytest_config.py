"""pytest runs strictly (principle 6.10, decision record 0033): every misconfiguration fails and names itself.

A mistyped marker, an unknown config key, an xfail that starts passing, an empty parameter set and a new warning
each fail the run. Each warning filter besides ``error`` names its warning and sits under a comment giving the
reason.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"
CONFIG = tomllib.loads(PYPROJECT.read_text())["tool"]["pytest"]["ini_options"]


def test_strict_options_are_set() -> None:
    """The options of principle 6.10, plus importlib import mode so test files need no package markers."""
    assert {"-ra", "--showlocals", "--strict-markers", "--strict-config", "--import-mode=importlib"} <= set(
        CONFIG["addopts"]
    )
    assert CONFIG["xfail_strict"] is True
    assert CONFIG["empty_parameter_set_mark"] == "fail_at_collect"
    assert CONFIG["testpaths"] == ["tests"]
    assert CONFIG["filterwarnings"][0] == "error"


def test_tier_markers_are_declared() -> None:
    """The markers the tiers select on exist, each with its description."""
    declared = {marker.split(":")[0]: marker for marker in CONFIG["markers"]}
    assert {"cuda", "slow"} <= declared.keys()
    assert all(":" in marker for marker in declared.values())


def test_each_warning_exception_names_its_warning_and_reason() -> None:
    """Every ``ignore`` filter has a message and a category, and the run of filters it belongs to opens with a comment."""
    block = re.search(r"^filterwarnings = \[\n(.*?)^\]", PYPROJECT.read_text(), re.MULTILINE | re.DOTALL)
    lines = [line.strip() for line in block[1].splitlines()]
    problems = []
    for index, line in enumerate(lines):
        if not line.startswith('"ignore'):
            continue
        fields = line.rstrip(",").strip('"').split(":")
        if len(fields) < 3 or not fields[1] or not fields[2]:
            problems.append(f"{line}: name the message and the category")
        previous = index - 1
        while previous >= 0 and lines[previous].startswith('"ignore'):
            previous -= 1
        if previous < 0 or not lines[previous].startswith("#"):
            problems.append(f"{line}: put a comment with the reason above it")
    assert not problems, "pyproject.toml filterwarnings:\n  " + "\n  ".join(problems)
