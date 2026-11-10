"""A ratchet only shrinks over the mainline history (principle 6.1): ``tools/ratchet_counts.py`` passes on this
repository, and planted histories show it fails a new key or a raised count, in a commit or in the working tree.

It reads git history, so the full tier runs it.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow

REPO = Path(__file__).resolve().parents[2]
TOOL = REPO / "tools" / "ratchet_counts.py"


def _run(repo: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, TOOL, "--repo", repo], capture_output=True, text=True, check=False)


def test_no_ratchet_of_this_repository_grew() -> None:
    result = _run(REPO)
    assert result.returncode == 0, result.stdout + result.stderr


def _history(tmp_path: Path, versions: list[dict[str, object]], *, commit_last: bool = True) -> Path:
    """A scratch repository whose one ratchet, ``r.json``, takes each version in turn, one commit each (the last one
    left uncommitted unless ``commit_last``)."""
    ratchet = tmp_path / "tests" / "contract" / "ratchets" / "r.json"
    ratchet.parent.mkdir(parents=True)
    git = ["git", "-C", str(tmp_path), "-c", "user.name=plant", "-c", "user.email=plant@example.invalid"]
    subprocess.run([*git, "init", "-q"], check=True)
    for index, version in enumerate(versions):
        ratchet.write_text(json.dumps(version))
        if commit_last or index < len(versions) - 1:
            subprocess.run([*git, "add", "-A"], check=True)
            subprocess.run([*git, "commit", "-q", "-m", f"version {index}"], check=True)
    return tmp_path


def _entry(count: int) -> dict[str, object]:
    return {"count": count, "reason": "planted"}


@pytest.mark.parametrize(
    ("versions", "commit_last", "fails"),
    [
        ([{"a": _entry(2), "b": _entry(1)}, {"a": _entry(1), "b": _entry(1)}], True, False),
        ([{"a": "planted", "b": "planted"}, {"a": _entry(3)}], True, False),
        ([{"a": _entry(1)}, {"a": _entry(1), "b": _entry(1)}], True, True),
        ([{"a": _entry(1), "b": _entry(2)}, {"a": _entry(2), "b": _entry(1)}], True, True),
        ([{"a": _entry(2)}, {"a": _entry(1)}, {"a": _entry(2)}], True, True),
        ([{"a": "planted"}, {"a": "planted", "b": "planted"}], True, True),
        ([{"a": _entry(1)}, {"a": _entry(2)}], False, True),
    ],
    ids=[
        "counts-fall",
        "first-counted-step-keeps-keys",
        "new-key",
        "count-rises-total-equal",
        "count-rises-again",
        "plain-new-key",
        "working-tree-count-rises",
    ],
)
def test_the_tool_fails_a_ratchet_that_grew(
    tmp_path: Path, versions: list[dict[str, object]], commit_last: bool, fails: bool
) -> None:
    result = _run(_history(tmp_path, versions, commit_last=commit_last))
    assert result.returncode == int(fails), result.stdout + result.stderr


def test_the_tool_prints_sites_from_the_first_step_that_stores_them(tmp_path: Path) -> None:
    """A ratchet created plain (two keys) and counted later (one key, three sites) prints 3 then and 3 now, never
    its key count beside a site count, which would read as growth."""
    result = _run(_history(tmp_path, [{"a": "planted", "b": "planted"}, {"a": _entry(3)}]))
    *_, sites_then, sites_now = result.stdout.splitlines()[1].split()
    assert (sites_then, sites_now) == ("3", "3"), result.stdout
