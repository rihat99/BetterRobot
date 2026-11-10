"""Pyright's errors in ``src`` are a shrink-only ratchet (E4; C-TEST-2): no new error, and the count only falls.

Today's errors sit in ``tests/contract/ratchets/pyright.json`` (``"path::symbol rule"`` to the reason, 102 at most,
the baseline count); the fix that clears an error removes its entry. Pyright takes seconds, so the full tier runs
this and the fast tier does not.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from functools import cache
from pathlib import Path

import pytest

from ._ratchet_sites import REPO, site

pytestmark = pytest.mark.slow

RATCHET = Path(__file__).parent / "ratchets" / "pyright.json"


@cache  # one pyright run serves both tests
def _sites() -> set[str]:
    command = [sys.executable, "-m", "pyright", "--outputjson", f"--pythonpath={sys.executable}", "src"]
    environment = {**os.environ, "PYRIGHT_PYTHON_IGNORE_WARNINGS": "1"}  # no "new version" line in the JSON
    # pyright exits 1 whenever it reports an error; the JSON on stdout is the result either way
    run = subprocess.run(command, cwd=REPO, capture_output=True, text=True, env=environment, check=False)
    report = json.loads(run.stdout)
    return {
        f"{site(Path(error['file']), error['range']['start']['line'] + 1)} {error.get('rule', 'error')}"
        for error in report["generalDiagnostics"]
        if error["severity"] == "error"
    }


def test_src_has_no_new_pyright_error() -> None:
    ratchet = json.loads(RATCHET.read_text())
    assert len(ratchet) <= 102, "the pyright ratchet holds at most the baseline's 102 errors"
    new = sorted(_sites() - ratchet.keys())
    assert new == [], f"`pyright src` reports new errors: {new}"


def test_pyright_ratchet_has_no_stale_entries() -> None:
    """An error that is gone leaves the ratchet in the commit that fixes it."""
    stale = sorted(json.loads(RATCHET.read_text()).keys() - _sites())
    assert stale == [], f"drop these entries from tests/contract/ratchets/pyright.json: {stale}"
