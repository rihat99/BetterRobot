"""Pyright's errors in ``src`` are a shrink-only ratchet (E4; C-TEST-2): no new error, and the count only falls.

Today's errors sit in ``tests/contract/ratchets/pyright.json`` (``"path::symbol rule"`` to the count of errors and the
reason; ``CAP`` errors at most, today's total); the fix that clears an error lowers its count or removes its entry,
and lowers ``CAP``. Pyright takes seconds, so the full tier runs
this and the fast tier does not.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import Counter
from functools import cache
from pathlib import Path

import pytest

from ._ratchet_sites import REPO, compare, counts, site

pytestmark = pytest.mark.slow

RATCHET = Path(__file__).parent / "ratchets" / "pyright.json"
CAP = 53


@cache  # one pyright run serves both tests
def _sites() -> Counter[str]:
    command = [sys.executable, "-m", "pyright", "--outputjson", f"--pythonpath={sys.executable}", "src"]
    environment = {**os.environ, "PYRIGHT_PYTHON_IGNORE_WARNINGS": "1"}  # no "new version" line in the JSON
    # pyright exits 1 whenever it reports an error; the JSON on stdout is the result either way
    run = subprocess.run(command, cwd=REPO, capture_output=True, text=True, env=environment, check=False)
    report = json.loads(run.stdout)
    return Counter(
        f"{site(Path(error['file']), error['range']['start']['line'] + 1)} {error.get('rule', 'error')}"
        for error in report["generalDiagnostics"]
        if error["severity"] == "error"
    )


def test_src_has_no_new_pyright_error() -> None:
    assert sum(counts(RATCHET).values()) <= CAP, f"the pyright ratchet holds at most {CAP} errors"
    over, _ = compare(RATCHET, _sites())
    assert over == [], f"`pyright src` reports new errors: {over}"


def test_pyright_ratchet_has_no_stale_entries() -> None:
    """An error that is gone leaves the ratchet in the commit that fixes it."""
    _, under = compare(RATCHET, _sites())
    assert under == [], f"lower these counts in tests/contract/ratchets/pyright.json (drop an entry at 0): {under}"
