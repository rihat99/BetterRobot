"""Ruff's contract rules that still have open sites in ``src`` run here as shrink-only ratchets (R-3).

``N`` (C-NAME-2; ``pyproject.toml`` turns N803 and N806 off in ``lie``, ``spatial``, ``kinematics`` and ``dynamics``,
whose locals use monogram names, C-NAME-1), ``D1`` under the numpy convention (C-DOC-1), ``ANN001`` (C-TEST-2) and
``SLF001`` (C-EXT-3; it also keeps every module off another object's private attributes, the no-probing rule for
``_`` names). Today's sites sit in ``tests/contract/ratchets/ruff.json`` (``"path::symbol CODE"`` to the count of sites
and the reason); the ratchet only shrinks, and the fix that clears a site lowers its count or removes its entry. The rules join ``select`` in
``pyproject.toml`` when their ratchet is empty.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

from ._ratchet_sites import REPO, compare, site

CODES = "N,D1,ANN001,SLF001"
RATCHET = Path(__file__).parent / "ratchets" / "ruff.json"


def _sites() -> Counter[str]:
    command = [sys.executable, "-m", "ruff", "check", "src", f"--select={CODES}", "--output-format=json", "--exit-zero"]
    findings = json.loads(subprocess.run(command, cwd=REPO, capture_output=True, text=True, check=True).stdout)
    return Counter(
        f"{site(Path(finding['filename']), finding['location']['row'])} {finding['code']}" for finding in findings
    )


def test_src_breaks_no_ruff_contract_rule_anew() -> None:
    """A new site fails: a descriptive name (N), a docstring (D1), an annotation (ANN001), no other object's
    private attribute (SLF001)."""
    over, _ = compare(RATCHET, _sites())
    assert over == [], f"`ruff check src --select {CODES}` reports new sites: {over}"


def test_ruff_ratchet_has_no_stale_entries() -> None:
    """A site that passes now leaves the ratchet in the commit that fixes it."""
    _, under = compare(RATCHET, _sites())
    assert under == [], f"lower these counts in tests/contract/ratchets/ruff.json (drop an entry at 0): {under}"
