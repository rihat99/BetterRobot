"""Ruff's contract rules that still have open sites in ``src`` run here as shrink-only ratchets.

``N`` (C-NAME-2; ``pyproject.toml`` turns N803 and N806 off in ``lie``, ``spatial``, ``kinematics`` and ``dynamics``,
whose locals use monogram names, C-NAME-1), ``D1`` under the numpy convention (C-DOC-1), ``ANN001`` (C-TEST-2) and
``SLF001`` (C-EXT-3; it also keeps every module off another object's private attributes, the no-probing rule for
``_`` names). Today's sites sit in ``tests/contract/ratchets/ruff.json`` (``"path::symbol CODE"`` to the count of sites
and the reason); the ratchet only shrinks, and the fix that clears a site lowers its count or removes its entry. The rules join ``select`` in
``pyproject.toml`` when their ratchet is empty. ``PERMANENT`` holds the sites that are design, not debt (key to its
count and the reason); a site beyond its count is new.
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
#: Record 0041 took the state's world geometry off this side door; the model memo keeps it by design (records 0011,
#: 0040, 0042): a plan or value is derived by the package that uses it, which ``model`` may not import (edge table),
#: and stored through the memo's one writer.
MODEL_MEMO = "kinematics and dynamics fill the model's memo through its one writer, Model._plan / Model._value"
#: Private names shared inside one module or one class: optim's rule is that a private name is used only by the module
#: that defines it (``tests/optim/test_problem.py::test_each_private_name_is_used_only_by_the_module_that_defines_it``).
SAME_MODULE = "ArrowNormal factors and reorders its time band through the BandedNormal it holds, a class of its module"
SAME_CLASS = "a Problem method fills the private registries of the copy it made (copy.copy(self)), a Problem"
PERMANENT: dict[str, tuple[int, str]] = {
    "src/better_robot/dynamics/_world.py::dynamics_plan SLF001": (1, MODEL_MEMO),
    "src/better_robot/dynamics/_world.py::world_kinematics SLF001": (1, MODEL_MEMO),
    "src/better_robot/kinematics/_fk.py::placement_matrices SLF001": (2, MODEL_MEMO),
    "src/better_robot/kinematics/_plans.py::fk_plan SLF001": (1, MODEL_MEMO),
    "src/better_robot/kinematics/_plans.py::jacobian_plan SLF001": (1, MODEL_MEMO),
    "src/better_robot/kinematics/configuration.py::manifold_plan SLF001": (1, MODEL_MEMO),
    "src/better_robot/kinematics/configuration.py::mimic_plan SLF001": (1, MODEL_MEMO),
    "src/better_robot/optim/linear.py::ArrowNormal.masked SLF001": (1, SAME_MODULE),
    "src/better_robot/optim/linear.py::ArrowNormal.predicted_decrease SLF001": (1, SAME_MODULE),
    "src/better_robot/optim/linear.py::ArrowNormal.solve SLF001": (4, SAME_MODULE),
    "src/better_robot/optim/problem.py::Problem._copy SLF001": (3, SAME_CLASS),
    "src/better_robot/optim/problem.py::Problem.with_ SLF001": (4, SAME_CLASS),
}


def _sites() -> Counter[str]:
    command = [sys.executable, "-m", "ruff", "check", "src", f"--select={CODES}", "--output-format=json", "--exit-zero"]
    findings = json.loads(subprocess.run(command, cwd=REPO, capture_output=True, text=True, check=True).stdout)
    return Counter(
        f"{site(Path(finding['filename']), finding['location']['row'])} {finding['code']}" for finding in findings
    )


def _open_sites() -> Counter[str]:
    """Ruff's sites less the ``PERMANENT`` ones; the sites of a permanent key beyond its count stay."""
    return _sites() - Counter({key: count for key, (count, _) in PERMANENT.items()})


def test_src_breaks_no_ruff_contract_rule_anew() -> None:
    """A new site fails: a descriptive name (N), a docstring (D1), an annotation (ANN001), no other object's
    private attribute (SLF001)."""
    over, _ = compare(RATCHET, _open_sites())
    assert over == [], f"`ruff check src --select {CODES}` reports new sites: {over}"


def test_ruff_ratchet_has_no_stale_entries() -> None:
    """A site that passes now leaves the ratchet, or ``PERMANENT``, in the commit that fixes it."""
    found = _sites()
    _, under = compare(RATCHET, _open_sites())
    gone = [f"{key}: {found[key]} (permanent {count})" for key, (count, _) in PERMANENT.items() if found[key] < count]
    assert under == [], f"lower these counts in tests/contract/ratchets/ruff.json (drop an entry at 0): {under}"
    assert gone == [], f"lower these counts in PERMANENT (drop an entry at 0): {gone}"
