"""Every repository path cited in tracked text resolves, and no line number is cited (principle 7.8).

Citations are found by ``better_robot.testing.dead_citations``: a ``src/``, ``tests/``, ``docs/``, ``tools/``,
``examples/`` or ``benchmarks/`` path must be tracked, a ``path::symbol`` must name a symbol the file defines, and a
``.py:<line>`` citation is always dead. The dead citations that exist today sit in ``ratchets/citations.json``
(``"<file>: <citation>"`` to the reason); the ratchet only shrinks, and the fix that resolves a citation removes its
entry. Permanent exceptions: the resolver's test cases, this module's own exception list, and citations in accepted
ADRs, whose text never changes after acceptance (principle 8.2).
"""

from __future__ import annotations

import json
from pathlib import Path

from better_robot.testing import dead_citations

REPO = Path(__file__).resolve().parents[2]
CITATION_RATCHET: dict[str, str] = json.loads((Path(__file__).parent / "ratchets" / "citations.json").read_text())
EXEMPT_FILES: dict[str, str] = {
    "tests/testing/test_citations.py": "its cases write dead citations into a tmp_path repository on purpose",
    "tests/contract/test_citations.py": "its exception list below repeats the dead citations it exempts",
}
FROZEN_RECORD_CITATIONS: dict[str, str] = {
    "docs/contracts/decisions/0001-torch-only.md: tests/contract/test_import_dag.py": "now test_layer_dependencies.py",
    "docs/contracts/decisions/0003-terms-plain-functions.md: tests/contract/test_import_dag.py": (
        "now test_layer_dependencies.py"
    ),
    "docs/contracts/decisions/0004-viser-viewer.md: tests/contract/test_import_dag.py": "now test_layer_dependencies.py",
    "docs/contracts/decisions/0004-viser-viewer.md: tests/contract/test_optional_deps.py": "now test_optional_imports.py",
}


def _dead() -> set[str]:
    return {
        entry
        for entry in dead_citations(REPO)
        if entry.split(": ", 1)[0] not in EXEMPT_FILES and entry not in FROZEN_RECORD_CITATIONS
    }


def test_cited_paths_resolve() -> None:
    """A new dead citation fails: cite a tracked path or ``path::symbol``, never a line number."""
    new = sorted(_dead() - CITATION_RATCHET.keys())
    assert not new, "cite a tracked path or path::symbol, never a line number:\n  " + "\n  ".join(new)


def test_citation_ratchet_has_no_stale_entries() -> None:
    """A citation that resolves now leaves ``ratchets/citations.json`` in the commit that fixes it."""
    stale = sorted(CITATION_RATCHET.keys() - _dead())
    assert not stale, "remove these entries from tests/contract/ratchets/citations.json:\n  " + "\n  ".join(stale)
