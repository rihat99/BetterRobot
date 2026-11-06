"""Every repository path cited in tracked text resolves, and no line number is cited (principle 7.8).

Citations are found by ``better_robot.testing.dead_citations``: a relative path with a directory and an extension must
be tracked (from the repository root; a Markdown link from its page, a ``mkdocs.yml`` path from ``docs/``), a
``path::symbol`` must name a symbol the Python file defines along its full dotted path, and a ``<path>:<line>``
citation is always dead. The dead citations that exist today sit in ``tests/contract/ratchets/dead_citations.json``
(``"<file>: <citation>"`` to the reason); the ratchet only shrinks, and the fix that resolves a citation removes its
entry. Permanent exceptions: the resolver's and the edge walker's test cases, this module's own exception list, and
citations in accepted ADRs, whose text never changes after acceptance (principle 8.2).
"""

from __future__ import annotations

import json
from pathlib import Path

from better_robot.testing import dead_citations

REPO = Path(__file__).resolve().parents[2]
CITATION_RATCHET: dict[str, str] = json.loads((Path(__file__).parent / "ratchets" / "dead_citations.json").read_text())
EXEMPT_FILES: dict[str, str] = {
    "tests/testing/test_citations.py": "its cases write dead citations into a tmp_path repository on purpose",
    "tests/testing/test_import_edges.py": "its cases write package files into tmp_path, named relative to the package",
    "tests/contract/test_citations.py": "its exception list below repeats the dead citations it exempts",
}
FROZEN_RECORD_CITATIONS: dict[str, str] = {
    "docs/contracts/decisions/0001-torch-only.md: tests/contract/test_import_dag.py": "now test_layer_dependencies.py",
    "docs/contracts/decisions/0003-terms-plain-functions.md: tests/contract/test_import_dag.py": (
        "now test_layer_dependencies.py"
    ),
    "docs/contracts/decisions/0004-viser-viewer.md: tests/contract/test_import_dag.py": "now test_layer_dependencies.py",
    "docs/contracts/decisions/0004-viser-viewer.md: tests/contract/test_optional_deps.py": "now test_optional_imports.py",
    "docs/contracts/decisions/0003-terms-plain-functions.md: tests/contract/test_protocols.py": (
        "trashed with the JointModel protocol (0013); C-EXT-1 is review-enforced"
    ),
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
    """A citation that resolves now leaves the citation ratchet in the commit that fixes it."""
    stale = sorted(CITATION_RATCHET.keys() - _dead())
    assert not stale, "remove these entries from tests/contract/ratchets/dead_citations.json:\n  " + "\n  ".join(stale)
