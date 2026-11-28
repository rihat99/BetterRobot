"""Every repository path cited in tracked text resolves, and no line number is cited (principle 7.8).

Citations are found by ``better_robot.testing.dead_citations``: a relative path with a directory and an extension must
be tracked (from the repository root; a Markdown link from its page, a ``mkdocs.yml`` path from ``docs/``), a
``path::symbol`` must name a symbol the Python file defines along its full dotted path, and a ``<path>:<line>``
citation is always dead. The dead citations that exist today sit in ``tests/contract/ratchets/dead_citations.json``
(``"<file>: <citation>"`` to the reason); the ratchet only shrinks, and the fix that resolves a citation removes its
entry. Permanent exceptions: the resolver's and the edge walker's test cases and this module's own exception list. A
docs page tells no history: no plan code, no "no longer", no "was moved".
Tracked text also names no code of the plan that wrote it (a work package, lane, ruling, phase, review round or audit
finding), which nothing in the repository defines.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from better_robot.testing import dead_citations

REPO = Path(__file__).resolve().parents[2]
CITATION_RATCHET: dict[str, str] = json.loads((Path(__file__).parent / "ratchets" / "dead_citations.json").read_text())
EXEMPT_FILES: dict[str, str] = {
    "tests/testing/test_citations.py": "its cases write dead citations into a tmp_path repository on purpose",
    "tests/testing/test_import_edges.py": "its cases write package files into tmp_path, named relative to the package",
    "tests/testing/test_repository.py": "its cases write repository files into tmp_path, named relative to that root",
    "tests/contract/test_citations.py": "its docstring spells a plan-code path as an example of what it rejects",
}
PLAN_CODE = re.compile(
    r"\bwp/[A-Z]|\blane [A-D]\b|\bR-\d+\b|\(P-\d+\)|\bP\d+-D\d+\b|\bPhase \d+ pass|\bowning WP\b|\bwork package\b"
    r"|\bR\d+'s\b|\bAUDIT_\w+|\bN-\d|\([AC]\d+\)|\bT\d\.\d\b"
)
TEXT_SUFFIXES = (".md", ".py", ".toml", ".yml", ".yaml", ".json")


def _dead() -> set[str]:
    return {entry for entry in dead_citations(REPO) if entry.split(": ", 1)[0] not in EXEMPT_FILES}


def test_cited_paths_resolve() -> None:
    """A new dead citation fails: cite a tracked path or ``path::symbol``, never a line number."""
    new = sorted(_dead() - CITATION_RATCHET.keys())
    assert not new, "cite a tracked path or path::symbol, never a line number:\n  " + "\n  ".join(new)


def test_citation_ratchet_has_no_stale_entries() -> None:
    """A citation that resolves now leaves the citation ratchet in the commit that fixes it."""
    stale = sorted(CITATION_RATCHET.keys() - _dead())
    assert not stale, "remove these entries from tests/contract/ratchets/dead_citations.json:\n  " + "\n  ".join(stale)


HISTORY = re.compile(r"\bP\d+-[A-Z]|\bruling R-\d|(?i:\bno\s+longer\b|\bwas\s+moved\b)")


def test_docs_pages_tell_no_history() -> None:
    """A docs page says what holds for a reader with no history (principle 7.9); what changed, and the plan code
    that changed it, belong in the commit message."""
    found = []
    for page in sorted((REPO / "docs").rglob("*.md")):
        relative = page.relative_to(REPO).as_posix()
        found += [f"{relative}: {' '.join(match[0].split())}" for match in HISTORY.finditer(page.read_text())]
    assert not found, "state what holds now instead:\n  " + "\n  ".join(found)


def test_no_plan_code_in_tracked_text() -> None:
    """Tracked text states a reason, never a plan code such as ``wp/R9b.md``, ``lane D`` or ``ruling R-3``; this module
    is not read, since its pattern spells the codes."""
    listing = subprocess.run(["git", "ls-files", "-z"], cwd=REPO, capture_output=True, text=True, check=True).stdout
    found = [
        f"{path}: {match[0]}"
        for path in listing.split("\0")
        if path.endswith(TEXT_SUFFIXES) and path != "tests/contract/test_citations.py"
        for match in PLAN_CODE.finditer((REPO / path).read_text())
    ]
    assert not found, "write the reason instead of the plan code:\n  " + "\n  ".join(found)
