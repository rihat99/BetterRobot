"""The contracts map figure names, for each rule family, the kinds of check the contracts index gives its rules.

A family's kinds are the union over its rules in the index's enforcement matrix: ``ruff`` for a ``[ruff:CODE]`` cell
or the ruff ratchet test, ``test`` for any test path, ``review`` for ``[review]``; the figure prints them in that
order, joined by `` · ``.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
INDEX = REPO / "docs" / "contracts" / "00-index.md"
FIGURE = REPO / "docs" / "figures" / "contracts_map.svg"
ROW = re.compile(r"^\| (?P<family>C-[A-Z]+)-\d+ \| (?P<cell>.*) \|$", re.MULTILINE)
BOX = re.compile(r">(?P<family>C-[A-Z]+)</text>\n(?:<text[^>]*>[^<]*</text>\n)<text[^>]*>(?P<kinds>[^<]*)</text>")


def _kinds_from_index() -> dict[str, str]:
    found: dict[str, set[str]] = {}
    for row in ROW.finditer(INDEX.read_text()):
        cell, kinds = row["cell"], found.setdefault(row["family"], set())
        if "[ruff:" in cell or "test_ruff_ratchets.py" in cell:
            kinds.add("ruff")
        if "tests/" in cell:
            kinds.add("test")
        if "[review]" in cell:
            kinds.add("review")
    return {family: " · ".join(k for k in ("ruff", "test", "review") if k in kinds) for family, kinds in found.items()}


def test_the_contracts_map_shows_the_checks_of_the_index() -> None:
    """A family whose rules gain or lose a kind of check changes the figure with it."""
    drawn = {box["family"]: box["kinds"] for box in BOX.finditer(FIGURE.read_text())}
    assert drawn == _kinds_from_index()
