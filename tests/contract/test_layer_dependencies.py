"""Imports between the first-level units of ``better_robot`` follow one edge table.

A unit is a first-level subpackage or module; the root ``__init__`` is the unit ``better_robot`` and re-exports
every other unit. ``ALLOWED`` has one row per other unit, and a unit imports only the units its row lists. Every
import counts, wherever it sits: module level, inside a function, under ``TYPE_CHECKING``. The edges that exist
today without being allowed sit in ``tests/contract/ratchets/edges.json`` (``"from -> to"`` with a ``path::symbol``
witness); the ratchet only shrinks, and the fix that removes an edge removes its entry. Imports from outside the
package are ``tests/contract/test_optional_imports.py``'s job. See ``docs/concepts/architecture.md``.
"""

from __future__ import annotations

import json
from pathlib import Path

from better_robot.testing import imported_packages

ROOT = "better_robot"
SRC = Path(__file__).resolve().parents[2] / "src"
EDGE_RATCHET: dict[str, str] = json.loads((Path(__file__).parent / "ratchets" / "edges.json").read_text())

ALLOWED: dict[str, set[str]] = {
    "_version": set(),
    "exceptions": set(),
    "lie": {"exceptions"},
    "spatial": {"exceptions", "lie"},
    "model": {"exceptions", "lie", "spatial"},
    "kinematics": {"exceptions", "lie", "model"},
    "dynamics": {"exceptions", "lie", "spatial", "model", "kinematics"},
    "collision": {"exceptions", "model", "kinematics"},
    "io": {"exceptions", "lie", "model"},
    "optim": {"exceptions"},
    "terms": {"exceptions", "lie", "model", "kinematics", "collision", "optim"},
    "viewer": {"exceptions", "lie", "model", "kinematics", "collision"},
    "testing": {"exceptions", "model"},
}


def _forbidden_edges() -> set[str]:
    edges = imported_packages(SRC, ROOT)
    return {
        f"{unit} -> {imported}"
        for unit, imports in edges.items()
        if unit != ROOT
        for imported in (imports & edges.keys()) - ALLOWED[unit]
    }


def test_edge_table_has_one_row_per_unit() -> None:
    """A new unit needs its row (a decision record places it); a removed unit loses its row."""
    units = set(imported_packages(SRC, ROOT)) - {ROOT}
    assert units == ALLOWED.keys(), (
        f"units without a row: {sorted(units - ALLOWED.keys())}; rows without a unit: {sorted(ALLOWED.keys() - units)}"
    )


def test_imports_follow_the_edge_table() -> None:
    """Every import between units is in ``ALLOWED`` or, until its fix lands, in the edge ratchet."""
    new = sorted(_forbidden_edges() - EDGE_RATCHET.keys())
    assert not new, (
        "imports outside the edge table of tests/contract/test_layer_dependencies.py: move the code to a unit the "
        "importer may import, or change the table with a decision record:\n  " + "\n  ".join(new)
    )


def test_edge_ratchet_has_no_stale_entries() -> None:
    """An edge that no longer exists leaves the edge ratchet in the commit that removes it."""
    stale = sorted(EDGE_RATCHET.keys() - _forbidden_edges())
    assert not stale, "remove these edges from tests/contract/ratchets/edges.json:\n  " + "\n  ".join(stale)
