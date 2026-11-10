"""The codemap's layer section equals the edge table (principle 7.7).

``docs/concepts/architecture.md`` copies ``ALLOWED`` of ``tests/contract/test_layer_dependencies.py`` as one
``unit -> imports`` line per row; this test reads both and fails when they differ.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

REPO = Path(__file__).parents[2]
PAGE = REPO / "docs" / "concepts" / "architecture.md"
EDGE_TEST = REPO / "tests" / "contract" / "test_layer_dependencies.py"


def _edge_table() -> dict[str, set[str]]:
    tree = ast.parse(EDGE_TEST.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.AnnAssign) and ast.unparse(n.target) == "ALLOWED")
    return {
        key.value: {element.value for element in getattr(value, "elts", ())}
        for key, value in zip(node.value.keys, node.value.values)
    }


def _codemap() -> dict[str, set[str]]:
    rows = re.findall(r"^(\w+) +-> (.+)$", PAGE.read_text(), re.MULTILINE)
    return {unit: set() if imports == "nothing" else set(imports.split()) for unit, imports in rows}


def test_codemap_matches_edge_table() -> None:
    """Every row of the edge table is on the page with the same imports, and the page has no other row."""
    assert _codemap() == _edge_table()
