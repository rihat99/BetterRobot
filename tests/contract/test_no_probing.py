"""No probing in ``src`` (principles 4.2; records 0036, 0037, 0022): no ``getattr(x, "literal", ...)`` and no
``hasattr`` (a typed adapter reads attributes directly and fails loudly), and only ``src/better_robot/viewer/layer.py`` reaches a
layer's ``_visible`` and ``_handles``. Today's sites sit in ``tests/contract/ratchets/probing.json``
(``path::symbol`` to reason); the ratchet only shrinks, and the fix that removes a site removes its entry."""

from __future__ import annotations

import ast
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src" / "better_robot"
LAYER_FILE = SRC / "viewer" / "layer.py"
LAYER_PRIVATES = {"_visible", "_handles"}
RATCHET = Path(__file__).parent / "ratchets" / "probing.json"


def _probes(file: Path) -> set[str]:
    """``path::symbol`` of every literal-name ``getattr``, every ``hasattr`` and every layer-private reach in ``file``."""
    found: set[str] = set()

    def visit(node: ast.AST, owner: str) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            owner = node.name
        probe = (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and (
                node.func.id == "hasattr"
                or (node.func.id == "getattr" and len(node.args) > 1 and isinstance(node.args[1], ast.Constant))
            )
        )
        reach = isinstance(node, ast.Attribute) and node.attr in LAYER_PRIVATES and file != LAYER_FILE
        if probe or reach:
            found.add(f"{file.relative_to(REPO).as_posix()}::{owner}")
        for child in ast.iter_child_nodes(node):
            visit(child, owner)

    visit(ast.parse(file.read_text()), "<module>")
    return found


def _sites() -> set[str]:
    return {site for file in SRC.rglob("*.py") for site in _probes(file)}


def test_src_probes_nowhere_new() -> None:
    ratchet = json.loads(RATCHET.read_text())
    new = sorted(_sites() - ratchet.keys())
    assert new == [], f"read the attribute directly and let a missing one fail: {new}"


def test_probing_ratchet_has_no_stale_entries() -> None:
    """A site that no longer probes leaves the ratchet in the commit that fixes it."""
    stale = sorted(json.loads(RATCHET.read_text()).keys() - _sites())
    assert stale == [], f"drop these entries with the fix that removed the probe: {stale}"
