"""No probing in ``src`` (principles 4.2; records 0036, 0037, 0022): no ``getattr(x, "literal", ...)`` and no
``hasattr``; a typed adapter reads attributes directly and fails loudly. Another object's ``_`` attribute is ruff's
``SLF001``, which ``tests/contract/test_ruff_ratchets.py`` checks."""

from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src" / "better_robot"


def _probes(file: Path) -> set[str]:
    """``path::symbol`` of every literal-name ``getattr`` and every ``hasattr`` in ``file``."""
    found: set[str] = set()

    def visit(node: ast.AST, owner: str) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            owner = node.name
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and (
                node.func.id == "hasattr"
                or (node.func.id == "getattr" and len(node.args) > 1 and isinstance(node.args[1], ast.Constant))
            )
        ):
            found.add(f"{file.relative_to(REPO).as_posix()}::{owner}")
        for child in ast.iter_child_nodes(node):
            visit(child, owner)

    visit(ast.parse(file.read_text()), "<module>")
    return found


def test_src_probes_nowhere() -> None:
    sites = sorted(site for file in SRC.rglob("*.py") for site in _probes(file))
    assert sites == [], f"read the attribute directly and let a missing one fail: {sites}"
