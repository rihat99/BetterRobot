"""The ``path::symbol`` key of a linter finding, shared by the ruff and pyright ratchets (R-3)."""

from __future__ import annotations

import ast
from functools import cache
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


@cache
def _scopes(path: Path) -> tuple[tuple[int, int, str], ...]:
    """``(first line, last line, dotted name)`` of every class and function in ``path``, outer before inner."""
    scopes: list[tuple[int, int, str]] = []

    def visit(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                name = prefix + child.name
                scopes.append((child.lineno, child.end_lineno or child.lineno, name))
                visit(child, name + ".")
            else:
                visit(child, prefix)

    visit(ast.parse(path.read_text()), "")
    return tuple(scopes)


def site(path: Path, line: int) -> str:
    """``<path from the repository root>::<innermost class or function holding line>``, ``<module>`` outside any;
    ``line`` counts from 1."""
    names = [name for first, last, name in _scopes(path.resolve()) if first <= line <= last]
    return f"{path.resolve().relative_to(REPO).as_posix()}::{names[-1] if names else '<module>'}"
