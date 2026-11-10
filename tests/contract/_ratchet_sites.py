"""The ``path::symbol`` key of a linter finding and the comparison of found sites with a counted ratchet (R-3).

A counted ratchet (``tests/contract/ratchets/*.json``) maps a key to ``{"count": n, "reason": ...}``: ``n`` open sites
under that key, so a second site inside a listed symbol fails (principle 6.1). ``tools/ratchet_counts.py`` checks
that no key or count ever rises in the history.
"""

from __future__ import annotations

import ast
import json
from collections.abc import Mapping
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


def counts(ratchet: Path) -> dict[str, int]:
    """Key to count of open sites of the counted ratchet file ``ratchet``."""
    return {key: entry["count"] for key, entry in json.loads(ratchet.read_text()).items()}


def compare(ratchet: Path, found: Mapping[str, int]) -> tuple[list[str], list[str]]:
    """``(over, under)``: keys with more sites than ``ratchet`` allows (new sites to fix) and keys with fewer (lower
    or drop the entry with the fix), each as ``"key: found (ratchet n)"``."""
    allowed = counts(ratchet)
    over = [f"{key}: {found[key]} (ratchet {allowed.get(key, 0)})" for key in found if found[key] > allowed.get(key, 0)]
    under = [
        f"{key}: {found.get(key, 0)} (ratchet {allowed[key]})" for key in allowed if found.get(key, 0) < allowed[key]
    ]
    return sorted(over), sorted(under)
