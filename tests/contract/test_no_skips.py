"""No test skips or xfails: a gate that cannot run fails (principle 6.5, decision record 0033).

A test that needs what the fast and full tiers lack (a GPU) carries the ``cuda`` marker, which deselects it there;
a missing oracle or fixture is an error. The scan is an AST walk of every Python file under ``tests/``, so a skip
spelled in a string or a comment does not count.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parents[1]
BANNED = {"skip", "skipif", "importorskip", "xfail"}


def _skips(file: Path) -> list[str]:
    """``<line>: <spelling>`` of every pytest skip or xfail in ``file``: calls, marks and ``from pytest import``."""
    found: list[str] = []
    for node in ast.walk(ast.parse(file.read_text())):
        if (
            isinstance(node, ast.Attribute)
            and node.attr in BANNED
            and ast.unparse(node.value) in {"pytest", "pytest.mark", "mark"}
        ):
            found.append(f"{node.lineno}: {ast.unparse(node)}")
        elif isinstance(node, ast.ImportFrom) and node.module == "pytest":
            found.extend(
                f"{node.lineno}: from pytest import {alias.name}" for alias in node.names if alias.name in BANNED
            )
    return found


@pytest.mark.parametrize("file", sorted(TESTS.rglob("*.py")), ids=lambda p: str(p.relative_to(TESTS)))
def test_no_skip_or_xfail(file: Path) -> None:
    """A skip or xfail reports green while checking nothing; mark a GPU test ``cuda`` and let anything else fail."""
    found = _skips(file)
    assert not found, f"{file.relative_to(TESTS)} skips or xfails:\n  " + "\n  ".join(found)
