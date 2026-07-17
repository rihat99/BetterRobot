"""Keep the roadmap's explicit ``NotImplementedError`` inventory honest."""

from __future__ import annotations

import ast
from pathlib import Path
import re


_REPO_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_ROOT = _REPO_ROOT / "src" / "better_robot"
_ROADMAP = _REPO_ROOT / "docs" / "reference" / "roadmap.md"
_START = "<!-- not-implemented-inventory:start -->"
_END = "<!-- not-implemented-inventory:end -->"
_ENTRY = re.compile(r"- `(?P<path>src/better_robot/.+\.py)`")


def _is_explicit_not_implemented(exc: ast.expr | None) -> bool:
    """Return whether a raise expression directly names NotImplementedError."""
    target = exc.func if isinstance(exc, ast.Call) else exc
    return (
        isinstance(target, ast.Name) and target.id == "NotImplementedError"
    ) or (
        isinstance(target, ast.Attribute) and target.attr == "NotImplementedError"
    )


def _live_inventory() -> list[str]:
    inventory: list[str] = []
    for path in sorted(_SOURCE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if any(
            isinstance(node, ast.Raise)
            and _is_explicit_not_implemented(node.exc)
            for node in ast.walk(tree)
        ):
            inventory.append(path.relative_to(_REPO_ROOT).as_posix())
    return inventory


def _documented_inventory() -> list[str]:
    text = _ROADMAP.read_text(encoding="utf-8")
    assert text.count(_START) == 1, f"expected one {_START!r} marker"
    assert text.count(_END) == 1, f"expected one {_END!r} marker"

    block = text.split(_START, 1)[1].split(_END, 1)[0]
    entries: list[str] = []
    for line in block.splitlines():
        if not line.strip():
            continue
        match = _ENTRY.fullmatch(line.strip())
        assert match is not None, f"malformed roadmap inventory line: {line!r}"
        entries.append(match.group("path"))
    return entries


def test_roadmap_inventory_matches_explicit_not_implemented_raises() -> None:
    """Every direct source raise has exactly one sorted roadmap file entry."""
    live = _live_inventory()
    documented = _documented_inventory()

    assert documented == sorted(set(documented)), (
        "roadmap inventory must be sorted and contain no duplicates"
    )
    assert documented == live, (
        "roadmap NotImplementedError inventory is out of date\n"
        f"missing from roadmap: {sorted(set(live) - set(documented))}\n"
        f"stale roadmap entries: {sorted(set(documented) - set(live))}"
    )
