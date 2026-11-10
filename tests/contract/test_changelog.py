"""The changelog groups changes by type and every removal reaches the migration page (principle 7.10).

``docs/reference/changelog.md`` opens with ``## Unreleased``; under it and under each ``## vX.Y.Z (YYYY-MM-DD)``
release, every ``###`` heading is one of the six change types, and every Removed or Deprecated entry names its
replacement (``use ... `name```) or says why there is none (``no replacement; <reason>``).

``docs/reference/migration-0.3.0.md`` maps the public names of the 0.2 line to 0.3.0 in tables whose first column
is the old name and whose second is the new one, both written relative to ``better_robot`` (``lie.SE3``). The old
public tree is ``public_api_before_0.3.0.json``: the ``__all__`` tree of the last 0.2 commit (20862fe) in the format
of ``better_robot.testing.public_api``, with each subpackage walked as today's root walks it and the ``lie``
modules that had no ``__all__`` listed by the names they define. A path of it that the current tree does not resolve
(as a path, a dataclass field or an inherited member) is removed. A backticked old name covers a removed path when
the name's dotted parts end it (``SE3`` covers ``better_robot.lie.SE3``); the members of a removed class or module
are covered with it. Every backticked name in the second column exists in the current tree, and a row without one
(``—``) says why in its note; a keyword (``iterations=``) or an expression is not a name.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import re
from pathlib import Path

from better_robot.testing import public_objects

REPO = Path(__file__).resolve().parents[2]
CHANGELOG = REPO / "docs" / "reference" / "changelog.md"
MIGRATION_PAGE = REPO / "docs" / "reference" / "migration-0.3.0.md"
BEFORE = Path(__file__).resolve().parent / "public_api_before_0.3.0.json"

CHANGE_TYPES = ("Added", "Changed", "Deprecated", "Removed", "Fixed", "Security")
RELEASE = re.compile(r"Unreleased|v\d+\.\d+\.\d+ \(\d{4}-\d{2}-\d{2}\)")
MIGRATES = re.compile(r"(?:^|[:;.] )(?:[Uu]se [^`]*`[^`]+`|[Nn]o replacement; \S)")
NAME = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*")
CURRENT = public_objects("better_robot")


def _sections(text: str) -> list[tuple[str, str, list[str]]]:
    """``(release, change type, entries)`` for every ``###`` section; an entry joins its continuation lines."""
    sections: list[tuple[str, str, list[str]]] = []
    release = ""
    for line in text.splitlines():
        if line.startswith("## "):
            release = line[3:]
        elif line.startswith("### "):
            sections.append((release, line[4:], []))
        elif line.startswith("- ") and sections:
            sections[-1][2].append(line[2:])
        elif line.startswith("  ") and sections and sections[-1][2]:
            sections[-1][2][-1] += " " + line.strip()
    return sections


def _rows(text: str) -> list[list[str]]:
    """The cells of every table body row of a Markdown page."""
    rows = [[cell.strip() for cell in line.strip().strip("|").split(" | ")] for line in text.splitlines()]
    return [row for row in rows if len(row) > 1 and not set(row[0]) <= {"-"} and row[:2] != ["0.2", "0.3.0"]]


def _names(cell: str) -> list[tuple[str, ...]]:
    """The dotted parts of each backticked name of a cell; a call keeps its callee, ``better_robot.`` is dropped."""
    names = []
    for span in re.findall(r"`([^`]+)`", cell):
        callee = span.split("(", 1)[0].removeprefix("better_robot.")
        if NAME.fullmatch(callee):
            names.append(tuple(callee.split(".")))
    return names


def _resolves(path: str) -> bool:
    if path in CURRENT:
        return True
    parent, member = path.rsplit(".", 1)
    owner = CURRENT.get(parent)
    return inspect.isclass(owner) and (
        hasattr(owner, member)
        or (dataclasses.is_dataclass(owner) and member in {field.name for field in dataclasses.fields(owner)})
    )


def _current_names() -> set[tuple[str, ...]]:
    paths = set(CURRENT)
    for path, value in CURRENT.items():
        if inspect.isclass(value) and dataclasses.is_dataclass(value):
            paths |= {f"{path}.{field.name}" for field in dataclasses.fields(value)}
    return {tuple(path.split("."))[1:] for path in paths}


def test_unreleased_comes_first_and_every_heading_is_a_change_type() -> None:
    """``## Unreleased`` opens the changelog; ``###`` headings are the six types of Keep a Changelog."""
    text = CHANGELOG.read_text()
    releases = re.findall(r"^## (.*)$", text, re.MULTILINE)
    assert releases[0] == "Unreleased"
    assert [release for release in releases if not RELEASE.fullmatch(release)] == []
    assert [(release, kind) for release, kind, _ in _sections(text) if kind not in CHANGE_TYPES] == []
    assert re.findall(r"^####", text, re.MULTILINE) == []


def test_every_removal_names_its_replacement_or_why_none() -> None:
    """A Removed or Deprecated entry says ``use ... `name``` or ``no replacement; <reason>``."""
    entries = [
        entry
        for _, kind, section in _sections(CHANGELOG.read_text())
        if kind in ("Removed", "Deprecated")
        for entry in section
    ]
    assert entries
    assert [entry for entry in entries if not MIGRATES.search(entry)] == []


def test_every_removed_public_name_is_on_the_migration_page() -> None:
    """Each path of the 0.2 tree that 0.3.0 lacks is covered by an old name of a migration table."""
    old_names = {name for row in _rows(MIGRATION_PAGE.read_text()) for name in _names(row[0])}
    removed = {path for path in json.loads(BEFORE.read_text()) if not _resolves(path)}
    assert removed

    def covered(path: str) -> bool:
        parts = tuple(path.split("."))
        parent = path.rsplit(".", 1)[0]
        return any(parts[-len(name) :] == name for name in old_names) or (parent in removed and covered(parent))

    missing = sorted(path for path in removed if not covered(path))
    assert missing == [], f"add these to {MIGRATION_PAGE.name}:\n  " + "\n  ".join(missing)


def test_every_new_name_on_the_migration_page_exists() -> None:
    """A name in the second column is a path, a field or a member of the current public tree (its dotted tail)."""
    current = _current_names()
    rows = _rows(MIGRATION_PAGE.read_text())
    names = {name for row in rows for name in _names(row[1])}
    assert names
    unknown = sorted(".".join(name) for name in names if not any(path[-len(name) :] == name for path in current))
    assert unknown == []
    assert [row[0] for row in rows if row[1] == "—" and not row[2]] == []
