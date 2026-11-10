"""Every decision record follows the template, carries an allowed status and an index row, and keeps its text (8.2).

A record is a file ``NNNN-kebab-title.md`` in the decisions folder; the numbers run from 0001 without a gap. Its
status is ``Proposed``, ``Accepted`` (optionally ``; amended by NNNN, ...``) or ``Superseded by NNNN``, and every
number it points at is a later record. Once a record is accepted, its text differs from the text it had at
acceptance (the first commit in which its status was not ``Proposed``) only in its Status line: a change is a new
record. Records 0001 to 0008 predate the template and the frozen-text rule, so they sit in the permanent exception
list below; the resolver test (``tests/contract/test_citations.py``) checks the paths every record cites.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DECISIONS = REPO / "docs" / "contracts" / "decisions"
INDEX = DECISIONS / "index.md"
TEMPLATE = (
    "Context",
    "Decision",
    "Example",
    "Consequences",
    "Alternatives",
    "Compatibility",
    "Enforcement",
    "Open questions",
)
PRE_TEMPLATE: dict[str, str] = {
    number: "recorded on 2026-10-07/08, before the template and the rule that only the Status line changes"
    for number in ("0001", "0002", "0003", "0004", "0005", "0006", "0007", "0008")
}
STATUS = re.compile(
    r"^\*\*Status:\*\* (?P<word>Proposed|Accepted|Superseded by (?P<by>\d{4}))\b(?: \([^)]*\))?"
    r"(?:; amended by (?P<amended>\d{4}(?:, \d{4})*))?(?:\.| ·)"
)


def _records() -> dict[str, Path]:
    return {path.name[:4]: path for path in sorted(DECISIONS.glob("[0-9][0-9][0-9][0-9]-*.md"))}


def _status_line(text: str) -> str:
    return next(line for line in text.splitlines() if line.startswith("**Status:**"))


def test_record_files_are_numbered_without_a_gap_and_titled_by_their_number() -> None:
    """File names are ``NNNN-kebab-title.md`` from 0001 on, and each title line starts with the file's number."""
    records = _records()
    assert list(records) == [f"{number:04d}" for number in range(1, len(records) + 1)]
    for number, path in records.items():
        assert re.fullmatch(r"\d{4}-[a-z0-9]+(?:-[a-z0-9]+)*\.md", path.name), path.name
        assert path.read_text().startswith(f"# {number} "), f"{path.name}: the title line is '# {number} <Title>'"


def test_every_status_is_allowed_and_points_at_later_records() -> None:
    """The status is Proposed, Accepted (optionally amended by later records) or Superseded by a later record."""
    records = _records()
    for number, path in records.items():
        match = STATUS.match(_status_line(path.read_text()))
        assert match, f"{path.name}: status must be Proposed, Accepted[; amended by NNNN] or Superseded by NNNN"
        targets = [match["by"]] if match["by"] else []
        targets += match["amended"].split(", ") if match["amended"] else []
        for target in targets:
            assert target in records and target > number, f"{path.name}: {target} is not a later record"


def test_records_from_0009_have_exactly_the_template_sections() -> None:
    """Records from 0009 on have the eight template sections in order, and an accepted one has no open question."""
    for number, path in _records().items():
        if number in PRE_TEMPLATE:
            continue
        text = path.read_text()
        assert tuple(re.findall(r"^## (.+)$", text, re.M)) == TEMPLATE, f"{path.name}: sections must be {TEMPLATE}"
        open_questions = text.split("\n## Open questions\n", 1)[1].strip()
        if not _status_line(text).startswith("**Status:** Proposed"):
            assert open_questions in ("", "None."), f"{path.name}: an accepted record has no open question"


def test_the_index_lists_every_record_with_its_status() -> None:
    """``index.md`` has one row per record, linking the file and naming the record's status word."""
    index = INDEX.read_text()
    for number, path in _records().items():
        row = re.search(rf"^\| \[{number}\]\({re.escape(path.name)}\) \|.*\| (?P<status>[^|]+) \|$", index, re.M)
        assert row, f"index.md has no row for {path.name}"
        word = STATUS.match(_status_line(path.read_text()))["word"]
        assert row["status"].startswith(word.split(" by ")[0]), f"index.md: {path.name} is {word}"


def _accepted_text(path: Path) -> str:
    """The record's text in the first commit where its status is not Proposed (its text now when uncommitted)."""
    relative = path.relative_to(REPO).as_posix()
    log = subprocess.run(
        ["git", "log", "--reverse", "--format=%H", "--", relative], cwd=REPO, capture_output=True, text=True, check=True
    )
    for commit in log.stdout.split():
        shown = subprocess.run(
            ["git", "show", f"{commit}:{relative}"], cwd=REPO, capture_output=True, text=True, check=False
        )
        if shown.returncode == 0 and not _status_line(shown.stdout).startswith("**Status:** Proposed"):
            return shown.stdout
    return path.read_text()


def test_an_accepted_record_changes_only_in_its_status_line() -> None:
    """Once accepted, a record's text stays as accepted except the Status line: amend or supersede with a new record."""
    changed = []
    for number, path in _records().items():
        text = path.read_text()
        if number in PRE_TEMPLATE or _status_line(text).startswith("**Status:** Proposed"):
            continue
        accepted = _accepted_text(path)
        if text.replace(_status_line(text), "") != accepted.replace(_status_line(accepted), ""):
            changed.append(path.name)
    assert not changed, "write a new record that amends these instead of editing them:\n  " + "\n  ".join(changed)
