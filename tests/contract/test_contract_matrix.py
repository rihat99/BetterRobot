"""C-DOC-3: the 00-index enforcement matrix matches what every contract page says.

For each `## C-<AREA>-<n>` heading on a page, the paragraph starting with
`Enforcement:` names its enforcers: test paths (`tests/...py`, optionally
`::test_name`), ruff codes (`[ruff:CODE,...]`) or `[review]`. The matrix row for
that rule must carry exactly the same set. Every cited test path must exist
and every ruff code must be selected in `pyproject.toml`, unless the citation
is followed by a `[planned: L-<n>]` tag (a ledger entry).
"""

from __future__ import annotations

import re
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONTRACTS = ROOT / "docs" / "contracts"
RULE_RE = re.compile(r"^## (C-[A-Z]+-\d+)\b", re.MULTILINE)
RANGE_RE = re.compile(r"(C-[A-Z]+)-(\d+)\.\.(\d+)")
TEST_RE = re.compile(r"`(tests/[\w/]+\.py)(?:::(\w+))?`(\s*\[planned: L-\d+\])?")
RUFF_RE = re.compile(r"`\[ruff:([A-Z0-9,]+)\]`(\s*\[planned: L-\d+\])?")
REVIEW_RE = re.compile(r"`\[review\]`")


def _enforcers(text: str) -> set[str]:
    out: set[str] = set()
    for path, test, planned in TEST_RE.findall(text):
        out.add(f"{path}::{test}" if test else path)
    for codes, planned in RUFF_RE.findall(text):
        out.update(f"ruff:{c}" for c in codes.split(","))
    if REVIEW_RE.search(text):
        out.add("review")
    return out


def _page_rules() -> dict[str, tuple[str, set[str], str]]:
    """Return rule id -> (page name, enforcer set, enforcement paragraph)."""
    rules: dict[str, tuple[str, set[str], str]] = {}
    for page in sorted(CONTRACTS.glob("[0-9][0-9]-*.md")):
        if page.name.startswith("00-"):
            continue
        text = page.read_text()
        heads = list(RULE_RE.finditer(text))
        for i, head in enumerate(heads):
            end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
            body = text[head.end() : end]
            para = next((p for p in body.split("\n\n") if p.startswith("Enforcement:")), "")
            assert para, f"{page.name}: {head.group(1)} has no 'Enforcement:' paragraph"
            rules[head.group(1)] = (page.name, _enforcers(para), para)
    return rules


def _matrix_rows() -> dict[str, set[str]]:
    text = (CONTRACTS / "00-index.md").read_text()
    section = text.split("## Enforcement matrix", 1)[1]
    rows: dict[str, set[str]] = {}
    for line in section.splitlines():
        if not line.startswith("| C-"):
            continue
        rule_cell, enforcer_cell = (c.strip() for c in line.strip("|").split("|", 1))
        ids: list[str] = []
        for cell in rule_cell.split(","):
            cell = cell.strip()
            m = RANGE_RE.fullmatch(cell)
            if m:
                ids += [f"{m[1]}-{k}" for k in range(int(m[2]), int(m[3]) + 1)]
            else:
                ids.append(cell)
        for rule in ids:
            rows[rule] = _enforcers(enforcer_cell)
    return rows


def test_every_rule_in_matrix_with_same_enforcers() -> None:
    rules = _page_rules()
    rows = _matrix_rows()
    assert set(rules) == set(rows), {
        "on pages only": sorted(set(rules) - set(rows)),
        "in matrix only": sorted(set(rows) - set(rules)),
    }
    mismatch = {r: (sorted(rules[r][1]), sorted(rows[r])) for r in rules if rules[r][1] != rows[r]}
    assert not mismatch, mismatch


def test_cited_tests_exist_and_ruff_codes_selected() -> None:
    cfg = tomllib.loads((ROOT / "pyproject.toml").read_text())
    selected = cfg["tool"]["ruff"]["lint"]["select"]
    missing: list[str] = []
    for rule, (page, _, para) in _page_rules().items():
        for path, test, planned in TEST_RE.findall(para):
            if planned:
                continue
            target = ROOT / path
            if not target.exists():
                missing.append(f"{page} {rule}: {path} does not exist")
            elif test and f"def {test}(" not in target.read_text():
                missing.append(f"{page} {rule}: {path}::{test} not found")
        for codes, planned in RUFF_RE.findall(para):
            if planned:
                continue
            for code in codes.split(","):
                if not any(code.startswith(prefix) for prefix in selected):
                    missing.append(f"{page} {rule}: ruff code {code} not selected")
    assert not missing, "\n".join(missing)


def test_matrix_rows_cite_real_paths() -> None:
    """Row-level check: paths in the matrix exist, ignoring directory-only cites."""
    text = (CONTRACTS / "00-index.md").read_text().split("## Enforcement matrix", 1)[1]
    bad = [p for p, _, planned in TEST_RE.findall(text) if not planned and not (ROOT / p).exists()]
    assert not bad, bad


if __name__ == "__main__":
    sys.exit(subprocess.call(["pytest", __file__, "-q"]))
