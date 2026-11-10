"""C-DOC-3: the 00-index enforcement matrix matches what every contract page says.

For each `## C-<AREA>-<n>` heading on a page, the paragraph starting with
`Enforcement:` names its enforcers: test paths (`tests/...py`, optionally
`::test_name`), ruff codes (`[ruff:CODE,...]`) or `[review]`. The matrix row for
that rule must carry exactly the same set. Every cited test path must exist
and every ruff code must be a rule ruff knows, selected and not ignored in
`pyproject.toml`; no enforcer is
`[planned]`. A rule that only `[review]` checks waits in
`tests/contract/ratchets/review_only.json` (rule id to reason) for a test or
its deletion; the ratchet only shrinks.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tomllib
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONTRACTS = ROOT / "docs" / "contracts"
RULE_RE = re.compile(r"^## (C-[A-Z]+-\d+)\b", re.MULTILINE)
RANGE_RE = re.compile(r"(C-[A-Z]+)-(\d+)\.\.(\d+)")
TEST_RE = re.compile(r"`(tests/[\w/]+\.py)(?:::(\w+))?`")
RUFF_RE = re.compile(r"`\[ruff:([A-Z0-9,]+)\]`")
REVIEW_RE = re.compile(r"`\[review\]`")
REVIEW_ONLY = ROOT / "tests" / "contract" / "ratchets" / "review_only.json"


def _enforcers(text: str) -> set[str]:
    out: set[str] = set()
    for path, test in TEST_RE.findall(text):
        out.add(f"{path}::{test}" if test else path)
    for codes in RUFF_RE.findall(text):
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
        for cell in (part.strip() for part in rule_cell.split(",")):
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


@cache
def _known_ruff_codes() -> frozenset[str]:
    command = [sys.executable, "-m", "ruff", "rule", "--all", "--output-format", "json"]
    rules = json.loads(subprocess.run(command, capture_output=True, text=True, check=True).stdout)
    return frozenset(rule["code"] for rule in rules)


def _ruff_code_problem(code: str) -> str | None:
    """Why ``code`` cannot enforce a rule (no such ruff rule, not selected, or ignored in ``pyproject.toml``), or
    ``None``: a prefix of a selected code (``PLC9999`` under ``PLC``) is no rule."""
    lint = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["ruff"]["lint"]
    if code not in _known_ruff_codes():
        return f"ruff code {code} is no ruff rule"
    if not any(code.startswith(prefix) for prefix in lint["select"]) or code in lint["ignore"]:
        return f"ruff code {code} not selected"
    return None


def test_a_ruff_code_cited_as_an_enforcer_names_a_selected_rule() -> None:
    """The review's plant: ``[ruff:PLC9999]`` passed, matched by the selected prefix ``PLC``."""
    assert _ruff_code_problem("PLC9999") == "ruff code PLC9999 is no ruff rule"
    assert _ruff_code_problem("E501") == "ruff code E501 not selected"  # selected through E, then ignored
    assert _ruff_code_problem("PLC0415") is None


def test_cited_tests_exist_and_ruff_codes_selected() -> None:
    missing: list[str] = []
    for rule, (page, _, para) in _page_rules().items():
        for path, test in TEST_RE.findall(para):
            target = ROOT / path
            if not target.exists():
                missing.append(f"{page} {rule}: {path} does not exist")
            elif test and f"def {test}(" not in target.read_text():
                missing.append(f"{page} {rule}: {path}::{test} not found")
        for codes in RUFF_RE.findall(para):
            for code in codes.split(","):
                if problem := _ruff_code_problem(code):
                    missing.append(f"{page} {rule}: {problem}")
    assert not missing, "\n".join(missing)


def test_matrix_rows_cite_real_paths() -> None:
    """Row-level check: paths in the matrix exist, ignoring directory-only cites."""
    text = (CONTRACTS / "00-index.md").read_text().split("## Enforcement matrix", 1)[1]
    bad = [p for p, _ in TEST_RE.findall(text) if not (ROOT / p).exists()]
    assert not bad, bad


def test_no_enforcer_is_planned() -> None:
    """An enforcer exists today or is not cited: a plan is not a check."""
    planned = [rule for rule, (_, _, para) in _page_rules().items() if "[planned" in para]
    matrix = (CONTRACTS / "00-index.md").read_text().split("## Enforcement matrix", 1)[1]
    planned += [
        line.split("|")[1].strip() for line in matrix.splitlines() if line.startswith("| C-") and "[planned" in line
    ]
    assert not planned, f"cite an existing test or ruff code: {sorted(set(planned))}"


def _review_only() -> set[str]:
    return {rule for rule, (_, enforcers, _) in _page_rules().items() if enforcers == {"review"}}


def test_no_new_review_only_rule() -> None:
    """A rule is a test, a ratchet or deleted: a new rule that only review checks fails."""
    new = sorted(_review_only() - json.loads(REVIEW_ONLY.read_text()).keys())
    assert not new, f"give these rules a test or ruff code: {new}"


def test_review_only_ratchet_has_no_stale_entries() -> None:
    """A rule that gained a test, or was deleted, leaves the ratchet in that commit."""
    stale = sorted(json.loads(REVIEW_ONLY.read_text()).keys() - _review_only())
    assert not stale, f"drop these entries from tests/contract/ratchets/review_only.json: {stale}"


if __name__ == "__main__":
    sys.exit(subprocess.call(["pytest", __file__, "-q"]))
