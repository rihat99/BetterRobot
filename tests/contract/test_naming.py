"""Contract test: forbid old Pinocchio-style storage names outside the
deprecation shim.

Scans every ``.py`` file under ``src/better_robot/`` (excluding the
backing-store / alias table inside ``data_model/data.py``) for references
to the pre-rename ``Data`` field names. See ``docs/13_NAMING.md`` for the
rename plan and ``docs/02_DATA_MODEL.md §11`` for the shim policy.

If this test fails, grep shows you the offending site; rename the
reference to the new field (e.g. ``data.oMi`` → ``data.joint_pose_world``)
and run ``uv run pytest tests/contract/test_naming.py`` to confirm.
"""

from __future__ import annotations

import pathlib
import re

import pytest


_SRC_ROOT = pathlib.Path(__file__).resolve().parents[2] / "src" / "better_robot"

# ``data.py`` is the shim location — old names MUST live there.
_ALLOWED_FILES = {
    "data_model/data.py",
}

# One regex per old name. Word-boundary left + specific attribute-access pattern
# right (so ``model.gravity`` doesn't match ``data.g``).
_OLD_FIELD_PATTERNS: dict[str, re.Pattern] = {
    "oMi":  re.compile(r"\bdata\.oMi\b|\bstate\.data\.oMi\b"),
    "oMf":  re.compile(r"\bdata\.oMf\b|\bstate\.data\.oMf\b"),
    "liMi": re.compile(r"\bdata\.liMi\b|\bstate\.data\.liMi\b"),
    "ov":   re.compile(r"\bdata\.ov\b"),
    "oa":   re.compile(r"\bdata\.oa\b"),
    "v_joint": re.compile(r"\bdata\.v_joint\b"),
    "a_joint": re.compile(r"\bdata\.a_joint\b"),
    "nle":  re.compile(r"\bdata\.nle\b"),
    "Ag":   re.compile(r"\bdata\.Ag\b"),
    "hg":   re.compile(r"\bdata\.hg\b"),
    "vcom": re.compile(r"\bdata\.vcom\b"),
    "acom": re.compile(r"\bdata\.acom\b"),
    "M":    re.compile(r"\bdata\.M\b"),
    "C":    re.compile(r"\bdata\.C\b"),
    "g":    re.compile(r"\bdata\.g\b"),
    "J":    re.compile(r"\bdata\.J\b"),
    "dJ":   re.compile(r"\bdata\.dJ\b"),
    "com":  re.compile(r"\bdata\.com\b"),
}


def _iter_source_files() -> list[pathlib.Path]:
    return [p for p in _SRC_ROOT.rglob("*.py") if p.is_file()]


@pytest.mark.parametrize("old_name,pattern", list(_OLD_FIELD_PATTERNS.items()))
def test_no_old_names_in_source(old_name: str, pattern: re.Pattern) -> None:
    offenders: list[str] = []
    for path in _iter_source_files():
        relative = path.relative_to(_SRC_ROOT).as_posix()
        if relative in _ALLOWED_FILES:
            continue
        for lineno, line in enumerate(path.read_text().splitlines(), start=1):
            if pattern.search(line):
                offenders.append(f"{relative}:{lineno}: {line.strip()}")
    assert not offenders, (
        f"Found {len(offenders)} reference(s) to deprecated Data.{old_name}:\n"
        + "\n".join(offenders)
        + "\n\nRename to the new name (see docs/13_NAMING.md §3)."
    )


def test_data_module_hosts_the_shim() -> None:
    """Belt-and-braces: the allow-listed file actually contains the alias table."""
    shim_file = _SRC_ROOT / "data_model" / "data.py"
    text = shim_file.read_text()
    assert "_DEPRECATED_ALIASES" in text, "shim constant missing from data.py"
    for old in ("oMi", "oMf", "liMi", "nle", "Ag", "hg"):
        assert f'"{old}"' in text, f"{old} alias entry missing"
