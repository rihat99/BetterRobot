"""Before 1.0 the version is ``0.MINOR.PATCH`` (C-EXT-6), and this major keeps no deprecated alias (C-EXT-4)."""

from __future__ import annotations

import ast
import re
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEPRECATION_WARNINGS = {"DeprecationWarning", "PendingDeprecationWarning", "FutureWarning"}


def test_version_is_pre_1_0() -> None:
    """C-EXT-6: the distribution's version is ``0.MINOR.PATCH``."""
    version = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]["version"]
    assert re.fullmatch(r"0\.\d+\.\d+", version), version


def test_src_emits_no_deprecation_warning() -> None:
    """C-EXT-4: a changed public name is renamed or removed outright, so nothing in src warns of a deprecation."""
    found = [
        f"{path.relative_to(REPO)}: {node.id}"
        for path in sorted((REPO / "src" / "better_robot").rglob("*.py"))
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.Name) and node.id in DEPRECATION_WARNINGS
    ]
    assert not found, found
