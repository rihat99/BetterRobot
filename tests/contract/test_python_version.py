"""One Python version, 3.12, is the floor, the development version and ruff's target (decision record 0034)."""

from __future__ import annotations

import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def test_python_version_files_agree() -> None:
    """``.python-version``, ``requires-python`` and ruff's ``target-version`` name the same version."""
    pyproject = tomllib.loads((REPO / "pyproject.toml").read_text())
    version = (REPO / ".python-version").read_text().strip()
    assert version == "3.12"
    assert pyproject["project"]["requires-python"] == f">={version}"
    assert pyproject["tool"]["ruff"]["target-version"] == "py" + version.replace(".", "")
