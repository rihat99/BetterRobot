"""Forbid legacy stringly-typed reference frames in the library code.

The kinematics layer used to accept ``reference="world"`` strings; that
is replaced by the :class:`~better_robot.kinematics.ReferenceFrame` enum
and ``reference=`` kwargs that take an enum. This test scans ``src/`` for
the legacy strings.

Advisory until ``BR_STRICT=1``.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2] / "src" / "better_robot"
PATTERN = re.compile(r'reference\s*=\s*"(world|local|local_world_aligned)"')


def test_no_legacy_reference_strings() -> None:
    strict = os.environ.get("BR_STRICT", "") == "1"
    hits: list[str] = []
    for p in ROOT.rglob("*.py"):
        if "__pycache__" in p.parts:
            continue
        for i, line in enumerate(p.read_text().splitlines(), 1):
            if PATTERN.search(line):
                hits.append(f"{p.name}:{i}: {line.strip()}")
    if hits:
        msg = "legacy reference= strings found (use ReferenceFrame enum):\n  " + "\n  ".join(hits)
        if strict:
            pytest.fail(msg)
        else:
            pytest.skip(f"advisory: {msg}")
