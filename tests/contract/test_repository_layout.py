"""No tracked symlink and no tracked file over 1 MiB (principle 9.3, record 0039).

A tracked symlink over an ignored directory replaces that directory on checkout, and a large file weighs on every
clone forever. A file that truly cannot shrink goes in ``LARGE_ON_PURPOSE`` with its reason.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LIMIT = 1024 * 1024
LARGE_ON_PURPOSE: dict[str, str] = {}


def _tracked() -> list[tuple[str, str]]:
    """``(mode, path)`` of every tracked file."""
    listing = subprocess.run(["git", "ls-files", "-s"], cwd=REPO, capture_output=True, text=True, check=True).stdout
    return [(line.split()[0], line.split("\t", 1)[1]) for line in listing.splitlines()]


def test_no_symlinks_and_size() -> None:
    """Every tracked path is a regular file of at most 1 MiB, or on the reasoned list."""
    tracked = _tracked()
    symlinks = [path for mode, path in tracked if mode == "120000"]
    large = [
        f"{path}: {(REPO / path).stat().st_size} B"
        for mode, path in tracked
        if mode != "120000" and path not in LARGE_ON_PURPOSE and (REPO / path).stat().st_size > LIMIT
    ]
    assert not symlinks, f"tracked symlinks: {symlinks}"
    assert not large, "shrink these under 1 MiB (or list one in LARGE_ON_PURPOSE with its reason):\n  " + "\n  ".join(
        large
    )
