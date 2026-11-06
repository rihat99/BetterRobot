"""Dead repository citations in tracked text (principle 7.8).

A citation is a repository path such as ``src/better_robot/io/__init__.py``, optionally followed by ``::symbol``. It is
dead when the path is not tracked, or when the file defines no ``def``/``class`` or top-level assignment of the
symbol's last dotted part. A path written under another repository's name (``BetterHuman/tests/...``) is left to
that repository. A line number after a ``.py`` path is always dead: it drifts with every edit above it. Files under
``tests/contract/ratchets/`` are not read: a citations ratchet lists dead citations by design.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

_TEXT_SUFFIXES = (".md", ".py", ".toml", ".yml", ".yaml", ".json")
_CITED = re.compile(
    r"(?<![\w./-])((?:src|tests|docs|tools|examples|benchmarks)/[\w./-]+\.(?:py|md|toml|ya?ml|json))(?:::([\w.]+))?"
)
_LINE_CITED = re.compile(r"\b[\w/.-]+\.py:\d+")
_RATCHETS = "tests/contract/ratchets/"


def dead_citations(repo_root: Path) -> list[str]:
    """Sorted ``"<file>: <citation>"`` entries for every dead citation in the git-tracked text files of ``repo_root``.

    Files under ``tests/contract/ratchets/`` are skipped.
    """
    listing = subprocess.run(["git", "ls-files", "-z"], cwd=repo_root, capture_output=True, text=True, check=True)
    tracked = set(filter(None, listing.stdout.split("\0")))
    dead: set[str] = set()
    for page in tracked:
        if not page.endswith(_TEXT_SUFFIXES) or page.startswith(_RATCHETS):
            continue
        text = (repo_root / page).read_text()
        for match in _CITED.finditer(text):
            path, symbol = match[1], match[2]
            if path not in tracked or (symbol and not _defines((repo_root / path).read_text(), symbol)):
                dead.add(f"{page}: {match[0]}")
        dead.update(f"{page}: {match[0]}" for match in _LINE_CITED.finditer(text))
    return sorted(dead)


def _defines(source: str, symbol: str) -> bool:
    """Whether ``source`` defines the last dotted part of ``symbol`` by ``def``, ``class`` or a top-level name."""
    name = re.escape(symbol.rpartition(".")[2])
    return re.search(rf"^\s*(?:async\s+)?(?:def|class) {name}\b|^{name}\b", source, re.MULTILINE) is not None
