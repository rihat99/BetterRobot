"""Dead repository citations in tracked text (principle 7.8).

A citation is a relative path with at least one directory and a file extension (``src/better_robot/io/__init__.py``,
``docs/figures/architecture.svg``), optionally followed by ``::symbol``. It resolves from the repository root, except
that a Markdown link target resolves from the page that holds it and a path in ``mkdocs.yml`` from ``docs/``, as the
renderers read them. It is dead when the resolved path is not tracked, or when the file is not Python or does not
define ``symbol`` along its full dotted path (``Class.method``). A line number after a path is always dead: it
drifts with every edit above it.

Not citations: a path inside a URL or an absolute path, one after ``$VAR/`` or ``package@1/``, one that starts with
``./`` or ``../``, one without a directory (``README.md``), and one under another repository's name
(``BetterHuman/tests/...``), which that repository checks. Files under ``tests/contract/ratchets/`` are not read: a
citations ratchet lists dead citations by design.
"""

from __future__ import annotations

import ast
import posixpath
import re
import subprocess
from pathlib import Path

_TEXT_SUFFIXES = (".md", ".py", ".toml", ".yml", ".yaml", ".json")
_NOT_AFTER = r"(?<![\w./@:$-])"
_CITED = re.compile(_NOT_AFTER + r"(\.?[\w-][\w.-]*(?:/[\w.-]+)*/[\w-][\w.-]*\.[A-Za-z]\w*)(?:::([\w.]*\w))?")
_LINE_CITED = re.compile(_NOT_AFTER + r"[\w./-]*\.[A-Za-z]\w*:\d+")
_REPOSITORIES = ("BetterHuman", "BetterRobot")
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
            if path.split("/", 1)[0] in _REPOSITORIES:
                continue
            if text.endswith("](", 0, match.start()):
                path = posixpath.normpath(posixpath.join(posixpath.dirname(page), path))
            elif page == "mkdocs.yml":
                path = f"docs/{path}"
            if path not in tracked or (symbol and not _defines(repo_root / path, symbol)):
                dead.add(f"{page}: {match[0]}")
        dead.update(f"{page}: {match[0]}" for match in _LINE_CITED.finditer(text))
    return sorted(dead)


def _defines(path: Path, symbol: str) -> bool:
    """Whether the Python file at ``path`` defines ``symbol``, each dotted part inside the one before it."""
    if path.suffix != ".py":
        return False
    body = ast.parse(path.read_text(), filename=str(path)).body
    for part in symbol.split("."):
        names = _names(body)
        if part not in names:
            return False
        body = names[part]
    return True


def _names(body: list[ast.stmt]) -> dict[str, list[ast.stmt]]:
    """Names a block defines by ``def``, ``class`` or assignment, each with the block its own names live in.

    Blocks under ``if``, ``try`` and ``with`` count as the enclosing block, so a definition under
    ``if TYPE_CHECKING:`` or in a ``try``/``except ImportError`` pair resolves.
    """
    names: dict[str, list[ast.stmt]] = {}
    for node in body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names[node.name] = node.body
        elif isinstance(node, ast.Assign):
            names |= {target.id: [] for target in node.targets if isinstance(target, ast.Name)}
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names[node.target.id] = []
        elif isinstance(node, ast.TypeAlias):
            names[node.name.id] = []
        elif isinstance(node, ast.If):
            names |= _names(node.body) | _names(node.orelse)
        elif isinstance(node, ast.Try):
            for block in (node.body, *(handler.body for handler in node.handlers), node.orelse, node.finalbody):
                names |= _names(block)
        elif isinstance(node, ast.With):
            names |= _names(node.body)
    return names
