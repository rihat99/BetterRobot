"""Which first-level units of a package import which others: the data of an edge-table test.

A unit is a first-level subpackage or module of the package; the package's own ``__init__.py`` is the unit named
after the package. Every import counts, wherever it sits (module level, inside a function, under
``TYPE_CHECKING``), because each one is a dependency a reader has to know about.
"""

from __future__ import annotations

import ast
from pathlib import Path


def imported_packages(src_root: Path, package: str) -> dict[str, set[str]]:
    """Map each unit of ``package`` (found at ``src_root / package``) to the other units it imports.

    Relative imports are resolved, and ``from package import name`` counts as an import of the unit ``name`` when
    such a unit exists, else of the package's root ``__init__``. A unit's imports of itself are dropped; imports
    outside ``package`` are ignored. Every unit is a key, with an empty set when it imports no other unit.
    """
    package_dir = src_root / package
    files = sorted(package_dir.rglob("*.py"))
    units = {_unit_of(path.relative_to(package_dir).parts, package) for path in files}
    edges: dict[str, set[str]] = {unit: set() for unit in units}
    for path in files:
        parts = path.relative_to(package_dir).parts
        importer = _unit_of(parts, package)
        own_package = (package, *parts[:-1])
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
            for target in _targets(node, own_package):
                if target[0] != package:
                    continue
                imported = target[1] if len(target) > 1 and target[1] in units else package
                if imported != importer:
                    edges[importer].add(imported)
    return edges


def _unit_of(parts: tuple[str, ...], package: str) -> str:
    """Unit of a file, given its path parts below the package directory."""
    if parts == ("__init__.py",):
        return package
    return parts[0].removesuffix(".py")


def _targets(node: ast.AST, own_package: tuple[str, ...]) -> list[tuple[str, ...]]:
    """Absolute dotted parts of every module ``node`` imports; ``from a import b`` yields ``a.b``."""
    if isinstance(node, ast.Import):
        return [tuple(alias.name.split(".")) for alias in node.names]
    if not isinstance(node, ast.ImportFrom):
        return []
    if node.level == 0:
        base = tuple(node.module.split("."))
    else:
        base = own_package[: len(own_package) - node.level + 1]
        if node.module:
            base = (*base, *node.module.split("."))
    return [(*base, alias.name) for alias in node.names]
