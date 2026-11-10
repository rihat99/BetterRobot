"""Which first-level units of a package import which others, and what each imports from outside: edge-table data.

A unit is a first-level subpackage or module of the package; the package's own ``__init__.py`` is the unit named
after the package. Every import counts, wherever it sits (module level, inside a function, under
``TYPE_CHECKING``), because each one is a dependency a reader has to know about; so does ``importlib.import_module``
or ``__import__`` of a literal absolute name.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

# The two libraries keep an edge table per unit, so an import of either names the unit it reaches.
_PER_UNIT = ("better_robot", "better_human")


def imported_packages(src_root: Path, package: str) -> dict[str, set[str]]:
    """Map each unit of ``package`` (found at ``src_root / package``) to the targets it imports.

    A target inside ``package`` is the other unit's name (``"kinematics"``; ``"better_robot"`` for the root
    ``__init__``); a unit's imports of itself are dropped. A target outside is its top-level package (``"torch"``),
    except that a target in ``better_robot`` or ``better_human`` keeps its unit (``"better_robot.io"``, or
    ``"better_robot"`` for a name of the root ``__init__``). Standard-library modules are not targets, so ``import io``
    never reads as the unit ``io``. Relative imports are resolved, and ``from pkg import name`` reaches the unit
    ``name`` when ``pkg`` has such a unit. Every unit is a key, with an empty set when it imports nothing.
    """
    package_dir = src_root / package
    files = sorted(package_dir.rglob("*.py"))
    units_of = {package: {_unit_of(path.relative_to(package_dir).parts, package) for path in files}}
    edges: dict[str, set[str]] = {unit: set() for unit in units_of[package]}
    for path in files:
        parts = path.relative_to(package_dir).parts
        importer = _unit_of(parts, package)
        own_package = (package, *parts[:-1])
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
            for module, name in _imports(node, own_package):
                top = module[0]
                if top in sys.stdlib_module_names or top == "__future__":
                    continue
                if top != package and top not in _PER_UNIT:
                    edges[importer].add(top)
                    continue
                if top not in units_of:
                    units_of[top] = _installed_units(top)
                unit = module[1] if len(module) > 1 else name if name in units_of[top] else None
                if top == package:
                    target = unit or package
                    if target != importer:
                        edges[importer].add(target)
                else:
                    edges[importer].add(f"{top}.{unit}" if unit else top)
    return edges


def _unit_of(parts: tuple[str, ...], package: str) -> str:
    """Unit of a file, given its path parts below the package directory."""
    if parts == ("__init__.py",):
        return package
    return parts[0].removesuffix(".py")


def _imports(node: ast.AST, own_package: tuple[str, ...]) -> list[tuple[tuple[str, ...], str | None]]:
    """``(absolute module parts, imported name)`` of each import in ``node``; the name is ``None`` for ``import``."""
    if isinstance(node, ast.Import):
        return [(tuple(alias.name.split(".")), None) for alias in node.names]
    if isinstance(node, ast.Call) and _is_dynamic_import(node.func) and node.args:
        name = node.args[0]
        if isinstance(name, ast.Constant) and isinstance(name.value, str) and not name.value.startswith("."):
            return [(tuple(name.value.split(".")), None)]
        return []
    if not isinstance(node, ast.ImportFrom):
        return []
    base = own_package[: len(own_package) - node.level + 1] if node.level else ()
    if node.module:
        base = (*base, *node.module.split("."))
    return [(base, alias.name) for alias in node.names]


def _is_dynamic_import(function: ast.expr) -> bool:
    """``importlib.import_module``, ``import_module`` or ``__import__``."""
    if isinstance(function, ast.Attribute):
        return function.attr == "import_module"
    return isinstance(function, ast.Name) and function.id in {"import_module", "__import__"}


def _installed_units(package: str) -> set[str]:
    """First-level subpackages and modules of an installed package, found without importing it."""
    spec = importlib.util.find_spec(package)
    if spec is None or spec.submodule_search_locations is None:
        raise ModuleNotFoundError(f"{package} is imported but not installed as a package; install it to walk its edges")
    return {
        entry.name.removesuffix(".py")
        for location in spec.submodule_search_locations
        for entry in Path(location).iterdir()
        if (entry.suffix == ".py" and entry.name != "__init__.py") or (entry / "__init__.py").is_file()
    }
