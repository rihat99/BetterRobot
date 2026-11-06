"""C-EXT (record 0025): every extension point is a base class or a plain function, so ``src/better_robot`` defines
no ``register_*`` function and no module-level mutable registry, a module-level dict, list or set that a function
of the module writes into (record 0038 removed the last one, ``io.register_parser``)."""

from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src" / "better_robot"
MUTATORS = {"append", "extend", "insert", "add", "update", "setdefault", "pop", "clear"}


def _module_mutables(tree: ast.Module) -> set[str]:
    """Module-level names bound to a dict, list or set display or constructor."""
    names: set[str] = set()
    for node in tree.body:
        targets = (
            node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        )
        value = node.value if isinstance(node, (ast.Assign, ast.AnnAssign)) else None
        literal = isinstance(value, (ast.Dict, ast.List, ast.Set, ast.DictComp, ast.ListComp, ast.SetComp))
        constructor = (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id in {"dict", "list", "set"}
        )
        if literal or constructor:
            names.update(target.id for target in targets if isinstance(target, ast.Name))
    return names


def _registries(file: Path) -> list[str]:
    """``path::symbol`` of every ``register_*`` function and every write into a module-level mutable."""
    tree = ast.parse(file.read_text())
    mutables = _module_mutables(tree)
    found: list[str] = []

    def visit(node: ast.AST, owner: str) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            owner = node.name
            if node.name.startswith("register_"):
                found.append(f"{file.relative_to(REPO).as_posix()}::{node.name}")
        written = (
            isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name) and t.value.id in mutables
                for t in node.targets
            )
        ) or (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in MUTATORS
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in mutables
        )
        if written and owner != "<module>":
            found.append(f"{file.relative_to(REPO).as_posix()}::{owner}")
        for child in ast.iter_child_nodes(node):
            visit(child, owner)

    visit(tree, "<module>")
    return found


def test_src_defines_no_registry() -> None:
    found = sorted({site for file in SRC.rglob("*.py") for site in _registries(file)})
    assert found == [], f"a registry or register_* function; extend through a base class or a plain function: {found}"
