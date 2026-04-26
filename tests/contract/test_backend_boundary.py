"""Backend boundary contract.

1. Only modules under ``lie/``, ``kinematics/``, ``dynamics/``, and
   ``backends/<name>/`` may import a backend implementation module.
2. Library code must not call ``set_backend(`` internally — the active
   backend is the user's choice.

See ``docs/design/10_BATCHING_AND_BACKENDS.md §7`` and
``docs/UPDATE_PHASES.md §P1``.
"""

from __future__ import annotations

import ast
from pathlib import Path

PKG = "better_robot"
SRC = Path(__file__).resolve().parent.parent.parent / "src" / PKG

# Modules that ARE backend implementations — lazy importers of these are
# the only allowed callers from outside their layer.
BACKEND_IMPL_MODULES: tuple[str, ...] = (
    "better_robot.lie._torch_native_backend",
)

# Sub-packages allowed to top-level-import a backend implementation module.
ALLOWED_LAYERS: tuple[str, ...] = ("lie", "kinematics", "dynamics", "backends")


def _iter_py_files() -> list[Path]:
    return sorted(SRC.rglob("*.py"))


def _file_layer(path: Path) -> str | None:
    rel = path.relative_to(SRC)
    parts = rel.parts
    if parts and parts[-1] == "__init__.py":
        parts = parts[:-1]
    return parts[0] if parts else None


def _resolve_relative(file_parts: tuple[str, ...], node: ast.ImportFrom) -> str | None:
    if node.level == 0:
        return node.module
    base = file_parts[: -node.level] if node.level <= len(file_parts) else ()
    if node.module:
        base = base + tuple(node.module.split("."))
    return ".".join((PKG, *base)) if base else PKG


def _file_to_module_parts(path: Path) -> tuple[str, ...]:
    rel = path.relative_to(SRC).with_suffix("")
    parts = tuple(rel.parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return parts


def test_backend_impl_imports_are_localised() -> None:
    """Top-level imports of a backend implementation module are only
    permitted from within ``lie/``, ``kinematics/``, ``dynamics/``, or
    ``backends/``.
    """
    offenders: list[str] = []
    for path in _iter_py_files():
        layer = _file_layer(path)
        if layer in ALLOWED_LAYERS:
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        file_parts = _file_to_module_parts(path)
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            full = _resolve_relative(file_parts, node)
            if full is None:
                continue
            for impl in BACKEND_IMPL_MODULES:
                if full == impl or full.startswith(impl + "."):
                    offenders.append(
                        f"{path.relative_to(SRC)}: imports {full}"
                    )
    assert not offenders, (
        "library files outside lie/kinematics/dynamics/backends are "
        "importing a backend implementation module:\n  " + "\n  ".join(offenders)
    )


def test_library_does_not_call_set_backend() -> None:
    """``set_backend(...)`` calls must not appear in any module under ``src/``.

    The active backend is a user-side choice; library code routes through
    ``default_backend()`` or accepts ``backend=`` kwargs. Mentions inside
    docstrings or comments are fine — this checks AST call nodes only.
    """
    offenders: list[str] = []
    for path in _iter_py_files():
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Name) and func.id == "set_backend":
                offenders.append(
                    f"{path.relative_to(SRC)}:{func.lineno}: set_backend(...)"
                )
            elif isinstance(func, ast.Attribute) and func.attr == "set_backend":
                offenders.append(
                    f"{path.relative_to(SRC)}:{func.lineno}: ....set_backend(...)"
                )
    assert not offenders, (
        "library code calls set_backend internally:\n  " + "\n  ".join(offenders)
    )
