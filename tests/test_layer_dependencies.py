"""Layer-dependency linter — enforces the architectural DAG.

Walks every ``.py`` file under ``src/better_robot``, parses its
top-level imports, and fails if any layer imports from a strictly higher
layer. Imports guarded by ``if TYPE_CHECKING:`` are ignored (they don't
cross layer boundaries at runtime).

See ``docs/design/01_ARCHITECTURE.md §Dependency rule``.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterable

PKG = "better_robot"
SRC = Path(__file__).resolve().parent.parent / "src" / PKG

# Higher number = higher in the stack. A file at rank K may only import
# from ranks <= K (same-rank imports from a different sub-package are also
# disallowed — see _is_violation below).
LAYER_RANK: dict[str, int] = {
    "backends": 0,
    "utils": 1,
    "_typing": 1,
    "lie": 2,
    "spatial": 3,
    "data_model": 4,
    "io": 5,
    "kinematics": 5,
    "dynamics": 5,
    "collision": 5,
    "residuals": 6,
    "costs": 7,
    "optim": 8,
    "tasks": 9,
    "viewer": 10,
}


def _layer_of(module_parts: tuple[str, ...]) -> str | None:
    """Return the top-level sub-package name, or ``None`` for the root."""
    if not module_parts:
        return None
    return module_parts[0]


def _iter_py_files() -> Iterable[Path]:
    yield from sorted(SRC.rglob("*.py"))


def _file_to_module_parts(path: Path) -> tuple[str, ...]:
    rel = path.relative_to(SRC).with_suffix("")
    parts = tuple(rel.parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return parts


def _resolve_import(
    file_parts: tuple[str, ...],
    node: ast.ImportFrom,
) -> tuple[str, ...] | None:
    """Return the absolute module parts for an ``ImportFrom`` node, or ``None``
    if the import is not inside ``better_robot``.
    """
    if node.level == 0:
        # absolute import
        if not node.module or not node.module.startswith(PKG):
            return None
        rest = node.module[len(PKG) :].lstrip(".")
        return tuple(rest.split(".")) if rest else ()
    # relative import
    base = file_parts[: -node.level] if node.level <= len(file_parts) else ()
    if node.module:
        base = base + tuple(node.module.split("."))
    return base


def _resolve_plain_import(module: str) -> tuple[str, ...] | None:
    if not module.startswith(PKG):
        return None
    rest = module[len(PKG) :].lstrip(".")
    return tuple(rest.split(".")) if rest else ()


def _is_typecheck_block(node: ast.stmt) -> bool:
    if not isinstance(node, ast.If):
        return False
    test = node.test
    if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
        return True
    if isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING":
        return True
    return False


def _top_level_imports(tree: ast.Module) -> list[ast.stmt]:
    """Collect top-level Import / ImportFrom nodes, skipping TYPE_CHECKING blocks."""
    out: list[ast.stmt] = []
    for node in tree.body:
        if _is_typecheck_block(node):
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            out.append(node)
    return out


def _is_violation(
    importer_layer: str | None,
    importee_layer: str | None,
) -> bool:
    """Return True iff ``importer_layer`` is forbidden from depending on ``importee_layer``."""
    if importer_layer is None or importee_layer is None:
        return False
    if importer_layer == importee_layer:
        return False  # same sub-package
    if importer_layer not in LAYER_RANK or importee_layer not in LAYER_RANK:
        return False  # unknown sub-package — stay silent
    return LAYER_RANK[importee_layer] > LAYER_RANK[importer_layer]


def test_no_upward_imports() -> None:
    violations: list[str] = []
    for path in _iter_py_files():
        file_parts = _file_to_module_parts(path)
        importer_layer = _layer_of(file_parts)
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in _top_level_imports(tree):
            if isinstance(node, ast.ImportFrom):
                target = _resolve_import(file_parts, node)
            else:
                targets: list[tuple[str, ...] | None] = [
                    _resolve_plain_import(alias.name) for alias in node.names
                ]
                # process every alias separately
                for t in targets:
                    if t is not None:
                        importee_layer = _layer_of(t)
                        if _is_violation(importer_layer, importee_layer):
                            violations.append(
                                f"{path.relative_to(SRC)} imports {'.'.join(t)} "
                                f"({importer_layer} → {importee_layer})"
                            )
                continue
            if target is None:
                continue
            importee_layer = _layer_of(target)
            if _is_violation(importer_layer, importee_layer):
                violations.append(
                    f"{path.relative_to(SRC)} imports {'.'.join(target)} "
                    f"({importer_layer} → {importee_layer})"
                )
    assert not violations, "layer violations:\n  " + "\n  ".join(violations)


def test_no_pypose_imports() -> None:
    """No module under ``src/`` may ``import pypose`` after P10-D.

    The pure-PyTorch backend in ``lie/_torch_native_backend.py`` is now
    the only Lie implementation.
    """
    offenders: list[str] = []
    for path in _iter_py_files():
        text = path.read_text()
        for lineno, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("import pypose") or stripped.startswith("from pypose"):
                offenders.append(f"{path.relative_to(SRC)}:{lineno}: {stripped}")
    assert not offenders, (
        "modules importing pypose after P10-D drop:\n  " + "\n  ".join(offenders)
    )


# ---------------------------------------------------------------------------
# Viewer dependency hygiene (docs/design/12_VIEWER.md §17)
# ---------------------------------------------------------------------------


def _check_forbidden_import(allowed_file_suffix: str, forbidden_pkg: str) -> list[str]:
    """Return offenders: files (other than *allowed_file_suffix*) that contain a
    top-level import of *forbidden_pkg*.
    """
    offenders: list[str] = []
    for path in _iter_py_files():
        if str(path).endswith(allowed_file_suffix):
            continue
        text = path.read_text()
        for lineno, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith(f"import {forbidden_pkg}") or stripped.startswith(
                f"from {forbidden_pkg}"
            ):
                offenders.append(f"{path.relative_to(SRC)}:{lineno}: {stripped}")
    return offenders


def test_only_viser_backend_imports_viser() -> None:
    """Only ``viewer/renderers/viser_backend.py`` may import viser.

    See docs/design/12_VIEWER.md §17.
    """
    offenders = _check_forbidden_import("viser_backend.py", "viser")
    assert not offenders, (
        "non-viser_backend files importing viser:\n  " + "\n  ".join(offenders)
    )


def test_only_offscreen_backend_imports_pyrender() -> None:
    """Only ``viewer/renderers/offscreen_backend.py`` may import pyrender.

    See docs/design/12_VIEWER.md §17.
    """
    offenders = _check_forbidden_import("offscreen_backend.py", "pyrender")
    assert not offenders, (
        "non-offscreen_backend files importing pyrender:\n  " + "\n  ".join(offenders)
    )


def test_only_urdf_mesh_imports_trimesh() -> None:
    """Only ``viewer/render_modes/urdf_mesh.py`` may import trimesh.

    See docs/design/12_VIEWER.md §17.
    """
    offenders = _check_forbidden_import("urdf_mesh.py", "trimesh")
    assert not offenders, (
        "non-urdf_mesh files importing trimesh:\n  " + "\n  ".join(offenders)
    )


def test_only_recorder_imports_imageio() -> None:
    """Only ``viewer/recorder.py`` may import imageio.

    See docs/design/12_VIEWER.md §17.
    """
    offenders: list[str] = []
    for path in _iter_py_files():
        if str(path).endswith("recorder.py"):
            continue
        text = path.read_text()
        for lineno, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("import imageio") or stripped.startswith("from imageio"):
                offenders.append(f"{path.relative_to(SRC)}:{lineno}: {stripped}")
    assert not offenders, (
        "non-recorder files importing imageio:\n  " + "\n  ".join(offenders)
    )
