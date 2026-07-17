"""Module-boundary discipline for core and optional integrations.

``yourdfpy`` is a core dependency, but it remains confined to the URDF
boundary.  The heavy integrations (MuJoCo, mesh rendering, Viser, demos,
Pinocchio, and Warp) are extras or test-only dependencies and may be
top-level-imported only by their dedicated boundary modules.  Keeping these
imports out of the rest of ``src/`` makes ``import better_robot`` independent
of optional extras and keeps the layered DAG honest.

This contract is blocking: adding an unauthorized import fails in every test
run, without an environment-variable escape hatch.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2] / "src" / "better_robot"
# Modules that may legally be imported only inside dedicated boundary code.
RESTRICTED_OPTIONAL = {
    "yourdfpy": ("io/parsers/urdf.py",),
    "mujoco": ("io/parsers/mjcf.py",),
    "trimesh": ("io/parsers/", "viewer/", "collision/"),
    "viser": ("viewer/",),
    "robot_descriptions": (),  # test-only
    "pinocchio": (),  # test-only
    "warp": ("kinematics/_warp_bridge.py", "kinematics/_warp_kernels.py"),
}


def _allowed(module: str, path: Path) -> bool:
    """Return True if ``module`` is allowed in ``path``."""
    rel = path.relative_to(ROOT).as_posix()
    for boundary in RESTRICTED_OPTIONAL.get(module, ()):
        if rel == boundary or (boundary.endswith("/") and rel.startswith(boundary)):
            return True
    return False


def _toplevel_imports(file: Path) -> list[tuple[int, str]]:
    """Return ``(lineno, module)`` for top-level imports only."""
    tree = ast.parse(file.read_text())
    out: list[tuple[int, str]] = []
    for node in tree.body:  # only top-level
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.append((node.lineno, alias.name.split(".")[0]))
        elif isinstance(node, ast.ImportFrom) and node.module:
            out.append((node.lineno, node.module.split(".")[0]))
    return out


@pytest.mark.parametrize(
    "file",
    [p for p in ROOT.rglob("*.py") if "__pycache__" not in p.parts],
    ids=lambda p: str(p.relative_to(ROOT)),
)
def test_no_unauthorized_optional_imports(file: Path) -> None:
    """Top-level import of a restricted optional dep outside the allowed dirs fails."""
    violations: list[str] = []
    for lineno, mod in _toplevel_imports(file):
        if mod in RESTRICTED_OPTIONAL and not _allowed(mod, file):
            violations.append(f"{file.name}:{lineno}: top-level `import {mod}`")
    if violations:
        msg = (
            "optional-dep imports outside their dedicated boundary:\n  "
            + "\n  ".join(violations)
        )
        pytest.fail(msg)
