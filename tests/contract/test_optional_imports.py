"""Module-boundary discipline for format / viewer / reference deps.

The kinematics / dynamics / optim layers must not top-level-import
``yourdfpy``, ``trimesh``, ``viser``, ``robot_descriptions``, or
``pinocchio`` — those imports belong in ``io/parsers/``, ``viewer/``,
``collision/``, or test code. Even though every dep is now installed
by default, keeping format-specific imports at the boundary keeps
``import better_robot`` light and the layered DAG honest.

Currently advisory: lists the imports it would forbid; failure is gated
on ``BR_STRICT=1`` so day-to-day work doesn't redline. This pattern
matches the layer-DAG advisory ladder.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2] / "src" / "better_robot"
# Modules that may legally be imported only inside dedicated boundary code.
RESTRICTED_OPTIONAL = {
    "yourdfpy": ("io/parsers/urdf.py",),
    "trimesh": ("io/parsers/", "viewer/", "collision/"),
    "viser": ("viewer/",),
    "robot_descriptions": (),  # test-only
    "pinocchio": (),  # test-only
}


def _allowed(module: str, path: Path) -> bool:
    """Return True if ``module`` is allowed in ``path``."""
    rel = str(path.relative_to(ROOT))
    for prefix in RESTRICTED_OPTIONAL.get(module, ()):
        if rel.startswith(prefix.rstrip("/")):
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
    strict = os.environ.get("BR_STRICT", "") == "1"
    violations: list[str] = []
    for lineno, mod in _toplevel_imports(file):
        if mod in RESTRICTED_OPTIONAL and not _allowed(mod, file):
            violations.append(f"{file.name}:{lineno}: top-level `import {mod}`")
    if violations:
        msg = (
            "optional-dep imports outside their dedicated boundary:\n  "
            + "\n  ".join(violations)
        )
        if strict:
            pytest.fail(msg)
        else:
            pytest.skip(f"advisory (BR_STRICT=0): {msg}")
