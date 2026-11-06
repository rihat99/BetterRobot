"""Each optional or test-only third-party package is imported only in its boundary module.

``yourdfpy`` is a core dependency kept at the URDF boundary; MuJoCo, trimesh, viser and Pillow are extras; pyrender
and imageio are not used (recordings go through Pillow and the ``ffmpeg`` binary); Pinocchio and
``robot_descriptions`` are test-only. Every import counts, wherever it sits, so ``import better_robot`` and every core
package stay independent of the extras (C-DEP-2).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2] / "src" / "better_robot"
BOUNDARY: dict[str, tuple[str, ...]] = {
    "yourdfpy": ("io/__init__.py", "io/parsers/urdf.py"),
    "mujoco": ("io/parsers/mjcf.py",),
    "trimesh": ("viewer/robot.py",),
    "viser": ("viewer/viewer.py",),
    "PIL": ("viewer/capture.py",),
    "pyrender": (),
    "imageio": (),
    "robot_descriptions": (),
    "pinocchio": (),
}


def _imported_packages(file: Path) -> list[tuple[int, str]]:
    """``(lineno, top-level package)`` of every absolute import in ``file``, at any depth."""
    out: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(file.read_text())):
        if isinstance(node, ast.Import):
            out.extend((node.lineno, alias.name.split(".")[0]) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            out.append((node.lineno, node.module.split(".")[0]))
    return out


@pytest.mark.parametrize("file", sorted(ROOT.rglob("*.py")), ids=lambda p: str(p.relative_to(ROOT)))
def test_optional_packages_stay_in_their_boundary_module(file: Path) -> None:
    """An import of a ``BOUNDARY`` package outside its listed modules fails."""
    rel = file.relative_to(ROOT).as_posix()
    violations = [
        f"{rel}:{lineno}: import {package} (allowed only in {BOUNDARY[package] or 'no module'})"
        for lineno, package in _imported_packages(file)
        if package in BOUNDARY and rel not in BOUNDARY[package]
    ]
    assert not violations, "optional imports outside their boundary module:\n  " + "\n  ".join(violations)
