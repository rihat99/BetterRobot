"""The public surface is what the ``__all__`` tree lists (principle 1.6; records 0019, 0024 and 0035; C-API-4, C-API-5).

The root ``__all__`` is exactly its twelve subpackages and fourteen first-level names. Walking the tree
(``better_robot.testing.public_objects``: the root, each listed name, the listed modules recursively) every listed
name resolves and no ``__all__`` lists a name twice; the root has no other public attribute; no module is named like a
function its package lists; docs and examples take a first-level name from the root; ``testing`` is listed and
documented, and ``FakeServer`` is defined once.
"""

from __future__ import annotations

import ast
import inspect
import pkgutil
import re
import subprocess
import textwrap
from pathlib import Path

import better_robot as br
from better_robot.testing import public_objects

REPO = Path(__file__).resolve().parents[2]
SUBPACKAGES = [
    "exceptions",
    "lie",
    "spatial",
    "model",
    "kinematics",
    "dynamics",
    "collision",
    "io",
    "optim",
    "terms",
    "viewer",
    "testing",
]
FIRST_LEVEL = [
    "load",
    "Model",
    "ModelBuilder",
    "JointFreeFlyer",
    "forward_kinematics",
    "KinematicState",
    "frame_jacobian",
    "joint_jacobian",
    "inverse_dynamics",
    "mass_matrix",
    "forward_dynamics",
    "center_of_mass",
    "solve_ik",
    "Viewer",
]
PUBLIC = public_objects("better_robot")
PYTHON_FENCE = re.compile(r"^(?P<indent>[ \t]*)```[ \t]*python[^\n]*\n(?P<body>.*?)^(?P=indent)```", re.M | re.S)
MODULES = {path: module for path, module in PUBLIC.items() if inspect.ismodule(module)}


def test_root_lists_exactly_the_decided_names() -> None:
    """Record 0019: the twelve subpackages, then the fourteen first-level names, in that order."""
    assert br.__all__ == SUBPACKAGES + FIRST_LEVEL


def test_root_has_no_public_attribute_outside_its_all() -> None:
    """A name the root binds but does not list would be public by accident (principle 1.6)."""
    assert sorted(name for name in vars(br) if not name.startswith("_") and name not in br.__all__) == []


def test_every_all_lists_each_name_once() -> None:
    """The walk resolved every listed name (an unresolved one raises in ``public_objects``); none is listed twice."""
    repeated = {
        path: sorted({n for n in module.__all__ if module.__all__.count(n) > 1}) for path, module in MODULES.items()
    }
    assert {path: names for path, names in repeated.items() if names} == {}


def test_no_module_is_named_like_a_function_its_package_lists() -> None:
    """``kinematics.forward_kinematics`` is a function, never also a module (record 0023 rename rule)."""
    shadowed = [
        f"{path}.{info.name}"
        for path, module in MODULES.items()
        if hasattr(module, "__path__")
        for info in pkgutil.iter_modules(module.__path__)
        if inspect.isfunction(getattr(module, info.name, None))
    ]
    assert shadowed == []


def _tracked(*patterns: str) -> list[str]:
    return subprocess.run(
        ["git", "ls-files", "--", *patterns], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.split()


def _bound_modules(tree: ast.Module) -> dict[str, str]:
    """Each local name an import binds to a ``better_robot`` module, with that module's path."""
    bound = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == "better_robot":
                    bound[alias.asname or "better_robot"] = alias.name if alias.asname else "better_robot"
        elif isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "better_robot":
            for alias in node.names:
                if f"{node.module}.{alias.name}" in MODULES:
                    bound[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    return bound


def _module_path(node: ast.expr, bound: dict[str, str]) -> str | None:
    """The ``better_robot`` module an expression such as ``br.kinematics`` names, or ``None``."""
    if isinstance(node, ast.Name):
        return bound.get(node.id)
    if isinstance(node, ast.Attribute) and (parent := _module_path(node.value, bound)):
        return f"{parent}.{node.attr}" if f"{parent}.{node.attr}" in MODULES else None
    return None


def _first_level_through_a_subpackage(source: str) -> list[str]:
    """``from better_robot.<sub> import <first-level name>`` and ``<a binding of a subpackage>.<first-level name>``."""
    tree = ast.parse(source)
    bound = _bound_modules(tree)
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("better_robot."):
            found += [f"from {node.module} import {a.name}" for a in node.names if a.name in FIRST_LEVEL]
        elif isinstance(node, ast.Attribute) and node.attr in FIRST_LEVEL:
            if _module_path(node.value, bound) not in (None, "better_robot"):
                found.append(ast.unparse(node))
    return found


def test_docs_and_examples_take_first_level_names_from_the_root() -> None:
    """A reader meets ``br.forward_kinematics``, never also ``br.kinematics.forward_kinematics`` (record 0019).

    Reads the syntax tree of every example and of every page's ``python`` fences, joined in order as they run:
    ``ImportFrom`` names, multi-line ones included, and attribute chains over any binding of a subpackage
    (``br.terms``, ``from better_robot import kinematics``, ``import better_robot.dynamics as dyn``). Contract pages
    show library-internal imports and accepted records never change, so ``docs/contracts`` and the changelog are not
    read.
    """
    sources = {path: (REPO / path).read_text() for path in _tracked("examples/*.py")}
    for path in _tracked("README.md", "docs/*.md"):
        if not path.startswith("docs/contracts/") and path != "docs/reference/changelog.md":
            fences = PYTHON_FENCE.finditer((REPO / path).read_text())
            sources[path] = "\n".join(textwrap.dedent(fence["body"]) for fence in fences)
    offenders = [
        f"{path}: {spelling}"
        for path, source in sources.items()
        for spelling in _first_level_through_a_subpackage(source)
    ]
    assert offenders == [], "take these names from better_robot:\n  " + "\n  ".join(offenders)


def test_testing_is_listed_and_documented() -> None:
    """Record 0035: the shared test helpers are a documented public module."""
    assert "better_robot.testing" in MODULES
    assert all((inspect.getdoc(PUBLIC[f"better_robot.testing.{name}"]) or "").strip() for name in br.testing.__all__)


def test_fake_server_is_defined_once() -> None:
    """Record 0035: one stand-in viser server; a second copy is the duplicate the record removed."""
    defining = [
        path for path in _tracked("*.py") if re.search(r"^\s*class FakeServer\b", (REPO / path).read_text(), re.M)
    ]
    assert defining == ["src/better_robot/testing/fake_server.py"]
