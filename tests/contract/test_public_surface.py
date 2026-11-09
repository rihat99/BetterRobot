"""The public surface is what the ``__all__`` tree lists (principle 1.6; records 0019, 0024 and 0035; C-API-4, C-API-5).

The root ``__all__`` is exactly its twelve subpackages and fourteen first-level names. Walking the tree
(``better_robot.testing.public_objects``: the root, each listed name, the listed modules recursively) every listed
name resolves and no ``__all__`` lists a name twice; the root has no other public attribute; no module is named like a
function its package lists; ``testing`` is listed and documented, and ``FakeServer`` is defined once.
"""

from __future__ import annotations

import inspect
import pkgutil
import subprocess
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


def test_testing_is_listed_and_documented() -> None:
    """Record 0035: the shared test helpers are a documented public module."""
    assert "better_robot.testing" in MODULES
    assert all((inspect.getdoc(PUBLIC[f"better_robot.testing.{name}"]) or "").strip() for name in br.testing.__all__)


def test_fake_server_is_defined_once() -> None:
    """Record 0035: one stand-in viser server; a second copy is the duplicate the record removed."""
    found = subprocess.run(
        ["git", "grep", "-l", "-E", r"^\s*class FakeServer\b", "--", "*.py"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.split()
    assert found == ["src/better_robot/testing/fake_server.py"]
