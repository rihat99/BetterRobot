"""Each unit of ``better_robot`` imports only the outside packages ``pyproject.toml`` declares for it.

``IMPORTERS`` maps every declared package, the core dependencies and then each extra's packages, to the units that
may import it; its keys are exactly the declared packages. ``BOUNDARY`` derives each unit's allow-list from it, except
that ``lie``, ``spatial`` and ``optim`` import torch only (C-DEP-1). An extra's packages belong to one unit, so
``import better_robot`` and the other units stay independent of the extras (C-DEP-2), and a package the manifest does
not declare, a test oracle such as ``pinocchio`` or ``scipy`` included, is allowed nowhere. Every import counts,
wherever it sits; ``better_robot.testing.imported_packages`` walks them, as for the layer test. Outside imports that
exist today without being allowed sit in ``tests/contract/ratchets/outside_imports.json``; the ratchet only shrinks.
"""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

from better_robot.testing import imported_packages

REPO = Path(__file__).resolve().parents[2]
PROJECT = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]
EDGES = imported_packages(REPO / "src", "better_robot")
OUTSIDE_RATCHET: dict[str, str] = json.loads((Path(__file__).parent / "ratchets" / "outside_imports.json").read_text())

TORCH_ONLY = {"lie", "spatial", "optim"}
IMPORTERS: dict[str, set[str]] = {
    "torch": set(EDGES),
    "numpy": {"io", "viewer"},
    "yourdfpy": {"io"},
    "viser": {"viewer"},  # extra viewer
    "mujoco": {"io"},  # extra mjcf
    "trimesh": {"viewer"},  # extra meshes
    "robot_descriptions": set(),  # extra demos: the examples and docs load robots with it
}
BOUNDARY: dict[str, set[str]] = {
    unit: {"torch"} if unit in TORCH_ONLY else {package for package, units in IMPORTERS.items() if unit in units}
    for unit in EDGES
}


def _declared() -> set[str]:
    """Package names of the core dependencies and of every extra in ``pyproject.toml``."""
    requirements = PROJECT["dependencies"] + [r for extra in PROJECT["optional-dependencies"].values() for r in extra]
    return {re.match(r"[\w.-]+", requirement)[0] for requirement in requirements}


def _forbidden_imports() -> set[str]:
    return {
        f"{unit} -> {package}" for unit, imports in EDGES.items() for package in imports - EDGES.keys() - BOUNDARY[unit]
    }


def test_importers_name_exactly_the_declared_packages() -> None:
    """A dependency added to or dropped from ``pyproject.toml`` gets or loses its ``IMPORTERS`` row."""
    assert IMPORTERS.keys() == _declared()


def test_units_import_only_their_declared_packages() -> None:
    """An import of an undeclared package, or of a declared one outside its units, fails."""
    new = sorted(_forbidden_imports() - OUTSIDE_RATCHET.keys())
    assert not new, (
        "imports outside BOUNDARY in tests/contract/test_optional_imports.py: declare the package in pyproject.toml "
        "with its unit (C-DEP-4) or import it where it is allowed:\n  " + "\n  ".join(new)
    )


def test_outside_import_ratchet_has_no_stale_entries() -> None:
    """An import that no longer exists leaves the outside-import ratchet in the commit that removes it."""
    stale = sorted(OUTSIDE_RATCHET.keys() - _forbidden_imports())
    assert not stale, "remove these entries from tests/contract/ratchets/outside_imports.json:\n  " + "\n  ".join(stale)
