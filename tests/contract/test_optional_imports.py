"""Each unit of ``better_robot`` imports only the outside packages ``pyproject.toml`` declares for it.

``IMPORTERS`` maps every declared package, the core dependencies and then each extra's packages, to the units that
may import it; its keys are exactly the declared packages. ``BOUNDARY`` derives each unit's allow-list of import names
from it (``MODULES`` maps a distribution such as ``pillow`` to its module ``PIL``), except that ``lie``, ``spatial``
and ``optim`` import torch only (C-DEP-1). An extra's packages belong to one unit, so
``import better_robot`` and the other units stay independent of the extras (C-DEP-2), and a package the manifest does
not declare, a test oracle such as ``pinocchio`` or ``scipy`` included, is allowed nowhere. Every import counts,
wherever it sits; ``better_robot.testing.imported_packages`` walks them, as for the layer test. Outside imports that
exist today without being allowed sit in ``tests/contract/ratchets/outside_imports.json``; the ratchet only shrinks.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
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
    "pillow": {"viewer"},  # extra viewer: Viewer.record writes its frames with it
    "mujoco": {"io"},  # extra mjcf
    "trimesh": {"viewer"},  # extra meshes
    "robot_descriptions": set(),  # extra demos: the examples and docs load robots with it
    "pytest": {"testing"},  # extra testing: the no-skip plugin, better_robot.testing.no_skips
}
MODULES = {"pillow": "PIL"}  # the import name of a distribution whose name differs from it
BOUNDARY: dict[str, set[str]] = {
    unit: {"torch"}
    if unit in TORCH_ONLY
    else {MODULES.get(package, package) for package, units in IMPORTERS.items() if unit in units}
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
    """A removed import leaves the outside-import ratchet in the commit that removes it."""
    stale = sorted(OUTSIDE_RATCHET.keys() - _forbidden_imports())
    assert not stale, "remove these entries from tests/contract/ratchets/outside_imports.json:\n  " + "\n  ".join(stale)


def test_the_root_imports_without_the_optional_packages() -> None:
    """``import better_robot`` and ``better_robot.Viewer`` work with every optional package and yourdfpy blocked.

    A fresh interpreter blocks each one (a ``None`` entry in ``sys.modules`` makes its import raise), so an eager
    import anywhere under the root fails here, and none of them is loaded afterwards (principle 1.7, record 0019).
    """
    blocked = ["viser", "mujoco", "trimesh", "PIL", "yourdfpy"]
    script = (
        "import sys\n"
        f"sys.modules.update(dict.fromkeys({blocked!r}))\n"
        "import better_robot\n"
        "assert better_robot.Viewer.__name__ == 'Viewer'\n"
        f"print([name for name in {blocked!r} if sys.modules[name] is not None])\n"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)
    assert (result.returncode, result.stdout.strip()) == (0, "[]"), result.stderr
