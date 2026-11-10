"""Only the decided packages exist, and raw kernels stay private (C-API-5, decision records 0013 and 0019).

The top level of ``src/better_robot`` holds exactly the subpackages the root ``__all__`` lists
(``test_public_surface.py`` pins that list) and private ``_name`` modules: a new public package needs a decision
first. What each package exports is pinned by ``tests/contract/public_api.json``.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import better_robot
from better_robot.testing import public_objects

SRC = Path(__file__).resolve().parents[2] / "src" / "better_robot"


def test_only_the_decided_packages_exist() -> None:
    """Every public top-level module or package on disk is a subpackage the root lists, and the reverse."""
    on_disk = {
        path.stem
        for path in SRC.iterdir()
        if not path.name.startswith("_") and (path.suffix == ".py" or any(path.glob("*.py")))
    }
    listed = {name for name in better_robot.__all__ if inspect.ismodule(getattr(better_robot, name))}
    assert sorted(on_disk) == sorted(listed)


def test_raw_kernels_are_not_public() -> None:
    """Raw tensor passes stay below the state API: no ``*_raw`` name anywhere in the ``__all__`` tree."""
    assert [path for path in public_objects("better_robot") if path.endswith("_raw")] == []
