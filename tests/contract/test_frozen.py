"""C-STATE-1 / C-STATE-2: public dataclasses are frozen.

Every dataclass exported from a public ``__all__`` must be declared
``frozen=True``. The exceptions are listed in ``MUTABLE_DATACLASSES`` with
their reason, and each entry must still name a public mutable dataclass.
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect

import pytest

PACKAGES = [
    "better_robot",
    "better_robot.lie",
    "better_robot.spatial",
    "better_robot.model",
    "better_robot.kinematics",
    "better_robot.dynamics",
    "better_robot.collision",
    "better_robot.io",
    "better_robot.optim",
    "better_robot.terms",
    "better_robot.viewer",
]

MUTABLE_DATACLASSES: dict[str, str] = {
    "better_robot.io.IRBody": "parser IR, filled incrementally while parsing",
    "better_robot.io.IRFrame": "parser IR, filled incrementally while parsing",
    "better_robot.io.IRJoint": "parser IR, filled incrementally while parsing",
    "better_robot.io.IRModel": "parser IR, filled incrementally while parsing",
    "better_robot.optim.State": "solver loop state, mutated by optimizers",
    "better_robot.viewer.MeshSpec": "known violation: a plain record that should be frozen; remove this entry then",
}


def _public_dataclasses() -> list[tuple[str, type]]:
    found: list[tuple[str, type]] = []
    for package in PACKAGES:
        module = importlib.import_module(package)
        for name in getattr(module, "__all__", ()):
            obj = getattr(module, name)
            if inspect.isclass(obj) and dataclasses.is_dataclass(obj):
                found.append((f"{package}.{name}", obj))
    return found


DATACLASSES = _public_dataclasses()
FROZEN = [(qualname, cls) for qualname, cls in DATACLASSES if qualname not in MUTABLE_DATACLASSES]


@pytest.mark.parametrize("qualname,cls", FROZEN, ids=[qualname for qualname, _ in FROZEN])
def test_public_dataclass_is_frozen(qualname: str, cls: type) -> None:
    assert cls.__dataclass_params__.frozen, f"{qualname} must be @dataclass(frozen=True)"


def test_mutable_list_names_public_mutable_dataclasses() -> None:
    """An entry whose class was frozen, renamed or removed leaves ``MUTABLE_DATACLASSES``."""
    found = dict(DATACLASSES)
    stale = [name for name in MUTABLE_DATACLASSES if name not in found or found[name].__dataclass_params__.frozen]
    assert not stale, f"remove from MUTABLE_DATACLASSES: {stale}"


def test_contract_covers_public_dataclasses() -> None:
    assert FROZEN
