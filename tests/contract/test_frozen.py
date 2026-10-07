"""C-STATE-1 / C-STATE-2: public dataclasses are frozen.

Every dataclass exported from a public ``__all__`` must be declared
``frozen=True``; mutable builders and lifecycle objects are listed in
``MUTABLE_ALLOWLIST`` with the reason.
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
    "better_robot.data_model",
    "better_robot.kinematics",
    "better_robot.dynamics",
    "better_robot.collision",
    "better_robot.io",
    "better_robot.optim",
    "better_robot.terms",
    "better_robot.viewer",
]

# Mutable by design (C-API-1 lifecycle objects); reviewed, not accidental.
MUTABLE_ALLOWLIST: dict[str, str] = {
    "better_robot.io.IRBody": "parser IR, filled incrementally while parsing",
    "better_robot.io.IRFrame": "parser IR, filled incrementally while parsing",
    "better_robot.io.IRGeom": "parser IR, filled incrementally while parsing",
    "better_robot.io.IRJoint": "parser IR, filled incrementally while parsing",
    "better_robot.io.IRModel": "parser IR, filled incrementally while parsing",
    "better_robot.optim.State": "solver loop state, mutated by optimizers",
    # Ledger items (p1_violations.md): plain records that should become frozen.
    "better_robot.viewer.MeshSpec": "LEDGER: plain record, make frozen",
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


@pytest.mark.parametrize("qualname,cls", DATACLASSES, ids=[q for q, _ in DATACLASSES])
def test_public_dataclass_is_frozen(qualname: str, cls: type) -> None:
    if qualname in MUTABLE_ALLOWLIST:
        pytest.skip(MUTABLE_ALLOWLIST[qualname])
    assert cls.__dataclass_params__.frozen, f"{qualname} must be @dataclass(frozen=True)"


def test_contract_covers_public_dataclasses() -> None:
    assert DATACLASSES
