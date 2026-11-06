"""C-EXT-2 (record 0013): every class in ``src`` is abstract or ``@final``, except the entries of the ratchet
``tests/contract/ratchets/abstract_or_final.json`` (classes whose owning WP has not decided yet; it only shrinks),
and every exported joint kind is a frozen dataclass under ``JointModel``."""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import json
import pkgutil
from pathlib import Path

import pytest

import better_robot
from better_robot.model import joints
from better_robot.model.joints import JointModel

RATCHET = Path(__file__).resolve().parent / "ratchets" / "abstract_or_final.json"


def _classes() -> list[tuple[str, type]]:
    """Every class defined under ``better_robot`` by qualified name, exceptions excluded (never final)."""
    found: dict[str, type] = {}
    for _, name, _ in pkgutil.walk_packages(better_robot.__path__, "better_robot."):
        module = importlib.import_module(name)
        for attribute, value in vars(module).items():
            if inspect.isclass(value) and value.__module__ == name and not issubclass(value, BaseException):
                found[f"{name}.{attribute}"] = value
    return sorted(found.items())


def _undecided(cls: type) -> bool:
    return not (inspect.isabstract(cls) or getattr(cls, "__final__", False))


@pytest.mark.parametrize(("qualname", "cls"), _classes(), ids=[qualname for qualname, _ in _classes()])
def test_class_is_abstract_or_final(qualname: str, cls: type) -> None:
    ratchet = json.loads(RATCHET.read_text())
    assert not _undecided(cls) or qualname in ratchet, f"{qualname} is neither abstract nor @final"


def test_ratchet_names_only_undecided_classes() -> None:
    """An entry whose class is gone or decided leaves with the change that decided it."""
    undecided = {qualname for qualname, cls in _classes() if _undecided(cls)}
    stale = sorted(set(json.loads(RATCHET.read_text())) - undecided)
    assert stale == [], f"drop from the ratchet: {stale}"


@pytest.mark.parametrize("name", [name for name in joints.__all__ if name != "JointModel"])
def test_exported_kind_is_a_frozen_joint_model(name: str) -> None:
    cls = getattr(joints, name)
    assert issubclass(cls, JointModel) and not inspect.isabstract(cls), name
    assert dataclasses.is_dataclass(cls) and cls.__dataclass_params__.frozen, name
    for attribute in ("nq", "nv", "manifold", "axis"):
        assert hasattr(cls, attribute) or attribute in {field.name for field in dataclasses.fields(cls)}, attribute
