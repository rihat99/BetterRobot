"""C-EXT-2 (record 0013): every class in ``better_robot.model.joints`` and ``better_robot.model.manifolds`` is
abstract or ``@final``, and every exported joint kind is a frozen dataclass under ``JointModel``."""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import pkgutil

import pytest

from better_robot.model import joints, manifolds
from better_robot.model.joints import JointModel


def _classes() -> list[tuple[str, type]]:
    modules = [manifolds, joints] + [
        importlib.import_module(name) for _, name, _ in pkgutil.walk_packages(joints.__path__, joints.__name__ + ".")
    ]
    found: dict[str, type] = {}
    for module in modules:
        for name, value in vars(module).items():
            if inspect.isclass(value) and value.__module__ == module.__name__:
                found[f"{module.__name__}.{name}"] = value
    return sorted(found.items())


@pytest.mark.parametrize(("qualname", "cls"), _classes(), ids=[qualname for qualname, _ in _classes()])
def test_class_is_abstract_or_final(qualname: str, cls: type) -> None:
    assert inspect.isabstract(cls) or getattr(cls, "__final__", False), f"{qualname} is neither abstract nor @final"


@pytest.mark.parametrize("name", [name for name in joints.__all__ if name != "JointModel"])
def test_exported_kind_is_a_frozen_joint_model(name: str) -> None:
    cls = getattr(joints, name)
    assert issubclass(cls, JointModel) and not inspect.isabstract(cls), name
    assert dataclasses.is_dataclass(cls) and cls.__dataclass_params__.frozen, name
    for attribute in ("nq", "nv", "manifold", "axis"):
        assert hasattr(cls, attribute) or attribute in {field.name for field in dataclasses.fields(cls)}, attribute
