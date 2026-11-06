"""Caches and plans (decision record 0011).

``Model._plans`` and ``KinematicState._memo`` are the only caches: both are ``init=False``, never copied by
``dataclasses.replace``, and nothing in ``src`` writes through ``object.__setattr__``, reads a tensor's
``_version`` or touches ``torch._C``.
"""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

import torch

from better_robot import load
from better_robot.kinematics import forward_kinematics_state
from better_robot.kinematics.state import KinematicState
from better_robot.model import Model
from tests.support.robots import panda_urdf

SRC = Path(__file__).resolve().parents[2] / "src" / "better_robot"
#: ``JointComposite.__post_init__`` derives ``nq``/``nv`` from its sub-joints; R2 replaces it with a factory.
SETATTR_ALLOWED = {SRC / "model" / "joint_models" / "composite.py"}


def _cache_field(cls: type, name: str) -> dataclasses.Field:
    return next(field for field in dataclasses.fields(cls) if field.name == name)


def test_cache_fields_are_not_constructor_arguments() -> None:
    for cls, name in ((Model, "_plans"), (KinematicState, "_memo")):
        field = _cache_field(cls, name)
        assert not field.init and not field.repr and not field.compare, f"{cls.__name__}.{name}"


def test_replace_starts_with_empty_caches() -> None:
    model = load(panda_urdf())
    q = model.q_neutral
    state = forward_kinematics_state(model, q)
    assert model._plans, "forward kinematics builds the FK plan"
    assert dataclasses.replace(model, name="other")._plans == {}
    assert model.to()._plans == {} and model.with_values(gravity=model.gravity)._plans == {}

    pose = state.frame_pose
    assert state._memo, "frame_pose is memoised on the state"
    moved = dataclasses.replace(state, q=q + 0.1)
    assert moved._memo == {}
    assert moved.frame_pose is not pose


def _violations() -> list[str]:
    found: list[str] = []
    for file in SRC.rglob("*.py"):
        for node in ast.walk(ast.parse(file.read_text())):
            if not isinstance(node, ast.Attribute):
                continue
            owner = node.value
            if node.attr == "__setattr__" and isinstance(owner, ast.Name) and owner.id == "object":
                if file not in SETATTR_ALLOWED:
                    found.append(f"{file.relative_to(SRC)}:{node.lineno} object.__setattr__")
            elif node.attr == "_version":
                found.append(f"{file.relative_to(SRC)}:{node.lineno} ._version")
            elif node.attr == "_C" and isinstance(owner, ast.Name) and owner.id == "torch":
                found.append(f"{file.relative_to(SRC)}:{node.lineno} torch._C")
    return found


def test_no_hidden_cache_mechanics_in_src() -> None:
    assert _violations() == []
    assert all(path.exists() for path in SETATTR_ALLOWED), "drop the exception once the file is gone"
    assert isinstance(torch.zeros(1)._version, int)  # the attribute exists; the library just never reads it
