"""Caches and plans (decision records 0011 and 0040).

``Model._plans``, ``Model._memo`` and ``KinematicState._memo`` are the only caches: all are ``init=False``, never
copied by ``dataclasses.replace``, and nothing in ``src`` writes through ``object.__setattr__``, reads a tensor's
``_version`` or touches ``torch._C``.
"""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

import torch

from better_robot import load
from better_robot.kinematics import forward_kinematics_state
from better_robot.kinematics._fk import placement_matrices
from better_robot.kinematics.state import KinematicState
from better_robot.model import Model
from tests.support.robots import panda_urdf

SRC = Path(__file__).resolve().parents[2] / "src" / "better_robot"
SETATTR_ALLOWED: set[Path] = set()


def _cache_field(cls: type, name: str) -> dataclasses.Field:
    return next(field for field in dataclasses.fields(cls) if field.name == name)


def test_cache_fields_are_not_constructor_arguments() -> None:
    for cls, name in ((Model, "_plans"), (Model, "_memo"), (KinematicState, "_memo")):
        field = _cache_field(cls, name)
        assert not field.init and not field.repr and not field.compare, f"{cls.__name__}.{name}"


def test_replace_starts_with_empty_caches() -> None:
    model = load(panda_urdf())
    q = model.q_neutral
    state = forward_kinematics_state(model, q)
    assert model._plans, "forward kinematics builds the FK plan"
    assert model._memo, "forward kinematics memoises the placement matrices"
    for copy in (dataclasses.replace(model, name="other"), model.to(), model.with_values(gravity=model.gravity)):
        assert copy._plans == {} and copy._memo == {}

    pose = state.frame_pose
    assert state._memo, "frame_pose is memoised on the state"
    moved = dataclasses.replace(state, q=q + 0.1)
    assert moved._memo == {}
    assert moved.frame_pose is not pose


def test_values_with_a_graph_are_never_memoised() -> None:
    """Placements with a graph are converted per call, so each differentiable call owns its graph (record 0040)."""
    model = load(panda_urdf(), dtype=torch.float64)
    placements = model.joint_placements.clone().requires_grad_()
    model = model.with_values(joint_placements=placements)
    first, second = (placement_matrices(model)[0] for _ in range(2))
    assert first.requires_grad and first is not second and ("joint_matrices", True) not in model._memo
    forward_kinematics_state(model, model.q_neutral).frame_pose.sum().backward()
    assert placements.grad is not None
    with torch.no_grad():
        forward_kinematics_state(model, model.q_neutral)
    assert ("joint_matrices", False) in model._memo and ("joint_matrices", True) not in model._memo
    assert forward_kinematics_state(model, model.q_neutral).frame_pose.requires_grad, (
        "a no-grad memo serves no grad call"
    )


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
