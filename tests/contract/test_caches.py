"""Caches and plans (decision records 0011 and 0040).

``Model._plans``, ``Model._memo`` and ``KinematicState._memo`` are the only caches: all are ``init=False``, never
copied by ``dataclasses.replace``, written by one function each (``Model._plan``, ``Model._value``,
``KinematicState._memoised``) and keyed by grad mode on the value side, and nothing in ``src`` writes through
``object.__setattr__``, reads a tensor's ``_version`` or touches ``torch._C``.
"""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

import torch

from better_robot import load
from better_robot.dynamics import inverse_dynamics
from better_robot.kinematics import KinematicState, forward_kinematics, frame_jacobian
from better_robot.kinematics._fk import placement_matrices
from better_robot.model import Model
from tests.support.robots import panda_urdf

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src" / "better_robot"
SETATTR_ALLOWED: set[Path] = set()
MEMO_WRITERS = {
    "src/better_robot/model/model.py": {"_plan", "_value"},
    "src/better_robot/kinematics/forward_kinematics.py": {"_memoised"},
}


def _cache_field(cls: type, name: str) -> dataclasses.Field:
    return next(field for field in dataclasses.fields(cls) if field.name == name)


def _derived(state: KinematicState, model: Model) -> dict[str, torch.Tensor]:
    """One tensor per memo name of a state; the dynamics memo is read through the gravity torque."""
    zeros = torch.zeros(model.nv, dtype=model.dtype)
    return {
        "joint_poses": state.joint_poses,
        "frame_poses": state.frame_poses,
        "jacobian_columns": state.jacobian_columns,
        "world_geometry": inverse_dynamics(model, state, v=zeros, a=zeros),
    }


def test_cache_fields_are_not_constructor_arguments() -> None:
    for cls, name in ((Model, "_plans"), (Model, "_memo"), (KinematicState, "_memo")):
        field = _cache_field(cls, name)
        assert not field.init and not field.repr and not field.compare, f"{cls.__name__}.{name}"


def test_replace_starts_with_empty_caches() -> None:
    """``dataclasses.replace`` on a model or a state never carries a memo over (the stale ``frame_pose`` of N-4)."""
    model = load(panda_urdf())
    q = model.q_neutral
    state = forward_kinematics(model, q)
    assert model._plans, "forward kinematics builds the FK plan"
    assert model._memo, "forward kinematics memoises the placement matrices"
    for copy in (dataclasses.replace(model, name="other"), model.to(), model.with_values(gravity=model.gravity)):
        assert copy._plans == {} and copy._memo == {}

    before = _derived(state, model)
    assert set(state._memo) == {(name, True) for name in before}
    fresh = forward_kinematics(model, q + 0.1)
    moved = dataclasses.replace(
        state, q=fresh.q, joint_transforms=fresh.joint_transforms, frame_transforms=fresh.frame_transforms
    )
    assert moved._memo == {}
    after = _derived(fresh, model)
    for name, value in _derived(moved, model).items():
        assert value is not before[name], name
        torch.testing.assert_close(value, after[name], msg=name)


def test_state_memo_is_keyed_by_grad_mode() -> None:
    """A derived value made under ``torch.no_grad`` never serves a differentiable call of the same state."""
    model = load(panda_urdf(), dtype=torch.float64)
    q = model.q_neutral.clone().requires_grad_()
    state = forward_kinematics(model, q)
    with torch.no_grad():
        detached = _derived(state, model)
    assert not any(value.requires_grad for value in detached.values())
    assert set(state._memo) == {(name, False) for name in detached}
    live = _derived(state, model)
    assert all(value.requires_grad for value in live.values()), "a no-grad memo served a grad call"
    frame_jacobian(model, state, frame=model.frame_names[-1]).sum().backward()
    assert q.grad is not None
    assert set(state._memo) == {(name, mode) for name in detached for mode in (False, True)}


def test_values_with_a_graph_are_never_memoised() -> None:
    """Placements with a graph are converted per call, so each differentiable call owns its graph (record 0040)."""
    model = load(panda_urdf(), dtype=torch.float64)
    placements = model.joint_placements.clone().requires_grad_()
    model = model.with_values(joint_placements=placements)
    first, second = (placement_matrices(model)[0] for _ in range(2))
    assert first.requires_grad and first is not second and ("joint_matrices", True) not in model._memo
    forward_kinematics(model, model.q_neutral).frame_poses.sum().backward()
    assert placements.grad is not None
    with torch.no_grad():
        forward_kinematics(model, model.q_neutral)
    assert ("joint_matrices", False) in model._memo and ("joint_matrices", True) not in model._memo
    assert forward_kinematics(model, model.q_neutral).frame_poses.requires_grad, "a no-grad memo serves no grad call"


def _memo_writers(tree: ast.AST) -> set[str]:
    """Names of the functions that assign into a ``_memo`` or ``_plans`` subscript."""
    writers: set[str] = set()
    for function in ast.walk(tree):
        if not isinstance(function, ast.FunctionDef):
            continue
        for node in ast.walk(function):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.value, ast.Attribute)
                    and target.value.attr in {"_memo", "_plans"}
                ):
                    writers.add(function.name)
    return writers


def _violations() -> list[str]:
    found: list[str] = []
    for file in SRC.rglob("*.py"):
        tree = ast.parse(file.read_text())
        relative = file.relative_to(REPO).as_posix()
        for writer in sorted(_memo_writers(tree) - MEMO_WRITERS.get(relative, set())):
            found.append(f"{relative} {writer} writes a cache; only {MEMO_WRITERS} may")
        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute):
                continue
            owner = node.value
            if node.attr == "__setattr__" and isinstance(owner, ast.Name) and owner.id == "object":
                if file not in SETATTR_ALLOWED:
                    found.append(f"{relative}:{node.lineno} object.__setattr__")
            elif node.attr == "_version":
                found.append(f"{relative}:{node.lineno} ._version")
            elif node.attr == "_C" and isinstance(owner, ast.Name) and owner.id == "torch":
                found.append(f"{relative}:{node.lineno} torch._C")
    return found


def test_no_hidden_cache_mechanics_in_src() -> None:
    assert _violations() == []
    assert all(path.exists() for path in SETATTR_ALLOWED), "drop the exception once the file is gone"
    assert isinstance(torch.zeros(1)._version, int)  # the attribute exists; the library just never reads it
