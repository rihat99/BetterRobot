"""Caches and plans.

``Model._plans``, ``Model._memo`` and ``KinematicState._memo`` are the only caches: all are ``init=False``, never
copied by ``dataclasses.replace``, written by one function each (``Model._plan``, ``Model._value``,
``KinematicState._memoised``) and keyed by grad mode on the value side (the model memo also by the tensor fields'
``requires_grad`` flags; both keep inference-mode values under a key of their own; a compiled call reads the model
memo and writes neither), and nothing in ``src`` writes
through ``object.__setattr__``, reads a tensor's ``_version`` or touches ``torch._C``. A ``functools`` cache lives as
long as the process, so ``FUNCTOOLS_CACHES`` lists, with its reason, the one ``src`` keeps.
"""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

import torch

from better_robot import load
from better_robot.dynamics import inverse_dynamics
from better_robot.kinematics import KinematicState, forward_kinematics, frame_jacobian
from better_robot.kinematics._fk import joint_map
from better_robot.model import Model
from tests.support.robots import panda_urdf

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src" / "better_robot"
SETATTR_ALLOWED: set[Path] = set()
CACHE_WRITERS = {
    "_plans": {("src/better_robot/model/model.py", "_plan")},
    "_memo": {
        ("src/better_robot/model/model.py", "_value"),
        ("src/better_robot/kinematics/forward.py", "_memoised"),
    },
}
FUNCTOOLS_CACHE_NAMES = {"cache", "lru_cache", "cached_property"}
FUNCTOOLS_CACHES = {
    "src/better_robot/optim/linearization.py::_layout": (
        "the band layout of a time problem: a pure function of hashable structure (column spans, knots, block size, "
        "device), bounded at 64 entries; its owner Problem keeps its private names to problem.py, so a memo there "
        "would add a Problem method for a few index tensors"
    ),
}


def _cache_field(cls: type, name: str) -> dataclasses.Field:
    return next(field for field in dataclasses.fields(cls) if field.name == name)


def _model_modes(model: Model, kind: str) -> set[bool]:
    """The grad modes under which ``kind`` is memoised on the model (the key is ``(kind, grad mode, flags)``)."""
    return {key[1] for key in model._memo if key[0] == kind}


def _derived(state: KinematicState, model: Model) -> dict[str, torch.Tensor]:
    """One tensor per memo name of a state; the dynamics memo is read through the gravity torque."""
    zeros = torch.zeros(model.nv, dtype=model.dtype)
    return {
        "frame_transforms": state.frame_transforms,
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
    moved = dataclasses.replace(state, q=fresh.q, joint_transforms=fresh.joint_transforms)
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
    """Placements with a graph are converted per call, so each differentiable call owns its graph."""
    model = load(panda_urdf(), dtype=torch.float64)
    placements = model.joint_placements.clone().requires_grad_()
    model = model.with_values(joint_placements=placements)
    # the placements folded into the FK weight
    first, second = (joint_map(model)[0] for _ in range(2))
    assert first.requires_grad and first is not second and _model_modes(model, "joint_matrices") == set()
    forward_kinematics(model, model.q_neutral).frame_poses.sum().backward()
    assert placements.grad is not None
    with torch.no_grad():
        forward_kinematics(model, model.q_neutral)
    assert _model_modes(model, "joint_matrices") == {False}
    assert forward_kinematics(model, model.q_neutral).frame_poses.requires_grad, "a no-grad memo serves no grad call"


def test_requires_grad_after_a_query_still_yields_a_gradient() -> None:
    """``requires_grad_()`` on a model tensor is an edit C-STATE-1 forbids; the flags in the key make it harmless."""
    model = load(panda_urdf(), dtype=torch.float64)
    q = model.q_neutral
    forward_kinematics(model, q)
    assert _model_modes(model, "joint_matrices") == {True}
    model.joint_placements.requires_grad_()
    forward_kinematics(model, q).frame_translations.sum().backward()
    assert model.joint_placements.grad is not None, "the memo served the graph-free matrices"


def test_a_leaf_stepped_in_place_is_not_seen() -> None:
    """A model is read-only after construction: a leaf stepped by ``torch.optim`` needs a new ``with_values`` model
    (the differentiate guide rebuilds the model every step)."""
    model = load(panda_urdf(), dtype=torch.float64)
    q = model.q_neutral
    placements = model.joint_placements.clone().requires_grad_()
    tuned = model.with_values(joint_placements=placements)
    optimizer = torch.optim.SGD([placements], lr=1e-1)
    forward_kinematics(tuned, q).frame_translations.norm(dim=-1).sum().backward()
    before = forward_kinematics(tuned, q).frame_translations.detach().clone()
    optimizer.step()
    with torch.no_grad():
        torch.testing.assert_close(forward_kinematics(tuned, q).frame_translations, before)
        rebuilt = forward_kinematics(model.with_values(joint_placements=placements), q).frame_translations
    assert not torch.allclose(rebuilt, before), "the step changed the placements"


def test_inference_mode_values_are_served_only_under_inference_mode() -> None:
    """A value made under inference mode, even inside ``torch.enable_grad`` (grad mode reports on, autograd records
    nothing), is an inference tensor kept under its own key: inference-mode calls reuse it, and a later autograd call
    gets a fresh value and its gradient."""
    model = load(panda_urdf(), dtype=torch.float64)
    q = model.q_neutral
    with torch.inference_mode(), torch.enable_grad():
        forward_kinematics(model, q).frame_transforms
        weight = joint_map(model)[0]
        assert weight.is_inference() and joint_map(model)[0] is weight, "inference mode reuses its value"
    assert _model_modes(model, "joint_matrices") == _model_modes(model, "frame_matrices") == {"inference"}
    q = q.clone().requires_grad_()
    forward_kinematics(model, q).frame_translations.sum().backward()
    assert q.grad is not None and _model_modes(model, "joint_matrices") == {"inference", True}
    assert not joint_map(model)[0].is_inference()


def test_a_compiled_call_reads_the_model_memo_and_writes_nothing() -> None:
    """Under ``torch.compile`` the model memo is read under the grad mode, so the folded FK map an eager call stored
    enters the graph as an input instead of being derived in every compiled call, and nothing is written."""
    model = load(panda_urdf(), dtype=torch.float64)
    q = model.q_neutral.expand(2, -1)
    inputs: list[torch.Tensor] = []

    def backend(graph: torch.fx.GraphModule, example_inputs: list[torch.Tensor]):  # noqa: ANN202
        inputs.extend(example_inputs)
        return graph.forward

    torch._dynamo.reset()
    with torch.no_grad():
        forward_kinematics(model, q)
        stored, weight = dict(model._memo), joint_map(model)[0]
        torch.compile(lambda q: forward_kinematics(model, q).joint_transforms, backend=backend, fullgraph=True)(q)
    assert any(tensor.data_ptr() == weight.data_ptr() for tensor in inputs), "the stored fold is a graph input"
    assert model._memo == stored


def test_state_memo_keeps_inference_mode_values_apart() -> None:
    """A state read under ``inference_mode`` with ``enable_grad`` (grad mode reports on) stores its values under the
    inference key only, so a later autograd call of the same state gets the gradient of a fresh state."""
    model = load(panda_urdf(), dtype=torch.float64)

    def gradient(read_under_inference_mode: bool) -> torch.Tensor:
        q = (model.q_neutral + 0.2).requires_grad_()
        state = forward_kinematics(model, q)
        if read_under_inference_mode:
            with torch.inference_mode(), torch.enable_grad():
                _derived(state, model)
            assert {mode for _, mode in state._memo} == {"inference"}
        loss = sum((value**2).sum() for value in _derived(state, model).values())
        return torch.autograd.grad(loss, q)[0]

    torch.testing.assert_close(gradient(True), gradient(False), rtol=0, atol=0)


def _cache_reaches(tree: ast.AST) -> list[tuple[str, str | None, int]]:
    """``(cache, enclosing function or None, line)`` for every ``_memo`` or ``_plans`` reached by attribute or string.

    Any reach counts, read or write, so one rule covers every spelling of a write (augmented, annotated or tuple
    assignment, ``setdefault``, ``update``, ``__setitem__``, ``clear``, ``del``, an alias, ``getattr(x, "_memo")``,
    ``vars(x)["_memo"]``, an ``async def``, module level): only a cache's writer touches it at all.
    """
    reaches: list[tuple[str, str | None, int]] = []

    def visit(node: ast.AST, function: str | None) -> None:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            function = node.name
        name = node.attr if isinstance(node, ast.Attribute) else node.value if isinstance(node, ast.Constant) else None
        if isinstance(name, str) and name in CACHE_WRITERS:
            reaches.append((name, function, node.lineno))
        for child in ast.iter_child_nodes(node):
            visit(child, function)

    visit(tree, None)
    return reaches


def _violations() -> list[str]:
    found: list[str] = []
    for file in SRC.rglob("*.py"):
        tree = ast.parse(file.read_text())
        relative = file.relative_to(REPO).as_posix()
        for cache, function, line in _cache_reaches(tree):
            if (relative, function) not in CACHE_WRITERS[cache]:
                found.append(f"{relative}:{line} {function} reaches {cache}; only {sorted(CACHE_WRITERS[cache])} may")
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


def _functools_caches(tree: ast.AST) -> list[str | None]:
    """The enclosing function (``None`` at module level) of every ``functools.lru_cache``, ``cache`` or
    ``cached_property``, spelled through the module or imported by name; a decorator counts under the function it
    decorates."""
    imported = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "functools"
        for alias in node.names
        if alias.name in FUNCTOOLS_CACHE_NAMES
    }
    found: list[str | None] = []

    def visit(node: ast.AST, function: str | None) -> None:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            function = node.name
        through_module = (
            isinstance(node, ast.Attribute)
            and node.attr in FUNCTOOLS_CACHE_NAMES
            and isinstance(node.value, ast.Name)
            and node.value.id == "functools"
        )
        if through_module or (isinstance(node, ast.Name) and node.id in imported):
            found.append(function)
        for child in ast.iter_child_nodes(node):
            visit(child, function)

    visit(tree, None)
    return found


def test_functools_caches_are_the_reasoned_ones() -> None:
    """A new ``functools`` cache needs its reason in ``FUNCTOOLS_CACHES``; an entry leaves with its cache."""
    found = {
        f"{file.relative_to(REPO).as_posix()}::{function}"
        for file in SRC.rglob("*.py")
        for function in _functools_caches(ast.parse(file.read_text()))
    }
    assert found == FUNCTOOLS_CACHES.keys(), (
        f"functools caches without a reason in FUNCTOOLS_CACHES: {sorted(found - FUNCTOOLS_CACHES.keys())}; "
        f"entries whose cache is gone: {sorted(FUNCTOOLS_CACHES.keys() - found)}"
    )


def test_no_hidden_cache_mechanics_in_src() -> None:
    assert _violations() == []
    assert all(path.exists() for path in SETATTR_ALLOWED), "drop the exception once the file is gone"
    assert isinstance(torch.zeros(1)._version, int)  # the attribute exists; the library just never reads it
