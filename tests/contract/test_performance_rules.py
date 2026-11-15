"""Performance rules a check can read: no Python loop over a batch (C-PERF-2), compilation only on request (C-PERF-5).

C-PERF-2 is checked by behaviour: every entry point of ``tests/contract/_public_ops.py`` dispatches the same number
of torch operators for a batch of two as for a batch of seven, so no Python loop runs per batch element. C-PERF-5 reads
the source: ``torch.compile`` is called only inside ``kinematics.compiled``, never decorates a function, and no module
imports a kernel builder or the compiler's own modules, whose settings are process-wide.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from better_robot.kinematics import KinematicState, forward_kinematics

from ._public_ops import CASE_NAMES, build_case

SRC = Path(__file__).resolve().parents[2] / "src" / "better_robot"
BATCHES = (2, 7)
KERNEL_BUILDERS = ("torch.utils.cpp_extension", "triton", "numba", "cupy")
# Their configuration and decomposition tables are process-wide: a module that imports them can change every
# compile of the process (record 0053).
COMPILER_MODULES = ("torch._dynamo", "torch._inductor")
# The one function that calls torch.compile, on the caller's request (record 0053).
ENTRY_POINT = ("src/better_robot/kinematics/compilation.py", "compiled")


class _OperatorCount(TorchDispatchMode):
    def __init__(self) -> None:
        super().__init__()
        self.count = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):  # noqa: ANN001, ANN204
        self.count += 1
        return func(*args, **(kwargs or {}))


def _batched(value: Any, model: Any, batch: int) -> Any:
    """``value`` with a leading batch axis of ``batch``; a state is recomputed from its batched configuration, and an
    integer table (a mesh's faces) is shared by the batch."""
    if isinstance(value, torch.Tensor) and value.is_floating_point():
        return value.expand(batch, *value.shape)
    if isinstance(value, KinematicState):
        return forward_kinematics(model, value.q.expand(batch, *value.q.shape))
    return value


def _operators(fn: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> int:
    with torch.no_grad(), _OperatorCount() as counter:
        fn(*args, **kwargs)
    return counter.count


@pytest.mark.slow  # 1.5 s for 66 entry points: the full tier, so the fast tier keeps its budget
@pytest.mark.parametrize("name", CASE_NAMES)
def test_a_batch_dispatches_the_same_operators_at_every_size(name: str) -> None:
    """C-PERF-2: the operator count does not grow with the batch, so nothing loops over batch elements."""
    fn, args, kwargs = build_case(name, torch.float64, torch.device("cpu"))
    model = None if isinstance(args[0], torch.Tensor) else args[0]
    counts = []
    for batch in BATCHES:
        batched_args = tuple(_batched(arg, model, batch) for arg in args)
        batched_kwargs = {key: _batched(value, model, batch) for key, value in kwargs.items()}
        _operators(fn, batched_args, batched_kwargs)  # the first call builds plans and memos
        counts.append(_operators(fn, batched_args, batched_kwargs))
    assert counts[0] == counts[1], f"{name}: {counts[0]} operators at batch {BATCHES[0]}, {counts[1]} at {BATCHES[1]}"


def _is_torch_compile(node: ast.AST) -> bool:
    return isinstance(node, ast.Attribute) and node.attr == "compile" and ast.unparse(node.value) == "torch"


def test_only_the_entry_point_compiles_and_nothing_builds_a_kernel() -> None:
    """C-PERF-5: compilation is the caller's request through ``kinematics.compiled`` (record 0053), and the only
    implementation is torch (record 0001)."""
    found = []
    for path in sorted(SRC.rglob("*.py")):
        relative = str(path.relative_to(SRC))
        tree = ast.parse(path.read_text())
        allowed = {
            id(node)
            for entry in tree.body
            if path == SRC.parents[1] / ENTRY_POINT[0]
            and isinstance(entry, ast.FunctionDef)
            and entry.name == ENTRY_POINT[1]
            for statement in entry.body
            for node in ast.walk(statement)
            if _is_torch_compile(node)
        }
        for node in ast.walk(tree):
            if _is_torch_compile(node) and id(node) not in allowed:
                found.append(f"{relative}: torch.compile")
            if (
                isinstance(node, ast.ImportFrom)
                and node.module == "torch"
                and {"compile", "_dynamo", "_inductor"} & {a.name for a in node.names}
            ):
                found.append(f"{relative}: from torch import {', '.join(a.name for a in node.names)}")
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                modules = [node.module]
            forbidden = KERNEL_BUILDERS + COMPILER_MODULES
            found += [f"{relative}: {module}" for module in modules if module.startswith(forbidden)]
    assert not found, found
