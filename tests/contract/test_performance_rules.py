"""Performance rules a check can read: no Python loop over a batch (C-PERF-2), compilation only on request (C-PERF-5).

C-PERF-2 is checked by behaviour: every entry point of ``tests/contract/_public_ops.py`` dispatches the same number
of torch operators for a batch of two as for a batch of seven, so no Python loop runs per batch element. C-PERF-5 reads
the source, resolving import aliases: ``torch.compile`` appears once, inside ``kinematics.compiled``, and no module
names a kernel builder, the compiler's own modules (whose settings are process-wide), ``torch.compiler`` beyond
``is_compiling`` and ``assume_constant_result``, ``torch.jit`` or ``torch.library``, imported, as an attribute or
through ``getattr``.
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
# Kernel builders, the compiler's process-wide state (its configs, decomposition tables, stance), a second compiler
# and custom operators: no module of src may name them, imported or as an attribute.
FORBIDDEN = (
    "torch.utils.cpp_extension",
    "triton",
    "numba",
    "cupy",
    "torch._dynamo",
    "torch._inductor",
    "torch.compiler",
    "torch.jit",
    "torch.library",
)
# The parts of torch.compiler that neither compile nor change a setting.
COMPILER_READS = ("torch.compiler.is_compiling", "torch.compiler.assume_constant_result")
# The one function that calls torch.compile, once, on the caller's request.
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


def _torch_names(tree: ast.Module) -> list[tuple[ast.AST, str]]:
    """Every dotted name a module refers to through torch or a kernel builder, resolved through its import aliases
    (``import torch as t``, ``from torch.utils import cpp_extension``): imports, outermost attribute chains, names
    bound by an import, and ``getattr`` with a constant name."""
    aliases, names = {}, []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                aliases[alias.asname or alias.name.split(".")[0]] = (
                    alias.name if alias.asname else alias.name.split(".")[0]
                )
                names.append((node, alias.name))
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            for alias in node.names:
                aliases[alias.asname or alias.name] = f"{node.module}.{alias.name}"
                names.append((node, f"{node.module}.{alias.name}"))
    inner = {id(node.value) for node in ast.walk(tree) if isinstance(node, ast.Attribute)}

    def dotted(node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            return aliases.get(node.id)
        if isinstance(node, ast.Attribute):
            base = dotted(node.value)
            return None if base is None else f"{base}.{node.attr}"
        return None

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute | ast.Name) and id(node) not in inner:
            name = dotted(node)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
        ):
            base = dotted(node.args[0])
            name = None if base is None else f"{base}.{node.args[1].value}"
        else:
            continue
        if name is not None:
            names.append((node, name))
    return names


def _under(name: str, prefixes: tuple[str, ...]) -> bool:
    return any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes)


def test_only_the_entry_point_compiles_and_nothing_builds_a_kernel() -> None:
    """C-PERF-5: compilation is the caller's request through ``kinematics.compiled``, which calls ``torch.compile``
    once; nothing else compiles, changes a compiler setting or builds a kernel, and the only
    implementation is torch."""
    found = []
    for path in sorted(SRC.rglob("*.py")):
        relative = str(path.relative_to(SRC))
        tree = ast.parse(path.read_text())
        entry = [
            id(node)
            for statement in tree.body
            if path == SRC.parents[1] / ENTRY_POINT[0]
            and isinstance(statement, ast.FunctionDef)
            and statement.name == ENTRY_POINT[1]
            for node in ast.walk(statement)
        ]
        compiles = 0
        for node, name in _torch_names(tree):
            if name == "torch.compile" and id(node) in entry and compiles == 0:
                compiles += 1
            elif name == "torch.compile" or (_under(name, FORBIDDEN) and not _under(name, COMPILER_READS)):
                found.append(f"{relative}:{getattr(node, 'lineno', 0)}: {name}")
    assert not found, found
