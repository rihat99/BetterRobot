"""Opt-in compilation of library calls: `compiled`, the one place the library calls ``torch.compile`` (C-PERF-5).

Eager stays the default; a caller who repeats a call asks for a compiled one (record 0053). The wrapper handles what
measurement found to go wrong when a library call is handed to ``torch.compile`` directly:

- Plans and memoised model values built inside the trace. A model's first call builds its plans; traced, that
  call compiles a graph that rebuilds them (386 operators for the Panda's FK instead of 33) and is then replaced by
  a recompile. A memoised value, such as the placements folded into the FK map, was derived again in every compiled
  call. The first call under each grad mode now runs eagerly once, and a compiled call reads the model memo
  (`Model._value`), so both enter the graph as inputs.
- Batched products of small matrices (``3x4 @ 4x4``, ``3x3 @ 3x1``, ``6x6 @ 6x1``) left by inductor as extern
  batched matrix products, about half of a compiled batch-4096 call on CPU. `small_matmul` writes them as a
  multiply and a sum under ``torch.compile``, which inductor fuses with their neighbours, and as ``@`` in eager mode,
  where that form is slower.
- CUDA graphs (``mode="reduce-overhead"``) replay into one memory pool, so the next call overwrites the tensors the
  previous call returned, without an error. The wrapper clones every CUDA tensor of the result (`_owned`).
- Batch sizes. ``torch.compile``'s own policy applies: the first batch size compiles a graph specialised to it, a
  second size one graph for every size, and batch 1 one graph of its own. A graph for every size is up to a fifth
  slower on CPU, so this is cheaper than compiling for every size from the start.
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Callable
from typing import Any, Literal, ParamSpec, TypeVar

import torch
from torch.utils import _pytree

from ..exceptions import ArgumentError

_P = ParamSpec("_P")
_R = TypeVar("_R")
_MODES = ("default", "reduce-overhead")


def small_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``a @ b`` for batches of small matrices, ``(..., m, k) @ (..., k, n) -> (..., m, n)``.

    Under ``torch.compile`` a multiply and a sum over ``k``, which inductor fuses into the neighbouring kernels;
    eagerly ``@``, as the multiply and sum would materialise a ``(..., m, k, n)`` product.
    """
    if torch.compiler.is_compiling():
        return (a.unsqueeze(-1) * b.unsqueeze(-3)).sum(-2)
    return a @ b


def compiled(fn: Callable[_P, _R], *, mode: Literal["default", "reduce-overhead"] = "default") -> Callable[_P, _R]:
    """``fn`` compiled by ``torch.compile`` into one graph on its first call; same arguments and results.

    ``fn`` is a function of tensors that calls the library, such as
    ``lambda q: forward_kinematics(model, q).joint_transforms``; a graph break in it raises. The first call under
    each grad mode runs ``fn`` eagerly once, so the model's plans and memoised values exist before the trace, then
    compiles (seconds per graph). ``mode="reduce-overhead"`` replays CUDA graphs on CUDA, which removes the launch
    cost of small batches; the result's CUDA tensors, also those in the fields of a dataclass result, are clones,
    valid after the next call (a replay overwrites its own outputs without an error). Batch sizes follow
    ``torch.compile``'s policy (module docstring). Raises `ArgumentError` for another ``mode``.
    """
    if mode not in _MODES:
        raise ArgumentError(f"mode must be one of {_MODES}, got {mode!r}")
    graph = torch.compile(fn, fullgraph=True, mode=None if mode == "default" else mode)
    warmed: set[bool] = set()

    @functools.wraps(fn)
    def call(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        grad = torch.is_grad_enabled()
        if grad not in warmed:
            if torch.is_inference_mode_enabled():  # memoise under the no-grad key, the one a compiled call reads
                with torch.inference_mode(False), torch.no_grad():
                    fn(*args, **kwargs)
            else:
                fn(*args, **kwargs)
            warmed.add(grad)
        result = graph(*args, **kwargs)
        return _owned(result) if mode == "reduce-overhead" else result

    return call


def _owned(result: Any) -> Any:
    """``result`` with its CUDA tensors cloned out of the CUDA-graph pool: tensors in tuples, lists and dicts, and in
    the fields of a dataclass result (such as a `KinematicState` or a solver ``State``) and their containers. A field
    that is itself a dataclass, such as a state's model, is the caller's object and is kept."""

    def own(leaf: Any) -> Any:
        if isinstance(leaf, torch.Tensor):
            return leaf.clone() if leaf.is_cuda else leaf
        if not _is_instance(leaf):
            return leaf
        changes = {}
        for field in dataclasses.fields(leaf):
            value = getattr(leaf, field.name)
            if field.init and not _is_instance(value):
                changes[field.name] = _pytree.tree_map_only(torch.Tensor, own, value)
        return dataclasses.replace(leaf, **changes)

    return _pytree.tree_map(own, result, is_leaf=_is_instance)


def _is_instance(value: Any) -> bool:
    return dataclasses.is_dataclass(value) and not isinstance(value, type)


__all__ = ["compiled"]
