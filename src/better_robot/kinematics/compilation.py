"""Opt-in compilation of library calls: `compiled`, the one place the library calls ``torch.compile`` (C-PERF-5).

Eager stays the default; a caller who repeats a call asks for a compiled one. The wrapper handles what
measurement found to go wrong when a library call is handed to ``torch.compile`` directly:

- Plans and memoised model values built inside the trace. A model's first call builds its plans; traced, that
  call compiles a graph that rebuilds them (386 operators for the Panda's FK instead of 33) and is then replaced by
  a recompile. A memoised value, such as the placements folded into the FK map, was derived again in every compiled
  call. The first call of each key (below), and of each model argument new to it, runs eagerly, and a compiled call
  reads the model memo (`Model._value`), so both enter the graph as inputs.
- Batched products of small matrices (``3x4 @ 4x4``, ``3x3 @ 3x1``, ``6x6 @ 6x1``) left by inductor as extern
  batched matrix products, about half of a compiled batch-4096 call on CPU. While `compiled` traces
  (`fused_forms`), `small_matmul` writes them as a multiply and a sum, which inductor fuses with their neighbours;
  otherwise as ``@``: eagerly the multiply and sum is slower, and a caller's own ``torch.compile`` with a backend
  that does not fuse (``backend="eager"``) would materialise it (G1 mass matrix at batch 4096: 298 ms against 97).
- CUDA graphs (``mode="reduce-overhead"``) replay into one memory pool, so the next call overwrites the tensors the
  previous call returned, without an error. The wrapper clones every CUDA tensor of the result (`_owned`).
- Torch keeps at most 8 graphs per compiled function, and ``fullgraph=True`` makes a ninth an error. Grad modes,
  dtypes, frame names and models each need a graph of their own, so each key (the grad and inference mode and the
  `_signature` of the arguments) gets its own ``torch.compile`` with its own 8 (``isolate_recompiles``), and only
  batch sizes count against them. The first call of a key and of a shape holds a lock: threads compiling one key at
  once compiled a graph each.
- Batch sizes. ``torch.compile``'s own policy applies: the first batch size compiles a graph specialised to it, a
  second size one graph for every size, and batch 1 one graph of its own. A graph for every size is up to a fifth
  slower on CPU, so this is cheaper than compiling for every size from the start.
- Errors. A library error raised inside the graph reaches the caller as a torch error; the call then runs eagerly,
  which raises the library's own class.
"""

from __future__ import annotations

import dataclasses
import enum
import functools
import threading
import weakref
from collections.abc import Callable, Hashable
from typing import Any, Literal, ParamSpec, TypeVar, cast, final

import torch
from torch.utils import _pytree

from ..exceptions import ArgumentError
from ..model.model import Model

_P = ParamSpec("_P")
_R = TypeVar("_R")
_MODES = ("default", "reduce-overhead")
# Values a key holds as they are; an object of another type adds only its type (`_signature`).
_VALUES = (str, int, float, complex, bool, type(None), torch.dtype, torch.device, enum.Enum)


@final
class _Tracing(threading.local):
    active = False  # true while `compiled` runs a graph in this thread


_TRACING = _Tracing()


@torch.compiler.assume_constant_result  # read once while tracing, then a constant of the graph, without a guard
def _entry_point_active() -> bool:
    return _TRACING.active


def fused_forms() -> bool:
    """Whether `compiled` is tracing the calling kernel, which then takes the forms inductor fuses."""
    return torch.compiler.is_compiling() and _entry_point_active()


def small_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``a @ b`` for batches of small matrices, ``(..., m, k) @ (..., k, n) -> (..., m, n)``.

    While `compiled` traces, a multiply and a sum over ``k``, which inductor fuses into the neighbouring kernels;
    otherwise ``@``, as the multiply and sum would materialise a ``(..., m, k, n)`` product.
    """
    if fused_forms():
        return (a.unsqueeze(-1) * b.unsqueeze(-3)).sum(-2)
    return a @ b


def compiled(fn: Callable[_P, _R], *, mode: Literal["default", "reduce-overhead"] = "default") -> Callable[_P, _R]:
    """``fn`` compiled by ``torch.compile`` into one graph; same arguments and results.

    ``fn`` is a function of tensors that calls the library, such as
    ``lambda q: forward_kinematics(model, q).joint_transforms``; a graph break in it raises. Calls are grouped by
    key: the grad mode and inference mode, and for each argument, also inside tuples, lists, dicts and dataclasses,
    a tensor's dtype, device, ``requires_grad`` and number of axes; a model's name and its tensors'; a string,
    number, dtype, device or enum itself (a number in a dataclass, such as a solver state's iteration, only its
    type); of another object only its type, so a new problem or function of the same kind shares the key and torch's
    guards tell them apart. The first call of a key runs ``fn`` eagerly and returns its result, which builds the
    plans and memoised values; the second compiles (seconds); later calls run the graph. A model argument new to a
    key runs eagerly once too, so a graph never builds plans: a model made anew for every call runs eagerly every
    time. A side effect of ``fn`` runs once per call. The returned function keeps its own graphs: build it once, not
    once per call.

    Each key compiles at most 8 graphs, torch's limit, and a ninth raises. Batch sizes take three (the first size,
    one graph for every later size, and batch 1). A value ``fn`` reads from outside its arguments (a closure or
    global variable, such as a frame name) takes a graph of its key each time it changes: pass it as an argument. A
    Python number argument that changes on every call (an iteration count) compiles once per value, up to torch's
    256 graphs per function: pass it as a tensor. Calls that may compile (a new key or shape) run one at a time.

    ``mode="reduce-overhead"`` replays CUDA graphs on CUDA, which removes the launch cost of small batches; every
    CUDA tensor of the result, at any depth of tuples, lists, dicts and dataclasses, is a clone, valid after the next
    call (a replay overwrites its own outputs without an error); a model in the result is the caller's object. A
    library error raised inside the graph is raised again by an eager run of the call, as the library's own class.
    Raises `ArgumentError` for another ``mode``.
    """
    if mode not in _MODES:
        raise ArgumentError(f"mode must be one of {_MODES}, got {mode!r}")
    graphs: dict[Hashable, Callable[_P, _R]] = {}  # each key's own torch.compile, made by its eager first call
    warmed: dict[Hashable, weakref.WeakSet[Model]] = {}  # the model arguments each key's eager calls warmed
    ready: dict[Hashable, Callable[_P, _R]] = {}  # (key, shapes) that ran its graph once: no compile ahead
    lock = threading.Lock()

    def run(graph: Callable[_P, _R], /, *args: _P.args, **kwargs: _P.kwargs) -> _R:
        _TRACING.active = True
        try:
            result = graph(*args, **kwargs)
        except Exception:
            fn(*args, **kwargs)  # eagerly, a library error raises as its own class
            raise
        finally:
            _TRACING.active = False
        return _owned(result) if mode == "reduce-overhead" else result

    @functools.wraps(fn)
    def call(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        shapes: list[torch.Size] = []
        models: list[Model] = []
        key = (torch.is_grad_enabled(), torch.is_inference_mode_enabled(), _signature(args, shapes, models))
        if kwargs:
            key += (_signature(kwargs, shapes, models),)
        seen = (key, *shapes)
        graph = ready.get(seen)
        if graph is not None and (not models or warmed[key].issuperset(models)):
            return run(graph, *args, **kwargs)
        with lock:
            if key not in graphs or not warmed[key].issuperset(models):
                if torch.is_inference_mode_enabled():  # memoise under the no-grad key, the one a compiled call reads
                    with torch.inference_mode(False), torch.no_grad():
                        result = fn(*args, **kwargs)
                else:
                    result = fn(*args, **kwargs)
                if key not in graphs:
                    # torch 2.13's overloads of torch.compile omit isolate_recompiles, which the function takes
                    graphs[key] = cast("Any", torch.compile)(
                        fn, fullgraph=True, mode=None if mode == "default" else mode, isolate_recompiles=True
                    )
                    warmed[key] = weakref.WeakSet()
                warmed[key].update(models)
                return result
            result = run(graphs[key], *args, **kwargs)
            ready[seen] = graphs[key]
            return result

    return call


def _signature(value: Any, shapes: list[torch.Size], models: list[Model], *, in_dataclass: bool = False) -> Hashable:
    """What of ``value`` selects the graphs of `compiled`'s key; appends the shape of every tensor to ``shapes`` and
    every model to ``models``.

    A number in a dataclass field, such as a solver state's iteration, adds only its type: torch makes a number that
    changes dynamic after one recompile, while a key per value would compile at every step.
    """
    if isinstance(value, torch.Tensor):
        shapes.append(value.shape)
        return (value.dtype, value.device, value.requires_grad, value.ndim)
    if isinstance(value, (tuple, list)):
        return (type(value), *[_signature(item, shapes, models, in_dataclass=in_dataclass) for item in value])
    if isinstance(value, dict):
        return (
            type(value),
            *[(name, _signature(item, shapes, models, in_dataclass=in_dataclass)) for name, item in value.items()],
        )
    if isinstance(value, Model):
        models.append(value)
        return (Model, value.name, *[_signature(tensor, shapes, models) for tensor in _pytree.tree_leaves(value)])
    if _is_instance(value):
        fields = dataclasses.fields(value)
        return (
            type(value),
            *[
                _signature(getattr(value, field.name), shapes, models, in_dataclass=True)
                for field in fields
                if field.init
            ],
        )
    number = isinstance(value, (int, float, complex)) and not isinstance(value, bool)
    return (type(value), value) if isinstance(value, _VALUES) and not (in_dataclass and number) else type(value)


def _map_tensors(apply: Callable[[torch.Tensor], torch.Tensor], value: Any) -> Any:
    """``value`` with ``apply`` applied to every tensor at any depth of tuples, lists, dicts and dataclasses; a
    `Model` is kept as it is, and a dataclass is rebuilt from its init fields."""

    def visit(leaf: Any) -> Any:
        if isinstance(leaf, torch.Tensor):
            return apply(leaf)
        if isinstance(leaf, Model) or not _is_instance(leaf):
            return leaf
        fields = dataclasses.fields(leaf)
        return dataclasses.replace(
            leaf, **{f.name: _map_tensors(apply, getattr(leaf, f.name)) for f in fields if f.init}
        )

    return _pytree.tree_map(visit, value, is_leaf=lambda leaf: isinstance(leaf, Model) or _is_instance(leaf))


def _owned(result: Any) -> Any:
    """``result`` with its CUDA tensors cloned out of the CUDA-graph pool (`_map_tensors`)."""
    return _map_tensors(lambda tensor: tensor.clone() if tensor.is_cuda else tensor, result)


def _is_instance(value: Any) -> bool:
    return dataclasses.is_dataclass(value) and not isinstance(value, type)


__all__ = ["compiled"]
