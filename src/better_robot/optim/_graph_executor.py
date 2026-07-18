"""Experimental lazy CUDA-graph execution with an eager CPU fallback.

This module stays internal until a production caller can own stable problem
inputs long enough to amortize capture. ``GraphExecutor`` owns stable copies
of a callable's tensor-pytree inputs.
CUDA calls warm up on a side stream, record lazily, update those stable inputs
with ``copy_``, and replay until the pytree layout, shape, stride, dtype,
storage offset, or device changes. A changed signature is re-recorded in a
controlled way.

Every returned tensor leaf is cloned.  Results therefore remain valid after a
later replay and never expose the graph's static output buffers.  The callable
must be a pure, fixed-structure tensor program: non-tensor input/output leaves,
input tensors requiring gradients, sparse tensors, and mixed CPU/CUDA inputs
are rejected on the captured path. Overlapping/zero-stride input views and
outputs requiring gradients are rejected as well. Autograd operations *inside*
the callable may still be captured (for example, named-block LM's tangent
Jacobian work), but this executor is not an autograd bridge for differentiating
through replay.
One executor is not safe for concurrent host calls; callers must serialize
access. Sequential use from a different CUDA stream is detected, synchronized,
and re-recorded instead of racing shared static buffers.
Only explicit call arguments participate in the signature. Tensor storage or
Python configuration reachable only through the callable's closure must stay
alive and unchanged for the graph lifetime. Make changing tensors explicit
arguments, or call :meth:`GraphExecutor.reset` before replacing closure state.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import gc
import time
from typing import Any

import torch
from torch.utils._pytree import tree_flatten, tree_unflatten


class GraphCaptureError(RuntimeError):
    """A callable could not be safely recorded as a CUDA graph."""


@dataclass(frozen=True)
class _TensorSignature:
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    storage_offset: int
    dtype: torch.dtype
    device: torch.device
    layout: torch.layout


@dataclass(frozen=True)
class _CallSignature:
    tree_spec: Any
    tensors: tuple[_TensorSignature, ...]
    aliases: tuple[int, ...]


def _tensor_leaves(tree: Any, *, label: str) -> tuple[list[torch.Tensor], Any]:
    leaves, spec = tree_flatten(tree)
    non_tensors = tuple(type(leaf).__name__ for leaf in leaves if not isinstance(leaf, torch.Tensor))
    if non_tensors:
        raise TypeError(
            f"GraphExecutor {label} must be a tensor-only pytree; "
            f"found non-tensor leaves {non_tensors}"
        )
    return list(leaves), spec


def _clone_tensor_tree(tree: Any) -> Any:
    leaves, spec = _tensor_leaves(tree, label="outputs")
    return tree_unflatten([leaf.clone() for leaf in leaves], spec)


def _signature(leaves: list[torch.Tensor], tree_spec: Any) -> _CallSignature:
    identities: dict[int, int] = {}
    aliases: list[int] = []
    for tensor in leaves:
        identity = id(tensor)
        if identity not in identities:
            identities[identity] = len(identities)
        aliases.append(identities[identity])
    return _CallSignature(
        tree_spec=tree_spec,
        tensors=tuple(
            _TensorSignature(
                shape=tuple(tensor.shape),
                stride=tuple(tensor.stride()),
                storage_offset=int(tensor.storage_offset()),
                dtype=tensor.dtype,
                device=tensor.device,
                layout=tensor.layout,
            )
            for tensor in leaves
        ),
        aliases=tuple(aliases),
    )


def _overlapping_storage_pairs(leaves: list[torch.Tensor]) -> tuple[tuple[int, int], ...]:
    spans: list[tuple[int, int, int, int]] = []
    for index, tensor in enumerate(leaves):
        if tensor.numel() == 0:
            continue
        storage_pointer = tensor.untyped_storage().data_ptr()
        storage_offset = int(tensor.storage_offset())
        start = storage_offset * tensor.element_size()
        last_element = storage_offset + sum(
            (int(size) - 1) * int(stride)
            for size, stride in zip(tensor.shape, tensor.stride(), strict=True)
        )
        end = (last_element + 1) * tensor.element_size()
        spans.append((index, storage_pointer, start, end))

    overlapping: list[tuple[int, int]] = []
    for left_index, left_storage, left_start, left_end in spans:
        for right_index, right_storage, right_start, right_end in spans:
            if right_index <= left_index or leaves[left_index] is leaves[right_index]:
                continue
            if left_storage == right_storage and left_start < right_end and right_start < left_end:
                overlapping.append((left_index, right_index))
    return tuple(overlapping)


class GraphExecutor:
    """Wrap a fixed tensor-pytree callable with lazy CUDA graph replay.

    Parameters
    ----------
    function
        Pure callable whose positional and keyword inputs, and whose output,
        are tensor-only pytrees with fixed structure for one recording.
    enabled
        When false, always execute eagerly. CPU-only calls also execute eagerly
        even when enabled.
    warmup_runs
        Number of side-stream executions before recording. Three follows the
        repository's CUDA-capture protocol and is the default.

    Notes
    -----
    ``record_count`` and ``replay_count`` are cumulative for the executor's
    lifetime. :meth:`reset` clears graph-owned state but intentionally retains
    those observability counters; the next CUDA call records again. Instances
    are not thread-safe: concurrent host calls must be externally serialized.
    A sequential caller-stream change is synchronized and re-recorded.
    Closure state is deliberately not inspected: callers must keep it stable
    until :meth:`reset`, and pass every changing tensor as an explicit input.
    """

    def __init__(
        self,
        function: Callable[..., Any],
        *,
        enabled: bool = True,
        warmup_runs: int = 3,
    ) -> None:
        if not callable(function):
            raise TypeError("GraphExecutor function must be callable")
        if not isinstance(enabled, bool):
            raise TypeError("GraphExecutor enabled must be a bool")
        if isinstance(warmup_runs, bool) or not isinstance(warmup_runs, int) or warmup_runs < 0:
            raise ValueError("GraphExecutor warmup_runs must be a non-negative int")
        self.function = function
        self.enabled = enabled
        self.warmup_runs = warmup_runs
        self.record_count = 0
        self.replay_count = 0
        self.last_record_ms: float | None = None
        self._record_times_ms: list[float] = []
        self._graph: torch.cuda.CUDAGraph | None = None
        self._signature: _CallSignature | None = None
        self._input_buffers: list[torch.Tensor] = []
        self._input_spec: Any = None
        self._static_outputs: Any = None
        self._device: torch.device | None = None
        self._capture_stream: torch.cuda.Stream | None = None
        self._owner_stream_id: int | None = None

    @property
    def is_captured(self) -> bool:
        """Whether this executor currently owns a recorded CUDA graph."""
        return self._graph is not None

    @property
    def input_buffer_ptrs(self) -> tuple[int, ...]:
        """Stable captured-input addresses, exposed for lifecycle tests."""
        return tuple(buffer.data_ptr() for buffer in self._input_buffers)

    @property
    def record_times_ms(self) -> tuple[float, ...]:
        """Successful graph-record durations, excluding side-stream warmup."""
        return tuple(self._record_times_ms)

    @property
    def total_record_ms(self) -> float:
        """Cumulative successful graph-record time in milliseconds."""
        return sum(self._record_times_ms)

    def _clear_capture(self, *, synchronize: bool) -> None:
        if synchronize and self._device is not None and self._device.type == "cuda":
            torch.cuda.synchronize(self._device)
        self._graph = None
        self._signature = None
        self._input_buffers = []
        self._input_spec = None
        self._static_outputs = None
        self._device = None
        self._capture_stream = None
        self._owner_stream_id = None

    def reset(self) -> None:
        """Release the current graph and static buffers; counters are retained."""
        self._clear_capture(synchronize=True)
        gc.collect()

    @staticmethod
    def _capture_device(leaves: list[torch.Tensor]) -> torch.device | None:
        if not leaves or all(tensor.device.type == "cpu" for tensor in leaves):
            return None
        devices = {tensor.device for tensor in leaves}
        if any(device.type != "cuda" for device in devices) or len(devices) != 1:
            raise ValueError(
                "GraphExecutor capture requires every tensor input on one CUDA device; "
                f"got {sorted(map(str, devices))}"
            )
        return next(iter(devices))

    @staticmethod
    def _validate_capture_inputs(leaves: list[torch.Tensor]) -> None:
        if any(tensor.requires_grad for tensor in leaves):
            raise ValueError(
                "GraphExecutor captured inputs must not require gradients; capture autograd "
                "work inside the callable and return detached solver artifacts"
            )
        unsupported = tuple(index for index, tensor in enumerate(leaves) if tensor.layout != torch.strided)
        if unsupported:
            raise ValueError(f"GraphExecutor capture supports only strided tensors; invalid leaves {unsupported}")

        nonzero_offsets = tuple(
            index for index, tensor in enumerate(leaves) if tensor.storage_offset() != 0
        )
        if nonzero_offsets:
            raise ValueError(
                "GraphExecutor capture requires storage_offset()==0 because static buffers "
                f"do not preserve view offsets; invalid leaves {nonzero_offsets}"
            )

        zero_strides = tuple(
            index for index, tensor in enumerate(leaves) if any(stride == 0 for stride in tensor.stride())
        )
        if zero_strides:
            raise ValueError(
                "GraphExecutor capture rejects stride-zero inputs because copy_ into a raw "
                f"empty_strided buffer is ambiguous; invalid leaves {zero_strides}"
            )

        negative_strides = tuple(
            index for index, tensor in enumerate(leaves) if any(stride < 0 for stride in tensor.stride())
        )
        if negative_strides:
            raise ValueError(
                "GraphExecutor capture rejects negative-stride inputs; "
                f"invalid leaves {negative_strides}"
            )

        internally_overlapping = tuple(
            index for index, tensor in enumerate(leaves) if GraphExecutor._may_overlap_itself(tensor)
        )
        if internally_overlapping:
            raise ValueError(
                "GraphExecutor capture rejects internally overlapping input views; "
                f"invalid leaves {internally_overlapping}"
            )

        overlapping_pairs = _overlapping_storage_pairs(leaves)
        if overlapping_pairs:
            raise ValueError(
                "GraphExecutor capture rejects distinct input leaves with overlapping storage; "
                f"invalid leaf pairs {overlapping_pairs}"
            )

    @staticmethod
    def _may_overlap_itself(tensor: torch.Tensor) -> bool:
        """Conservatively prove that a strided tensor has unique element addresses."""
        span = 1
        dimensions = sorted(
            (stride, size)
            for size, stride in zip(tensor.shape, tensor.stride(), strict=True)
            if size > 1
        )
        for stride, size in dimensions:
            if stride < span:
                return True
            span += (size - 1) * stride
        return False

    @staticmethod
    def _validate_capture_outputs(outputs: Any) -> None:
        leaves, _ = _tensor_leaves(outputs, label="outputs")
        differentiable = tuple(index for index, tensor in enumerate(leaves) if tensor.requires_grad)
        if differentiable:
            raise ValueError(
                "GraphExecutor captured outputs must not require gradients because replay is "
                "not an autograd bridge; detach closure parameters or use eager execution. "
                f"Invalid leaves {differentiable}"
            )

    @staticmethod
    def _new_buffer(tensor: torch.Tensor) -> torch.Tensor:
        buffer = torch.empty_strided(
            tensor.shape,
            tensor.stride(),
            dtype=tensor.dtype,
            device=tensor.device,
        )
        with torch.no_grad():
            buffer.copy_(tensor)
        return buffer

    @classmethod
    def _new_buffers(cls, leaves: list[torch.Tensor]) -> list[torch.Tensor]:
        by_identity: dict[int, torch.Tensor] = {}
        buffers: list[torch.Tensor] = []
        for tensor in leaves:
            identity = id(tensor)
            if identity not in by_identity:
                by_identity[identity] = cls._new_buffer(tensor)
            buffers.append(by_identity[identity])
        return buffers

    def _copy_inputs(self, leaves: list[torch.Tensor]) -> None:
        copied: set[int] = set()
        with torch.no_grad():
            for buffer, value in zip(self._input_buffers, leaves, strict=True):
                if buffer.data_ptr() in copied:
                    continue
                if buffer.data_ptr() != value.data_ptr():
                    buffer.copy_(value)
                copied.add(buffer.data_ptr())

    def _static_call(self) -> Any:
        packed = tree_unflatten(self._input_buffers, self._input_spec)
        args, kwargs = packed
        return self.function(*args, **kwargs)

    def _record(
        self,
        leaves: list[torch.Tensor],
        input_spec: Any,
        signature: _CallSignature,
        device: torch.device,
    ) -> Any:
        self._clear_capture(synchronize=True)
        gc.collect()
        torch.cuda.synchronize(device)
        self._device = device
        self._signature = signature
        self._input_spec = input_spec
        self._input_buffers = self._new_buffers(leaves)
        capture_stream = torch.cuda.Stream(device=device)
        self._capture_stream = capture_stream
        current_stream = torch.cuda.current_stream(device)
        owner_stream_id = int(current_stream.cuda_stream)
        capture_stream.wait_stream(current_stream)

        try:
            with torch.cuda.stream(capture_stream):
                warmup_output = None
                for _ in range(self.warmup_runs):
                    warmup_output = self._static_call()
                    self._validate_capture_outputs(warmup_output)
                del warmup_output
            current_stream.wait_stream(capture_stream)
            torch.cuda.synchronize(device)

            record_started = time.perf_counter()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=capture_stream):
                static_outputs = self._static_call()
            current_stream.wait_stream(capture_stream)
            torch.cuda.synchronize(device)
            record_ms = (time.perf_counter() - record_started) * 1_000.0
            self._validate_capture_outputs(static_outputs)
        except Exception as error:
            self._clear_capture(synchronize=False)
            raise GraphCaptureError(
                f"CUDA graph recording failed for {self.function!r}: {error}"
            ) from error

        self._graph = graph
        self._static_outputs = static_outputs
        # Stream capture records operations but does not promise that the
        # captured output buffers contain this call's values. Launch once to
        # materialize the public first result; replay_count tracks only later
        # calls that reuse an already-recorded signature.
        graph.replay()
        self.record_count += 1
        self.last_record_ms = record_ms
        self._record_times_ms.append(record_ms)
        self._owner_stream_id = owner_stream_id
        return _clone_tensor_tree(static_outputs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        packed_inputs = (args, kwargs)
        leaves, input_spec = _tensor_leaves(packed_inputs, label="inputs")
        device = self._capture_device(leaves)
        if not self.enabled or device is None or not torch.cuda.is_available():
            if self.is_captured:
                self.reset()
            return _clone_tensor_tree(self.function(*args, **kwargs))

        self._validate_capture_inputs(leaves)
        signature = _signature(leaves, input_spec)
        current_stream_id = int(torch.cuda.current_stream(device).cuda_stream)
        stream_changed = self._owner_stream_id is not None and current_stream_id != self._owner_stream_id
        if self._graph is None or signature != self._signature or stream_changed:
            return self._record(leaves, input_spec, signature, device)

        self._copy_inputs(leaves)
        self._graph.replay()
        self.replay_count += 1
        return _clone_tensor_tree(self._static_outputs)


__all__ = ["GraphCaptureError", "GraphExecutor"]
