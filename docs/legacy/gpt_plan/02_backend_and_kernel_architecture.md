# Backend And Kernel Architecture

## Current State

The backend package is mostly a placeholder:

- `backends.current_backend()` returns a global string.
- `backends.set_backend("warp")` raises `NotImplementedError`.
- `backends.graph_capture` is a no-op.
- `backends/torch_native/ops.py` has no real kernels.
- `backends/warp` has bridge and graph-capture skeletons.

This is fine for a skeleton, but too weak for a long-lived package. A global backend switch will become hard to reason about with Torch compile, tests that need both implementations, per-kernel capability selection, and mixed Torch/Warp execution.

## Recommendation

Design backends around kernel families, not package-wide mode.

Use a small explicit backend protocol and let high-level code default to Torch. Warp should be an acceleration path behind the same tensor-returning API.

## Backend Layers

```text
public/task API
  -> algorithm functions: kinematics, dynamics, residuals, collision
    -> functional tensor kernels
      -> Backend protocol
        -> torch backend implementation
        -> warp backend implementation wrapped by torch.autograd.Function
```

No layer above `backends` should import Warp.

## Backend Protocol Sketch

```python
from typing import Protocol
import torch

class LieKernels(Protocol):
    def so3_exp(self, omega: torch.Tensor) -> torch.Tensor: ...
    def so3_log(self, quat: torch.Tensor) -> torch.Tensor: ...
    def se3_exp(self, xi: torch.Tensor) -> torch.Tensor: ...
    def se3_log(self, pose: torch.Tensor) -> torch.Tensor: ...

class KinematicsKernels(Protocol):
    def forward_kinematics(self, model_tensors, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: ...
    def joint_jacobians(self, model_tensors, q: torch.Tensor, poses: torch.Tensor) -> torch.Tensor: ...

class Backend(Protocol):
    name: str
    lie: LieKernels
    kinematics: KinematicsKernels
    capabilities: frozenset[str]
```

This should stay small. Do not create a generic op registry until multiple real kernel implementations need it.

## Backend Selection

Prefer explicit but quiet selection:

- `ExecutionConfig(backend="torch")` or `BackendConfig`.
- `Model` can carry a preferred backend in metadata only if that does not affect equality or serialization.
- Algorithm functions can take `backend: Backend | None = None`.
- `None` means "choose default for this input/device/capability".

Avoid:

- mutable process-global backend as the only mechanism,
- importing Warp at package import time,
- returning backend-specific arrays,
- letting user-level APIs require backend objects for normal use.

The current `set_backend` can remain as a convenience during migration, but it should become a context/config helper rather than the core architecture.

## Capability Matrix

Every backend should declare what it can do. Example capabilities:

- `lie.so3`
- `lie.se3`
- `fk.static_topology`
- `fk.batched`
- `jacobian.dense`
- `rnea`
- `crba`
- `collision.spheres`
- `collision.capsules`
- `autograd.forward`
- `autograd.backward`
- `graph_capture`

Tests can then assert fallback behavior:

- If Warp lacks a kernel, Torch executes the operation.
- If fallback would break performance assumptions, raise `BackendNotAvailableError` with a clear message.

## Warp Integration Rules

Warp is a kernel compiler, not a user-visible data model.

Rules:

- No `wp.array` in public API.
- No raw `wp.launch` above `backends/warp`.
- No `wp.Tape` as the primary autograd mechanism.
- Every differentiable Warp kernel is wrapped by `torch.autograd.Function`.
- Backward kernels are explicit, tested, and finite-difference checked.
- Torch tensors remain the only inputs and outputs.
- Stream bridging is explicit and tested.
- Graph capture is opt-in and must have a Torch no-op path.

## Kernel Source Policy

Warp kernels should live in real files, not dynamic strings. Suggested layout:

```text
backends/warp/
  bridge.py
  graph_capture.py
  cache.py
  kernels/
    lie_so3.py
    lie_se3.py
    fk.py
    jacobian.py
    rnea.py
    collision_spheres.py
```

Use content-addressed caching for compiled kernels. Avoid global caches that silently mix devices, dtypes, topology, or stream state.

## Model Tensor View

Warp and Torch kernels should consume a compact, tensor-only view of `Model`:

```python
@dataclass(frozen=True)
class ModelTensors:
    parents: torch.Tensor
    idx_qs: torch.Tensor
    idx_vs: torch.Tensor
    nqs: torch.Tensor
    nvs: torch.Tensor
    joint_kind_ids: torch.Tensor
    joint_axes: torch.Tensor
    joint_placements: torch.Tensor
    body_inertias: torch.Tensor
```

`Model` can keep Python tuples, dicts, names, and protocols for readability. Kernels should not need those at runtime.

## Adaptive Dispatch

Borrow the cuRobo lesson: dispatch should depend on batch size, topology, and capability.

Examples:

- Small CPU input: Python/Torch loop is fine.
- Large CUDA batch: specialized Torch vectorized or Warp kernel.
- Static topology and repeated calls: cached kernel plan.
- Long horizon: temporal kernels or matrix-free Jacobian products.

Add a `KernelPlan` object later only after there are at least two real implementations.

## Autograd Policy

For every differentiable backend kernel:

- Forward returns Torch tensors.
- Backward is either pure Torch autograd or a custom `torch.autograd.Function`.
- Backward agrees with central finite differences at documented tolerances.
- Tangent-space derivatives are documented separately from ambient tensor derivatives.

This matters most for Lie operations and pose residual Jacobians.

## Dtype And Precision Policy

The docs currently disagree: one file suggests fp16 is acceptable for FK, while contracts reject fp16 in kinematics and optimization. Choose one policy before backend work.

Recommended v1 policy:

- Supported: `torch.float32`, `torch.float64`.
- Rejected at public boundaries: `float16`, `bfloat16`.
- Experimental internal kernels may benchmark reduced precision but cannot be public contract.

## Tests To Add

- Backend imports are lazy.
- Torch backend and pure functional API return identical values.
- Capability fallback behaves predictably.
- Warp absence raises `BackendNotAvailableError`, not `ImportError` from deep inside a call.
- No non-backend module imports `warp`.
- `torch.compile` does not observe a changing process-global backend.
- Kernel results match Torch references across CPU/CUDA, dtype, batch shape, and horizon shape.
