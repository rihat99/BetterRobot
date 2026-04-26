# 02 · Make the backend abstraction load-bearing now, not later

★★★ **Foundational.** Pre-condition for the Warp backend (
[10_BATCHING_AND_BACKENDS.md §7](../../design/10_BATCHING_AND_BACKENDS.md))
and for [Proposal 03](03_replace_pypose.md) (PyPose replacement).

## Problem

`backends/` exists but is not on the call path. Today:

- `backends/__init__.py` defines `current_backend()`, `set_backend()`,
  `BackendName` (literal `"torch_native" | "warp"`).
- `backends/torch_native/ops.py` is a stub — "real kernels live in
  `kinematics/`".
- `backends/warp/bridge.py`, `backends/warp/graph_capture.py`,
  `backends/warp/kernels/` are stubs that raise `NotImplementedError`.
- **No file outside `backends/` calls `backends.current()`.** `lie/se3`
  imports `_pypose_backend` directly; `kinematics/forward.py` calls
  `lie.se3.compose` directly.

The plan in
[10_BATCHING_AND_BACKENDS.md §7.How it plugs in](../../design/10_BATCHING_AND_BACKENDS.md)
is:

> `kinematics/forward.py` is unchanged — it calls
> `backends.current().forward_kinematics(model, q)`. The torch_native
> backend implements it as described in 05; the Warp backend
> dispatches to `backends/warp/kernels/fk.py`.

The contradiction: the "torch_native backend" doesn't exist as
something `current()` returns. There is no object with a
`.forward_kinematics(...)` method. When Warp lands, every hot-path
call site has to be retrofitted. That is exactly when retrofitting is
most expensive — under deadline pressure to ship the perf win.

## What "load-bearing now" looks like

Make the dispatch real today, with a single backend. Routing five or
six primitive ops through the active backend costs almost nothing,
and *the discipline of writing the routes prevents new direct
imports from sneaking in*.

> **Architectural note.** The primary mechanism is **explicit backend
> objects**, not a process-global selector. A global default exists
> as a *convenience* (so users don't pass a backend on every call),
> but every algorithm can take an explicit `backend: Backend | None`
> override. This is the gpt_plan-driven correction to an earlier
> draft of this proposal that made `backends.current()` the load-
> bearing entry point. Reasons:
>
> - `torch.compile` traces against captured globals; switching the
>   global mid-trace silently invalidates compiled artefacts.
> - Tests that compare two backends side by side need both live in
>   the same process without monkey-patching.
> - Nested libraries that themselves call `set_backend` would clash.
>
> The visible surface stays small: `current()` and `set_backend()`
> remain for users who want a one-line global preference, but they
> are sugar over `BackendConfig` / explicit dispatch — not the
> architectural core.

## The proposal

### 1. A backend `Protocol`

```python
# src/better_robot/backends/protocol.py
from __future__ import annotations
from typing import Protocol, TYPE_CHECKING
import torch

if TYPE_CHECKING:
    from ..data_model.model import Model
    from ..data_model.data  import Data

class LieOps(Protocol):
    """Bottom-layer SE3/SO3 ops. Each takes plain torch tensors and returns
    plain torch tensors. Backward (autograd) compatibility is the backend's
    responsibility."""

    # SE3
    def se3_compose   (self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor: ...
    def se3_inverse   (self, t: torch.Tensor) -> torch.Tensor: ...
    def se3_log       (self, t: torch.Tensor) -> torch.Tensor: ...
    def se3_exp       (self, v: torch.Tensor) -> torch.Tensor: ...
    def se3_act       (self, t: torch.Tensor, p: torch.Tensor) -> torch.Tensor: ...
    def se3_adjoint   (self, t: torch.Tensor) -> torch.Tensor: ...
    def se3_adjoint_inv(self, t: torch.Tensor) -> torch.Tensor: ...
    def se3_normalize (self, t: torch.Tensor) -> torch.Tensor: ...
    # SO3 (mirror, omitted here)
    ...

class KinematicsOps(Protocol):
    """Hot-path kinematics. Single entry per algorithm."""
    def forward_kinematics(self, model: "Model", q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: ...
    def compute_joint_jacobians(self, model: "Model", joint_pose_world: torch.Tensor) -> torch.Tensor: ...

class DynamicsOps(Protocol):
    """Hot-path dynamics. Each method maps to a doc-06 algorithm."""
    def rnea (self, model, data, q, v, a, fext=None) -> torch.Tensor: ...
    def aba  (self, model, data, q, v, tau, fext=None) -> torch.Tensor: ...
    def crba (self, model, data, q) -> torch.Tensor: ...
    def center_of_mass(self, model, data, q, v=None, a=None) -> torch.Tensor: ...

class Backend(Protocol):
    """The full backend surface. A concrete backend is a single object
    exposing these three sub-namespaces."""
    name: str
    lie:        LieOps
    kinematics: KinematicsOps
    dynamics:   DynamicsOps
```

### 2. The `torch_native` backend

```python
# src/better_robot/backends/torch_native/__init__.py

from . import lie_ops, kinematics_ops, dynamics_ops

class TorchNativeBackend:
    name = "torch_native"
    lie        = lie_ops
    kinematics = kinematics_ops
    dynamics   = dynamics_ops

BACKEND = TorchNativeBackend()
```

```python
# src/better_robot/backends/torch_native/lie_ops.py
"""Today: routes through PyPose via lie/_pypose_backend.py.
After Proposal 03: routes through lie/_torch_backend.py."""

from ...lie import _pypose_backend as _pp

def se3_compose   (a, b): return _pp.se3_compose(a, b)
def se3_inverse   (t):    return _pp.se3_inverse(t)
def se3_log       (t):    return _pp.se3_log(t)
def se3_exp       (v):    return _pp.se3_exp(v)
def se3_act       (t, p): return _pp.se3_act(t, p)
def se3_adjoint   (t):    return _pp.se3_adjoint(t)
def se3_adjoint_inv(t):   return _pp.se3_adjoint_inv(t)
def se3_normalize (t):    return _pp.se3_normalize(t)
# ...
```

`backends/torch_native/kinematics_ops.py` and `dynamics_ops.py` host
the existing torch implementations; the files in
`kinematics/forward.py` and `kinematics/jacobian.py` become **thin
dispatchers** that call into the active backend.

### 3. The `lie/` modules become backend-aware shims

```python
# src/better_robot/lie/se3.py
from ..backends import default_backend

def compose(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    backend: "Backend | None" = None,
) -> torch.Tensor:
    """SE3 composition. (..., 7), (..., 7) -> (..., 7)."""
    return (backend or default_backend()).lie.se3_compose(a, b)

def inverse(t: torch.Tensor, *, backend: "Backend | None" = None) -> torch.Tensor:
    return (backend or default_backend()).lie.se3_inverse(t)
# ...
```

`default_backend()` returns the active default — `torch_native`
unless the user has set otherwise. `set_backend("warp")` rewrites
the default; tests and advanced users pass `backend=` explicitly to
opt in or out per call. Public functions that take a `Model` and a
`Data` carry the same `backend=` kwarg; `Model.meta` may *advise* a
backend but does not bind one.

### 4. `backends/__init__.py`

```python
# src/better_robot/backends/__init__.py
from __future__ import annotations
from typing import Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from .protocol import Backend

BackendName = Literal["torch_native", "warp"]

_DEFAULT_NAME: str = "torch_native"
_CACHE: dict[str, "Backend"] = {}

def default_backend() -> "Backend":
    """Return the current default backend. Cached per name."""
    if _DEFAULT_NAME not in _CACHE:
        _CACHE[_DEFAULT_NAME] = _load(_DEFAULT_NAME)
    return _CACHE[_DEFAULT_NAME]

# Convenience alias retained for users who imported it pre-refactor.
current = default_backend

def current_backend() -> str:
    return _DEFAULT_NAME

def set_backend(name: BackendName) -> None:
    """Set the process-wide default backend. Convenience only —
    explicit `backend=` kwargs override this per-call. Avoid calling
    this from inside library code; pass an explicit backend instead."""
    global _DEFAULT_NAME
    if name not in ("torch_native", "warp"):
        raise ValueError(f"unknown backend {name!r}")
    if name == "warp":
        _ensure_warp_available()
    _DEFAULT_NAME = name

def get_backend(name: BackendName) -> "Backend":
    """Return a specific backend by name without changing the default.
    Use this in tests and advanced workflows that compare backends."""
    if name not in _CACHE:
        _CACHE[name] = _load(name)
    return _CACHE[name]

def _load(name: str) -> "Backend":
    if name == "torch_native":
        from .torch_native import BACKEND
        return BACKEND
    if name == "warp":
        from .warp import BACKEND  # raises BackendNotAvailableError if warp absent
        return BACKEND
    raise ValueError(name)

def _ensure_warp_available() -> None:
    try:
        import warp  # noqa: F401
    except ImportError as e:
        from ..exceptions import BackendNotAvailableError
        raise BackendNotAvailableError(
            "warp backend requires `pip install better-robot[warp]`"
        ) from e
```

### 5. Where the backend boundary actually lives

| Module | Calls backend? | Today |
|--------|----------------|-------|
| `lie/se3.py`, `lie/so3.py` | yes | calls `_pypose_backend` directly |
| `lie/tangents.py` | no — pure PyTorch math | unchanged |
| `kinematics/forward.py` | yes (FK kernel) | calls `lie.se3.compose` directly |
| `kinematics/jacobian.py` | yes (joint jacobian assembly) | calls `lie.se3.adjoint` |
| `kinematics/jacobian.py:residual_jacobian` | no — uses central FD | unchanged |
| `dynamics/rnea.py`, `aba.py`, `crba.py` | yes (when implemented) | stubs |
| `spatial/motion.py`, `force.py`, `inertia.py` | indirectly (call into `lie/`) | unchanged |
| `data_model/joint_models/*.py` | indirectly | unchanged |
| Everything above `kinematics/` | no | unchanged |

The rule: **`lie/`, `kinematics/`, and `dynamics/` are the only
modules that cross the backend boundary.** Everything else composes
through them.

A new contract test enforces this:

```python
# tests/contract/test_backend_boundary.py
def test_only_lie_kinematics_dynamics_call_backend():
    forbidden = {"better_robot.backends"}
    allowed   = {"better_robot.lie", "better_robot.kinematics",
                 "better_robot.dynamics", "better_robot.backends.torch_native",
                 "better_robot.backends.warp"}
    for path in walk("src/better_robot"):
        ...
```

### 6. The Warp backend stays a stub

The directory layout in
[10 §7.Directory layout](../../design/10_BATCHING_AND_BACKENDS.md)
already exists. Under this proposal, the Warp backend object becomes
the *next* file to fill in, not a new thing to design — its protocol
is already pinned by `Backend`/`LieOps`/`KinematicsOps`/`DynamicsOps`.

## Why now, not at Phase 7

- The cost of touching `lie/se3.py` to add one line of dispatch is
  small *while it has no users*. Once external code is calling
  `lie.se3.compose` directly and depending on its identity (e.g. for
  monkeypatching in tests), the change becomes a deprecation cycle.
- Without the dispatch, the contract test that "no other module
  imports `pypose`" only covers half of the boundary. The full
  contract is "no other module imports a backend implementation
  module" — and we cannot enforce that without a real backend
  selector.
- It forces the question of *which protocol the Warp backend has to
  implement* now, when the answer is short. By the time RNEA derivatives
  land we will know the surface — and we will know it because it was
  pinned by the protocols this proposal adds.

## Tradeoffs

| For | Against |
|-----|---------|
| Phase 7 becomes a one-file landing (`backends/warp/__init__.py`). | A small overhead per call (Python attribute lookup `current().lie.se3_compose`). Negligible vs. the kernel cost; gets erased by `torch.compile`. |
| Contract test pins the surface; new direct imports are caught. | The protocol has to be kept in sync as new ops are added. We treat it as part of the public API. |
| Proposal 03 (PyPose replacement) is also a one-file change. | Adds a layer of indirection that may surprise contributors. Mitigation: a single section in `01_ARCHITECTURE.md` plus a CLAUDE.md note in `backends/`. |
| Backend identity becomes testable; tests can opt into the Warp backend by calling `get_backend("warp")` without mutating process state. | The convenience global still exists; users who race `set_backend` from threads will surprise themselves. We document `set_backend` as a one-time configuration call, not a runtime knob; the explicit `backend=` kwarg is the safe path. |
| Warp landing is **isolated**, not necessarily one-file: the protocol pins the surface, but the Warp backend still owns its kernel cache, stream bridge, autograd wrappers, graph-capture policy, dtype/device checks, and parity tests. The protocol makes Warp swappable; it does not make Warp trivial. | None significant — this is honest accounting, not a tradeoff. |

## Files that change

```
backends/protocol.py                    new — Backend / LieOps / KinematicsOps / DynamicsOps
backends/__init__.py                    extended — current() returns the cached object
backends/torch_native/__init__.py       new — TorchNativeBackend
backends/torch_native/lie_ops.py        new — routes to lie/_pypose_backend (today)
backends/torch_native/kinematics_ops.py new — routes to kinematics/forward._fk_impl etc.
backends/torch_native/dynamics_ops.py   new — stubs (until D2/D3/D4)
backends/warp/__init__.py               extended — exports BACKEND, raises if warp missing
lie/se3.py, lie/so3.py                  modified — call current().lie.* instead of _pp.*
kinematics/forward.py                   modified — keep _fk_impl; backend dispatches to it
kinematics/jacobian.py                  modified — same pattern
tests/contract/test_backend_boundary.py new — AST walk
```

## Acceptance criteria

- `from better_robot.backends import default_backend; default_backend().lie.se3_compose(a, b)`
  works and returns the same value as `lie.se3.compose(a, b)` to
  bit-precision in fp32 and fp64.
- `lie.se3.compose(a, b, backend=get_backend("torch_native"))` works
  and bypasses the global default.
- `set_backend("warp")` raises `BackendNotAvailableError` if warp is
  not installed, without touching `_DEFAULT_NAME`.
- `tests/contract/test_layer_dependencies.py` passes — no module
  outside the backend or `lie/`, `kinematics/`, `dynamics/` imports
  any `_pypose_backend`/`_torch_backend`/`backends.warp.kernels`
  identifier.
- `tests/contract/test_backend_boundary.py` (new) passes — no
  call site outside `lie/`, `kinematics/`, `dynamics/` imports a
  backend implementation; no library-internal call site uses
  `set_backend()` (the contract test greps for it).
- The benchmark `tests/bench/bench_forward_kinematics.py` does not
  regress beyond the 5% noise floor on the day this lands.
- `Proposal 03` (PyPose replacement) lands by swapping one file
  (`backends/torch_native/lie_ops.py`); the Warp landing isolates
  to `backends/warp/` (the protocol pins the surface) but is not
  promised to be a single file — Warp owns its own kernel cache,
  stream bridge, autograd wrappers, graph-capture policy, and
  parity tests.

## Cross-references

- [Proposal 03](03_replace_pypose.md) — what swaps in once this is in
  place.
- [10_BATCHING_AND_BACKENDS.md §7](../../design/10_BATCHING_AND_BACKENDS.md) —
  the doc this proposal makes operational.
- [01_ARCHITECTURE.md](../../design/01_ARCHITECTURE.md) — gets a new
  paragraph: "the backend boundary is `lie/`, `kinematics/`,
  `dynamics/`; everywhere else routes through them."
