# 15 · Extension Points

> **Status:** normative. Every "how do I add …" question should be
> answerable from this doc alone.

The library is intentionally small. Growth happens at a set of
**seams** — pluggable interfaces where user or contributor code joins
without touching the core. This doc lists every seam, how to use it,
and what contract it enforces.

## 0 · Seam map

```
                ┌───────────────────────────────────────────┐
                │              PUBLIC SURFACE               │
                │  load · forward_kinematics · solve_ik …   │
                └───────────────────────────────────────────┘
                                    │
           ┌────────────────────────┼───────────────────────┐
           │                        │                       │
  ┌────────▼─────────┐     ┌────────▼────────┐     ┌────────▼────────┐
  │ Parser           │     │ Residual        │     │ Optimizer       │
  │ (URDF, MJCF,     │     │ (@register_     │     │ (Protocol:      │
  │  programmatic)   │     │  residual)      │     │  .minimize)     │
  └──────────────────┘     └─────────────────┘     └─────────────────┘
           │                        │                       │
  ┌────────▼─────────┐     ┌────────▼────────┐     ┌────────▼────────┐
  │ Joint model      │     │ Robust kernel   │     │ Linear solver   │
  │ (JointModel      │     │ (Protocol:      │     │ (Protocol:      │
  │  Protocol)       │     │  .apply)        │     │  .solve)        │
  └──────────────────┘     └─────────────────┘     └─────────────────┘
           │                        │                       │
  ┌────────▼─────────┐     ┌────────▼────────┐     ┌────────▼────────┐
  │ Collision        │     │ Render mode     │     │ Backend         │
  │ primitive        │     │ (RenderMode     │     │ (backends.*)    │
  │ (@register_pair) │     │  Protocol)      │     │                 │
  └──────────────────┘     └─────────────────┘     └─────────────────┘
```

Each seam is a `typing.Protocol` — **no inheritance required**, only
structural typing. Users ship their extensions as a package that
registers itself on import.

## 1 · Add a residual

**Use when:** you need a new objective (reachability, manipulability
variant, user-defined cost).

```python
# my_package/residuals/min_torque.py
import torch
from better_robot import register_residual

@register_residual("min_torque")
class MinTorqueResidual:
    """Penalises the joint torque needed to hold the configuration static.

    dim = nv.
    """

    def __init__(self, weight: float = 1.0):
        self.weight = weight

    @property
    def dim(self) -> int:
        return self.model.nv    # set at attach time

    def attach(self, model, cost_stack) -> None:
        self.model = model

    def __call__(self, state) -> torch.Tensor:
        # state = (model, data, variables)
        from better_robot import rnea
        tau = rnea(state.model, state.data, state.q, state.v, state.a)
        return self.weight * tau

    # Optional — CostStack falls back to autodiff or FD if omitted.
    def jacobian(self, state) -> torch.Tensor:
        ...
```

Then:

```python
import better_robot as br
cost = br.CostStack()
cost.add("min_torque", br.get_residual("min_torque")(weight=0.1))
```

Contract (`Residual` Protocol — see
[07_RESIDUALS_COSTS_SOLVERS.md](07_RESIDUALS_COSTS_SOLVERS.md)):

| Member | Type | Required |
|--------|------|----------|
| `dim` | `int` property, post-attach | Yes |
| `attach(model, cost_stack)` | binds to a model | Yes |
| `__call__(state) -> Tensor` | residual `(B..., dim)` | Yes |
| `jacobian(state) -> Tensor` | `(B..., dim, nv)` | No (falls back to autodiff) |
| `sparsity() -> Tensor[bool]` | column-sparsity mask | No (assumed dense) |

The registry (`better_robot.residuals.registry`) keeps names → classes.
Re-registering the same name logs a warning and replaces — intentional,
for experimentation.

## 2 · Add a joint type

**Use when:** you need a coordinate class that does not fit the built-in
taxonomy (coupled joints, splined motion, continuous helical + angle
offset, etc.).

Built-in joint types live under `data_model/joint_models/`; each is a
single file that implements `JointModel`:

```python
# my_package/joint_models/coupled.py
from better_robot.data_model.joint_models.base import JointModel
import torch

class JointCoupled(JointModel):
    nq = 1
    nv = 1

    def __init__(self, axis: torch.Tensor, coupling: torch.Tensor):
        self.axis = axis          # (3,)
        self.coupling = coupling  # scalar

    def joint_transform(self, q: torch.Tensor) -> torch.Tensor:
        """Return SE3 7-vector (tx,ty,tz,qx,qy,qz,qw) for this joint's motion."""
        ...

    def joint_motion_subspace(self, q: torch.Tensor) -> torch.Tensor:
        """Return the 6 × nv motion subspace matrix (S)."""
        ...

    def integrate(self, q: torch.Tensor, v: torch.Tensor) -> torch.Tensor: ...
    def difference(self, q0: torch.Tensor, q1: torch.Tensor) -> torch.Tensor: ...
    def random_configuration(self, rng, lower, upper) -> torch.Tensor: ...
```

Register by passing an instance through the parser's `extensions={...}`
kwarg, or directly in a programmatic model:

```python
from better_robot.io import ModelBuilder
mb = ModelBuilder()
mb.add_joint(name="finger_1", kind=JointCoupled(axis=..., coupling=...))
```

Contract (`JointModel` Protocol — see
[02_DATA_MODEL.md §Joint Model Protocol](02_DATA_MODEL.md)):

| Member | Signature | Semantics |
|--------|-----------|-----------|
| `nq` | `int` | Config-space dim for this joint |
| `nv` | `int` | Tangent-space dim for this joint |
| `joint_transform(q)` | `(B..., nq) -> (B..., 7)` | Joint's own SE(3) motion |
| `joint_motion_subspace(q)` | `(B..., nq) -> (B..., 6, nv)` | Motion subspace `S(q)` |
| `integrate(q, v)` | `(B..., nq), (B..., nv) -> (B..., nq)` | Manifold retraction |
| `difference(q0, q1)` | `(B..., nq), (B..., nq) -> (B..., nv)` | Manifold log |
| `random_configuration(rng, lo, hi)` | → `(nq,)` | Sample respecting limits |

The FK loop calls `joint_transform`; Jacobian computation calls
`joint_motion_subspace`. Adding a new joint is isolated to one file.

## 3 · Add an optimizer

**Use when:** you want a new solver (IPOPT wrapper, DDP/iLQR, ADMM, …).

```python
# my_package/optim/ddp.py
from better_robot.optim import Optimizer, SolverState

class DDP:
    def __init__(self, *, max_iter: int = 50, tol: float = 1e-6):
        ...

    def minimize(self, problem) -> SolverState:
        """Return the final SolverState (converged or iter-capped)."""
        state = SolverState.from_problem(problem)
        for _ in range(self.max_iter):
            state = self._step(problem, state)
            if state.converged(self.tol):
                break
        return state
```

Contract (`Optimizer` Protocol):

```python
@runtime_checkable
class Optimizer(Protocol):
    def minimize(self, problem: LeastSquaresProblem) -> SolverState: ...
```

`SolverState` carries `x`, `residual_norm`, `iters`, `converged`, and
any solver-specific diagnostics (see
[07_RESIDUALS_COSTS_SOLVERS.md](07_RESIDUALS_COSTS_SOLVERS.md) for the
full schema).

A `LeastSquaresProblem` is fully self-describing — it knows its
residual, Jacobian strategy, bounds, and initial guess. The solver is
stateless across `minimize(...)` calls.

## 4 · Add a damping strategy (LM / trust region)

**Use when:** you want a non-default damping schedule for LM.

```python
# my_package/optim/strategies/dogleg.py
from better_robot.optim.strategies import DampingStrategy

class Dogleg(DampingStrategy):
    def initial(self) -> float: ...
    def accept(self, prev: float, gain_ratio: float) -> float: ...
    def reject(self, prev: float) -> float: ...
```

Then:

```python
cfg = OptimizerConfig(optimizer="lm", damping=Dogleg(radius0=0.5))
```

Contract (`DampingStrategy` Protocol):

| Member | Signature |
|--------|-----------|
| `initial()` | `float`, starting damping |
| `accept(prev, gain_ratio)` | update rule on accepted step |
| `reject(prev)` | update rule on rejected step |

## 5 · Add a linear solver

**Use when:** the problem has sparsity the default dense Cholesky can't
exploit (large trajopt), or you want KKT/iterative methods.

```python
# my_package/optim/solvers/schur.py
from better_robot.optim.solvers import LinearSolver

class Schur(LinearSolver):
    def solve(self, JtJ, Jtr) -> torch.Tensor:
        """Solve (JtJ) x = -Jtr. Return x."""
        ...
```

Contract: one method, `solve(JtJ: (n,n), Jtr: (n,)) -> (n,)`.

## 6 · Add a robust kernel

**Use when:** outlier-aware costs (Cauchy, Tukey, soft-L1).

```python
# my_package/optim/kernels/soft_l1.py
from better_robot.optim.kernels import RobustKernel

class SoftL1(RobustKernel):
    def apply(self, r: torch.Tensor) -> torch.Tensor:
        ...
    def weights(self, r: torch.Tensor) -> torch.Tensor:
        """Per-residual IRLS weights."""
        ...
```

Contract (`RobustKernel` Protocol): `apply(r)` and `weights(r)`.

## 7 · Add a collision primitive

**Use when:** new geometry (ellipsoid, superquadric, implicit SDF mesh).

```python
# my_package/collision/ellipsoid.py
from dataclasses import dataclass
import torch
from better_robot.collision.pairs import register_pair

@dataclass
class Ellipsoid:
    center: torch.Tensor          # (B..., 3)
    semi_axes: torch.Tensor       # (B..., 3)
    R: torch.Tensor               # (B..., 3, 3)

@register_pair(Ellipsoid, "Sphere")
def sdf_ellipsoid_sphere(a: Ellipsoid, b) -> torch.Tensor:
    ...
    return d                       # (B..., 1)
```

Contract: each pair is a function `(Primitive, Primitive) -> (B..., 1)`
returning signed distance. `register_pair` keys on `(type(a), type(b))`
and auto-registers the reversed order when symmetric. See
[09_COLLISION_GEOMETRY.md](09_COLLISION_GEOMETRY.md).

## 8 · Add a render mode

**Use when:** custom visualization (point-cloud overlay, occupancy grid,
debug vectors).

```python
# my_package/viewer/my_mode.py
from better_robot.viewer import RenderMode

class MyMode(RenderMode):
    name = "my_mode"

    def is_available(self, model) -> bool: ...
    def attach(self, ctx) -> None: ...
    def update(self, ctx, data) -> None: ...
    def set_visible(self, visible: bool) -> None: ...
    def detach(self) -> None: ...
```

Register:

```python
from better_robot.viewer import MODE_REGISTRY
MODE_REGISTRY["my_mode"] = MyMode
```

Contract (`RenderMode` Protocol — see [12_VIEWER.md](12_VIEWER.md)).

## 9 · Add a parser format

**Use when:** robot description in a format other than URDF/MJCF (SDF,
DRAKE YAML, custom XML).

```python
# my_package/io/parse_sdf.py
from better_robot.io import IRModel, register_parser

@register_parser(suffix=".sdf")
def parse_sdf(source) -> IRModel:
    ir = IRModel(...)
    ...
    return ir
```

`better_robot.load(path)` dispatches on suffix. The parser returns an
`IRModel` (intermediate representation); `build_model` finalises it.
See [04_PARSERS.md](04_PARSERS.md).

## 10 · Add a backend

**Use when:** swapping the tensor primitive layer (Warp, JAX, custom
CUDA kernels).

This seam is **experimental in v1** — only `torch_native` is shipped.
The skeleton is:

```
src/better_robot/backends/
├── torch_native/
│   └── ops.py          # matmul, scan, gather, …
├── warp/
│   ├── bridge.py       # wp.from_torch / wp.to_torch + autograd.Function
│   ├── kernels/
│   │   ├── fk.py       # @wp.kernel
│   │   └── rnea.py
│   └── graph_capture.py
└── your_backend/
    └── ops.py
```

Contract: each backend provides a module-level registry of primitive ops
keyed by name (`matmul`, `scan`, `gather`, `cross3`, …). The top-level
library calls `backends.current().matmul(...)` rather than
`torch.matmul(...)` in **backend-neutral** hot paths.

**Important:** the user-facing type stays `torch.Tensor`. A Warp kernel
receives a torch tensor, converts via `wp.from_torch`, runs, converts
back. Autograd wires via `torch.autograd.Function`. See
[10_BATCHING_AND_BACKENDS.md §Warp bridge](10_BATCHING_AND_BACKENDS.md).

## 11 · Registry mechanics

Every registry is a plain dict on the module:

```python
# better_robot/residuals/registry.py
_REGISTRY: dict[str, type[Residual]] = {}

def register_residual(name: str):
    def decorator(cls):
        if name in _REGISTRY:
            warnings.warn(f"residual '{name}' already registered; replacing",
                          RuntimeWarning)
        _REGISTRY[name] = cls
        return cls
    return decorator

def get_residual(name: str) -> type[Residual]:
    return _REGISTRY[name]

def list_residuals() -> tuple[str, ...]:
    return tuple(_REGISTRY)
```

Same pattern for `joint_models`, `render_modes`, `parsers`,
`collision_pairs`, `kernels`, `strategies`, `solvers`.

**Rule:** the registry is **process-local**. No auto-discovery via
entry points — users register explicitly in their package's
`__init__.py`. This keeps imports fast and failures loud.

## 12 · What is **not** pluggable (deliberately)

| Thing | Why not |
|-------|---------|
| `Model` / `Data` schemas | Would break every algorithm. Extend via `Model.meta` dict or a wrapper class. |
| The SE(3) representation `[tx,ty,tz,qx,qy,qz,qw]` | Every algorithm depends on it. Change requires a major version. |
| `LeastSquaresProblem` structure | Freeze the problem contract so solvers are interchangeable. |
| Layer DAG | If you want to import from a higher layer, refactor instead. |

## 13 · Pre-merge checklist for an extension

Every new extension PR:

1. **Adds one registry entry** — not more, not less.
2. **Passes the `Protocol` check** — `isinstance(instance, Protocol)` is True.
3. **Ships a unit test** in `tests/my_extension/` that:
   - Exercises the happy path on a toy model.
   - Exercises one failure mode (bad input, missing method).
4. **Updates exactly one cross-cutting doc** (this file) if the extension
   is generally useful, or ships its own doc in `docs/extensions/`.
5. **Does not alter the 25-symbol public API.** If you think it must,
   open an issue and discuss first.
