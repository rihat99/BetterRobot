# Extension Points

> **Status:** normative. Every "how do I add …" question should be
> answerable from this doc alone.

A small library is a feature. The temptation, when a user asks for a
new optimiser, a custom joint, or a new collision primitive, is to add
a flag, a config field, or a switch statement somewhere in the core.
That path leads to the kind of optimisation library where every solver
loop has eighteen branches because every kind of caller had its own
special case.

We took the opposite path: the core is small, and growth happens at
**seams** — pluggable interfaces where user or contributor code joins
without touching the core. Public plugin seams use `typing.Protocol`
(no inheritance, no MRO surprises, no metaclass tricks) and plain-dict
registries where discovery is needed. The internal whole-pass compute seam
is intentionally narrower: it uses an explicit branch in the owning pass,
not a public plugin registry. The result is that "add an IPOPT-backed optimiser",
"register a Schur-complement linear solver", or "support a custom robot
description format" are each isolated PRs in user packages, not
patches to BetterRobot's solvers, parsers, or task layer.

This document is the canonical list of seams, their contracts, and
recipes. If you can't find your extension here, the answer is probably
"don't subclass anything — implement the matching `Protocol` and
register."

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
  │ (URDF, MJCF,     │     │ (explicit       │     │ (Protocol:      │
  │  programmatic)   │     │  construction)  │     │  .minimize)     │
  └──────────────────┘     └─────────────────┘     └─────────────────┘
           │                        │                       │
  ┌────────▼─────────┐     ┌────────▼────────┐     ┌────────▼────────┐
  │ Joint model      │     │ Robust kernel   │     │ Linear solver   │
  │ (JointModel      │     │ (Protocol:      │     │ (Protocol:      │
  │  Protocol)       │     │  .apply)        │     │  .solve)        │
  └──────────────────┘     └─────────────────┘     └─────────────────┘
           │                        │                       │
  ┌────────▼─────────┐     ┌────────▼────────┐     ┌────────▼────────┐
  │ Collision        │     │ Render mode     │     │ Whole-pass lane │
  │ primitive        │     │ (RenderMode     │     │ (internal,      │
  │ (@register_pair) │     │  Protocol)      │     │ explicit branch)│
  └──────────────────┘     └─────────────────┘     └─────────────────┘
```

Public plugin seams use structural typing rather than inheritance. Users ship
those extensions as a package that registers itself on import. Whole-pass
compute work is a contributor integration and follows §10 instead.

## 1 · Add a residual

**Use when:** you need a new objective (reachability, manipulability
variant, user-defined cost).

```python
# my_package/residuals/min_torque.py
import torch

class MinTorqueResidual:
    """Penalises the joint torque needed to hold the configuration static.

    dim = nv.
    """

    name = "min_torque"

    def __init__(self, model, weight: float = 1.0):
        self.model = model
        self.weight = weight

    @property
    def dim(self) -> int:
        return self.model.nv

    def __call__(self, state) -> torch.Tensor:
        from better_robot import rnea
        tau = rnea(state.model, state.data, state.q, state.v, state.a)
        return self.weight * tau

    # Optional — CostStack falls back to central FD if omitted.
    def jacobian(self, state) -> torch.Tensor:
        ...
```

Then:

```python
import better_robot as br
cost = br.CostStack()
cost.add("min_torque", MinTorqueResidual(model, weight=0.1))
```

Contract (`Residual` Protocol — see {doc}`/concepts/residuals_and_costs`):

| Member | Type | Required |
|--------|------|----------|
| `name` | stable class/instance name | Yes |
| `dim` | `int` property | Yes |
| `__call__(state) -> Tensor` | residual `(B..., dim)` | Yes |
| `jacobian(state) -> Tensor` | `(B..., dim, nv)` | No (unbatched central-FD fallback) |
| `sparsity() -> Tensor[bool]` | column-sparsity mask | No (assumed dense) |

Residuals are instantiated and added explicitly. BetterRobot does not keep a
process-wide residual registry.

## 2 · Add a joint type

**Use when:** you need a coordinate class that does not fit the
built-in taxonomy (coupled joints, splined motion, helical with
non-constant pitch, etc.).

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
        self.axis = axis
        self.coupling = coupling

    def joint_transform(self, q: torch.Tensor) -> torch.Tensor:
        """Return SE3 7-vector for this joint's motion."""
        ...

    def joint_motion_subspace(self, q: torch.Tensor) -> torch.Tensor:
        """Return the 6 × nv motion subspace matrix S(q)."""
        ...

    def integrate(self, q, v) -> torch.Tensor: ...
    def difference(self, q0, q1) -> torch.Tensor: ...
    def random_configuration(self, rng, lower, upper) -> torch.Tensor: ...

    # Dynamics hooks (default: zeros). Override for non-trivial joints.
    def joint_bias_acceleration(self, q, v) -> torch.Tensor:           # (B..., 6)
        ...
    def joint_motion_subspace_derivative(self, q, v) -> torch.Tensor:  # (B..., 6, nv)
        ...
```

Register by passing an instance through the parser's `extensions={...}`
kwarg, or directly in a programmatic model:

```python
from better_robot.io import ModelBuilder
mb = ModelBuilder()
mb.add_joint(name="finger_1", kind=JointCoupled(axis=..., coupling=...))
```

Contract (`JointModel` Protocol):

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

## 3 · Add an optimiser

**Use when:** you want a new solver (IPOPT wrapper, DDP / iLQR, ADMM, …).

```python
# my_package/optim/ddp.py
from better_robot.optim import Optimizer, SolverState

class DDP:
    def __init__(self, *, max_iter: int = 50, tol: float = 1e-6):
        ...

    def minimize(self, problem) -> SolverState:
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
any solver-specific diagnostics. A `LeastSquaresProblem` is fully
self-describing — it knows its residual, Jacobian strategy, bounds, and
initial guess. The solver is stateless across `minimize(...)` calls.

## 4 · Add a damping strategy (LM / trust region)

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

## 5 · Add a linear solver

**Use when:** the problem has sparsity the default dense Cholesky
cannot exploit (large trajopt), or you want KKT / iterative methods.

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

```python
# my_package/optim/kernels/soft_l1.py
import torch

from better_robot.optim.kernels.base import RobustKernel

class SoftL1(RobustKernel):
    def rho(self, squared_norm: torch.Tensor) -> torch.Tensor:
        """Soft-L1 objective ``sqrt(1 + s) - 1``."""
        return torch.sqrt(1.0 + squared_norm) - 1.0

    def weight(self, squared_norm: torch.Tensor) -> torch.Tensor:
        """Normalized IRLS weight ``2·rho'(s)``."""
        return torch.rsqrt(1.0 + squared_norm)
```

LM uses `rho(s)` to accept trial steps and `weight(s)` to form the IRLS
normal equations. Both methods preserve the input shape; built-in kernels
use the normalized convention `weight(s) = 2·rho'(s)`.

## 7 · Add a collision primitive

```python
# my_package/collision/ellipsoid.py
from dataclasses import dataclass
import torch
from better_robot.collision.pairs import register_pair

@dataclass
class Ellipsoid:
    center: torch.Tensor
    semi_axes: torch.Tensor
    R: torch.Tensor

@register_pair(Ellipsoid, "Sphere")
def sdf_ellipsoid_sphere(a: Ellipsoid, b) -> torch.Tensor:
    ...
    return d                       # (B..., 1)
```

Contract: each pair is a function `(Primitive, Primitive) -> (B..., 1)`
returning signed distance. `register_pair` keys on `(type(a), type(b))`
and auto-registers the reversed order when symmetric.

## 8 · Add a render mode

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

## 9 · Add a parser format

**Use when:** robot description in a format other than URDF / MJCF
(SDF, Drake YAML, custom XML).

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
`IRModel`; `build_model` finalises it. See
{doc}`/concepts/parsers_and_ir`.

## 10 · Add a whole-pass compute lane

**Use when:** a measured FK, Jacobian, or dynamics pass needs a specialised
device kernel. This is an internal performance integration, not a public
plugin Protocol.

Keep the implementation beside the Torch counterpart in the package that
owns the pass. The public wrapper remains unchanged and the raw Torch pass
remains the default correctness oracle. Add one explicit selection branch at
the whole-pass boundary; do not add selectors to Lie primitives or mutable
process-global configuration.

An opt-in lane must:

1. consume `ModelStructure`, `ModelValues`, and ordinary Torch tensors;
2. reject or intentionally fall back for unsupported joint kinds, devices,
   dtypes, and shapes;
3. match the Torch raw pass on shared forward fixtures;
4. expose a tested analytic adjoint through Torch autograd without leaking
   optional runtime arrays;
5. preserve the flat-`E` `ExecutionBatch` mapping when inputs broadcast; and
6. keep optional imports local to the owning pass.

For Warp, conversion and custom-autograd plumbing belong in that local
integration module. Graph capture is separate roadmap work: when added, it
must record forward and backward together and must not invent a public
capture decorator before the solver lifecycle is capture-safe.

## 11 · Registry mechanics where discovery exists

Only discovery-oriented seams keep process-local dictionaries today:
parser suffixes, viewer render modes/renderers, and collision primitive
pairs. Residuals are constructed and added to a `CostStack` explicitly;
optimizers, kernels, strategies, and solvers are likewise passed as objects
or configuration choices rather than discovered through a global table.

**Rule:** do not add entry-point auto-discovery. A documented registry is a
plain dictionary, and extension packages register explicitly on import. This
keeps imports fast and failures loud. Whole-pass compute lanes never use a
registry.

## 12 · Trajectory parameterisations

**Use when:** you want a different optimisation variable shape for
trajopt (e.g. log/cosine basis, segmented polynomials, time-warped knot
grids). The Protocol owns the mapping `z ↔ Trajectory`:

```python
# my_package/parameterizations/log_basis.py
from better_robot.tasks.parameterization import TrajectoryParameterization, Trajectory
import torch

class LogBasisTrajectory(TrajectoryParameterization):
    @property
    def tangent_dim(self) -> int: ...
    def unpack(self, z: torch.Tensor) -> Trajectory: ...
    def retract(self, z: torch.Tensor, dz: torch.Tensor) -> torch.Tensor: ...
    def pack_initial(self, traj: Trajectory) -> torch.Tensor: ...
```

The shipped implementations are `KnotTrajectory` (identity) and
`BSplineTrajectory` (cuRobo-style smooth). See {doc}`/concepts/tasks`.

## 13 · Asset resolvers

**Use when:** you have a custom mesh-storage scheme (ROS package map, S3
bucket, embedded asset bundle, in-process cache).

```python
from pathlib import Path
from better_robot.io.assets import AssetResolver

class S3AssetResolver(AssetResolver):
    def __init__(self, bucket: str, cache_dir: Path) -> None: ...
    def resolve(self, uri: str, *, base_path: Path | None = None) -> Path: ...
    def exists (self, uri: str, *, base_path: Path | None = None) -> bool: ...

import better_robot as br
model = br.load("robot.urdf", resolver=S3AssetResolver(...))
```

Concrete resolvers shipped in core: `FilesystemResolver`,
`PackageResolver`, `CompositeResolver`, `CachedDownloadResolver`.

## 14 · Add an actuator (e.g. muscle)

**Use when:** you have a non-rigid actuator that contributes to joint
torque (DeGrooteFregly muscles, splined dampers, …).

A `Muscle` Protocol composes with `dynamics/`:

```python
class Muscle(Protocol):
    name: str
    nu: int
    def compute_force(self, q, v, u) -> Tensor:    # (B..., 1)
    def moment_arm(self, model, q) -> Tensor:      # (B..., nv)
```

The dispatch into `rnea` / `aba` is via the existing `fext` / additive
joint-space torque term:
`tau += sum(muscle.compute_force(q,v,u) * muscle.moment_arm(model,q))`.
`dynamics/` stays muscle-agnostic.

OpenSim DeGrooteFregly2016 muscles ship in the sibling
`better_robot_human` extension package (under the `[human]` extra);
core BetterRobot does not import `chumpy`, SMPL, or OpenSim.

## 15 · What is **not** pluggable (deliberately)

| Thing | Why not |
|-------|---------|
| `Model` / `Data` schemas | Would break every algorithm. Extend via `Model.meta` dict or a wrapper class. |
| The SE(3) representation `[tx,ty,tz,qx,qy,qz,qw]` | Every algorithm depends on it. Change requires a major version. |
| `LeastSquaresProblem` structure | Freeze the problem contract so solvers are interchangeable. |
| Layer DAG | If you want to import from a higher layer, refactor instead. |
| `IRModel` shape | Internal parser/build boundary; re-parse assets after upgrades. |

## 16 · Pre-merge checklist for an extension

Every new extension PR:

1. **Uses the owning seam's explicit construction or documented registry.**
2. **Passes the `Protocol` check** — `isinstance(instance, Protocol)`
   is True.
3. **Ships a unit test** that exercises the happy path on a toy model
   and one failure mode (bad input, missing method).
4. **Updates exactly one cross-cutting doc** if the extension is
   generally useful, or ships its own doc.
5. **Keeps the top-level API compact.** Pre-1.0 additions still require an
   intentional contract and documentation update.
