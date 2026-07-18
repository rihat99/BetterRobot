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
**seams** — structural interfaces and explicit construction points where user
or contributor code joins without a core switch statement. Most public seams
use `typing.Protocol` (no inheritance, MRO, or metaclass requirement). Parser
suffix dispatch is the one shipped discovery seam with a public registration
function. Solvers, kernels, residuals, render modes, and asset resolvers are
passed as objects. The internal whole-pass compute seam is narrower still: it
uses an explicit branch in the owning pass, not a public plugin registry.

This document is the canonical list of seams, their contracts, and
recipes. If you cannot find an extension here, do not assume a registry or
decorator exists. Implement the matching structural contract and pass the
object explicitly, or treat the surface as contributor-only when this guide
labels it reserved.

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
  │ Parser           │     │ Residual        │     │ Solver          │
  │ (register_parser │     │ (explicit       │     │ (explicit       │
  │  suffix function)│     │  construction)  │     │  .run)          │
  └──────────────────┘     └─────────────────┘     └─────────────────┘
           │                        │                       │
  ┌────────▼─────────┐     ┌────────▼────────┐     ┌────────▼────────┐
  │ Joint model      │     │ Robust kernel   │     │ Linear solver   │
  │ (JointModel      │     │ (Protocol:      │     │ (Protocol:      │
  │  Protocol)       │     │  .rho/.weight)  │     │  .solve)        │
  └──────────────────┘     └─────────────────┘     └─────────────────┘
           │                        │                       │
  ┌────────▼─────────┐     ┌────────▼────────┐     ┌────────▼────────┐
  │ Collision        │     │ Render mode     │     │ Whole-pass lane │
  │ pair dispatch    │     │ (RenderMode +   │     │ (internal,      │
  │ (reserved stub)  │     │  Scene.add_mode)│     │ explicit branch)│
  └──────────────────┘     └─────────────────┘     └─────────────────┘
```

Public object seams use structural typing rather than inheritance. Only a
custom parser calls a registration function; the other examples below are
constructed and passed directly. Whole-pass compute work is a contributor
integration and follows §10 instead.

## 1 · Add a residual

**Use when:** you need a new objective (reachability, manipulability
variant, user-defined cost).

:::{important} Recommended path for new integrations
For new named-block residuals and providers, follow
{doc}`/guides/custom_residuals`. It covers declared context reads, static
residual shapes, provider lifetime, robust groups, analytic blocks, and
per-element failure signaling.

The short recipe below uses that same named-block surface. `solve_ik` and
`solve_trajopt` also assemble named-block problems internally.
:::

```python
# my_package/residuals/neutral.py
import torch

class NeutralResidual:
    """Penalise tangent displacement from the model's neutral pose."""

    name = "neutral"
    reads = ("q",)

    def __init__(self, model):
        self.model = model
        self.dim = model.nv

    def __call__(self, ctx) -> torch.Tensor:
        q = ctx["q"]
        target = self.model.q_neutral.to(
            device=q.device,
            dtype=q.dtype,
        )
        return self.model.difference(target, q)
```

Then:

```python
from better_robot.optim import ResidualItem

residual = ResidualItem("neutral", NeutralResidual(model), weight=0.1)
```

Contract (see {doc}`/guides/custom_residuals`):

| Member | Type | Required |
|--------|------|----------|
| `name` | stable class/instance name | Yes |
| `reads` | tuple of context entry names | Yes |
| `dim` | `int` property | Yes |
| `__call__(ctx) -> Tensor` | residual `(B..., dim)` | Yes |
| `jacobian_blocks(ctx) -> dict[str, Tensor]` | complete analytic blocks for selected variables | No |

Residuals are instantiated and added explicitly. BetterRobot does not keep a
process-wide residual registry. Put the residual multiplier and optional
robust kernel on `ResidualItem`; `Problem` applies them consistently during
residual, objective, gradient, and solver linearization evaluation.

## 2 · Add a joint type

**Use when:** you need a coordinate class that does not fit the
built-in taxonomy (coupled joints, splined motion, helical with
non-constant pitch, etc.).

Built-in joint types live under `data_model/joint_models/`; each structurally
implements `JointModel`. The following is an **interface sketch**, not a
complete coupled-joint implementation—the transform, subspace, and manifold
math are specific to the joint:

```python
# my_package/joint_models/coupled.py
import torch

class JointCoupled:
    kind = "coupled"
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

    def joint_velocity(self, q, v) -> torch.Tensor:
        return (self.joint_motion_subspace(q) @ v.unsqueeze(-1)).squeeze(-1)

    def integrate(self, q, v) -> torch.Tensor: ...
    def difference(self, q0, q1) -> torch.Tensor: ...
    def random_configuration(self, generator, lower, upper) -> torch.Tensor: ...
    def neutral(self) -> torch.Tensor: ...

    # Optional dynamics hooks may be added here. When they are absent, the
    # dynamics dispatch helpers supply zero bias/subspace derivatives.
```

There is no joint registry and `load` has no `extensions=` argument. The
shipped external wiring point is the programmatic builder. It preserves the
exact `JointModel` instance in the built model:

```python
import torch

from better_robot.io import ModelBuilder, build_model

builder = ModelBuilder("coupled_finger")
base = builder.add_body("base")
finger = builder.add_body("finger")
builder.add_joint(
    "finger_1",
    kind=JointCoupled(
        axis=torch.tensor([0.0, 0.0, 1.0]),
        coupling=torch.tensor([1.0]),
    ),
    parent=base,
    child=finger,
)
model = build_model(builder.finalize())
```

Contract (`JointModel` Protocol):

| Member | Signature | Semantics |
|--------|-----------|-----------|
| `kind` | stable `str` | Joint discriminator preserved in the model |
| `nq` | `int` | Config-space dim for this joint |
| `nv` | `int` | Tangent-space dim for this joint |
| `axis` | `Tensor` or `None` | Unit axis for axis-based joints; otherwise `None` |
| `joint_transform(q)` | `(B..., nq) -> (B..., 7)` | Joint's own SE(3) motion |
| `joint_motion_subspace(q)` | `(B..., nq) -> (B..., 6, nv)` | Motion subspace `S(q)` |
| `joint_velocity(q, v)` | `(B..., nq), (B..., nv) -> (B..., 6)` | Joint-frame spatial velocity |
| `integrate(q, v)` | `(B..., nq), (B..., nv) -> (B..., nq)` | Manifold retraction |
| `difference(q0, q1)` | `(B..., nq), (B..., nq) -> (B..., nv)` | Manifold log |
| `random_configuration(generator, lo, hi)` | → `(nq,)` | Sample respecting limits |
| `neutral()` | → `(nq,)` | Identity/default configuration |

The FK loop calls `joint_transform`; Jacobian computation calls
`joint_motion_subspace`. The dynamics hooks are optional and default to zero.
Runtime construction accepts a stable custom string `kind`, although the
exported `JointKind` type alias enumerates built-ins and is not an open typing
registry.

## 3 · Add an optimiser

**Use when:** you are contributing a solver for the named-block
`Problem`/`Values` lifecycle. There is no public generic solver protocol or
solver registry. The concrete `Adam`, `GaussNewton`, and
`LevenbergMarquardt` implementations are the reference contracts.

A solver accepted by `Phase` is a dataclass with a replaceable `max_iter`
field and this structural lifecycle:

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class CustomSolver:
    max_iter: int = 50

    def init_state(self, values, problem): ...
    def update(self, values, state, problem): ...
    def run(self, values, problem, state=None): ...
```

`init_state` validates the solve boundary, `update` returns a fresh
`(Values, state)` pair for one fixed-shape step, and `run` returns the final
pair. State and status types are solver-specific. Implementing the lifecycle
correctly also requires batched per-element termination, manifold retraction,
bounds, failure isolation, and graph-lifetime tests; use the shipped solvers
as the implementation reference. Custom solvers are passed to `Phase` or
called directly and are never discovered by `solve_ik` or `solve_trajopt`.

## 4 · Configure LM damping

```python
from better_robot.optim import LevenbergMarquardt

fixed = LevenbergMarquardt(
    damping_parameter=1e-4,
    fixed_damping=True,
)
adaptive = LevenbergMarquardt(
    damping_parameter=1e-4,
    mu_min=1e-12,
    mu_max=1e8,
    increase_factor_max=1e6,
)
```

LM owns its damping policy through frozen solver fields; it does not accept a
separate policy object. `fixed_damping=True` holds damping fixed, while the
default adapts it after accepted and rejected steps. Task
`OptimizerConfig.damping` exposes the corresponding shipped string choices.

## 5 · Add a linear solver

**Use when:** the problem has sparsity the default dense Cholesky
cannot exploit (large trajopt), or you want KKT / iterative methods.

```python
# my_package/optim/solvers/dense_solve.py
import torch

class DenseSolve:
    supported_systems = frozenset({"dense"})

    def solve(self, A, b, ridge=None):
        matrix = A
        if ridge is not None:
            ridge = torch.as_tensor(ridge, dtype=A.dtype, device=A.device)
            identity = torch.eye(A.shape[-1], dtype=A.dtype, device=A.device)
            matrix = A + ridge[..., None, None] * identity
        return torch.linalg.solve(matrix, b.to(A.dtype)).to(b.dtype)
```

Contract: one method, `solve(A, b: (B...,n), ridge: (B...,) | scalar | None)`
returning `(B...,n)`, plus a static `supported_systems` set when the solver is
not dense-only. The recognized system strings are `"dense"`, `"banded"`, and
`"operator"`; `A` is respectively a tensor, `BlockBandedMatrix`, or
`NormalOperator`. The shipped implementations are dense `Cholesky`/`LSTSQ`,
`BandedCholesky`, and `NormalCG`. If `supported_systems` is absent, routing
assumes dense-only. Custom solvers keep the same `b`/`ridge` semantics and
should implement `solve_with_info` only when they return the full
`LinearSolveResult` diagnostics contract.

## 6 · Add a robust kernel

```python
# my_package/optim/kernels/soft_l1.py
import torch

class SoftL1:
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

## 7 · Collision pair extensions (reserved stub)

This is not a working external extension seam yet. Geometry value classes and
`register_pair(type_a, type_b)` exist, but the public `distance(a, b)`
dispatcher still raises `NotImplementedError` and does not read the pair
table. Registration stores only the exact forward `(type_a, type_b)` key; it
does not add a reversed entry or infer symmetry.

Until the dispatcher and built-in pair kernels ship together, keep a custom
signed-distance function in consumer code and call it directly. A future
contributor implementation must define the output/batching contract, wire the
table into `distance`, and register each supported order explicitly. Do not
publish a third-party `@register_pair` recipe as functional today.

## 8 · Add a render mode

```python
# my_package/viewer/my_mode.py
class JointFrameMode:
    name = "Joint frames"
    description = "One coordinate frame at every joint"

    def __init__(self):
        self.context = None
        self.nodes = []

    @classmethod
    def is_available(cls, model, data) -> bool:
        return data.joint_pose_world is not None

    def attach(self, context, model, data) -> None:
        self.context = context
        for joint_id in range(model.njoints):
            node = f"{context.namespace}/joint_{joint_id}"
            context.backend.add_frame(node, axes_length=0.08)
            self.nodes.append(node)
        self.update(data)

    def update(self, data) -> None:
        assert self.context is not None
        poses = data.joint_pose_world
        assert poses is not None and poses.ndim == 2  # unbatched example
        for node, pose in zip(self.nodes, poses, strict=True):
            self.context.backend.set_transform(node, pose)

    def set_visible(self, visible: bool) -> None:
        assert self.context is not None
        for node in self.nodes:
            self.context.backend.set_visible(node, visible)

    def detach(self) -> None:
        assert self.context is not None
        for node in self.nodes:
            self.context.backend.remove(node)
        self.nodes.clear()
        self.context = None
```

Attach an instance explicitly to a scene:

```python
viewer.scene().add_mode(JointFrameMode())
```

The lifecycle signature is
`is_available(model, data)`, `attach(context, model, data)`, `update(data)`,
`set_visible(visible)`, and `detach()`. `description` is also required by the
runtime-checkable Protocol. `better_robot.viewer` does not export a public
mode registry, and `Scene.default` does not discover third-party registry
entries; explicit `Scene.add_mode` is the supported seam.

## 9 · Add a parser format

**Use when:** robot description in a format other than URDF / MJCF
(SDF, Drake YAML, custom XML).

The format conversion is necessarily format-specific, so this is a **signature
sketch**, not a complete SDF parser:

```python
# my_package/io/parse_sdf.py
from better_robot.io import IRModel, register_parser

def parse_sdf(source) -> IRModel:
    ir = IRModel(...)
    ...
    return ir

register_parser("sdf", parse_sdf)
```

`register_parser` is a two-argument function, not a decorator. The key omits
the leading dot because `load` strips and lowercases a path suffix before
lookup. The registered callable receives only `source` and returns an
`IRModel`; `load` then calls `build_model`. Registration is process-local,
silently replaces an existing key, and takes effect only after the extension
module is imported. See {doc}`/concepts/parsers_and_ir`.

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
4. expose a tested Torch-autograd adjoint strategy—either a validated kernel
   or an explicitly correct oracle recomputation—without leaking optional
   runtime arrays;
5. preserve the flat-`E` `ExecutionBatch` mapping when inputs broadcast; and
6. keep optional imports local to the owning pass.

For Warp, conversion and custom-autograd plumbing belong in that local
integration module. Capture validation is caller-owned; the public solver
loop remains eager. New capture integration must keep forward and backward in
one stable lifecycle and must not invent a public capture decorator before an
end-to-end solver path is certified.

## 11 · Discovery and explicit construction

Parser suffixes are the only working public discovery registry. Use
`register_parser(suffix, fn)`; do not mutate the private `_PARSERS` dictionary.
Residuals, solvers, kernels, linear solvers, render modes, renderer backends,
and asset resolvers are passed explicitly. LM damping is configured on the
solver itself.

There are internal `MODE_REGISTRY`, `RENDERER_REGISTRY`, and collision pair
tables in the source tree, but they are not general public plugin seams:
`Scene.default` chooses its built-ins directly, the renderer table has no
public registration flow, and collision `distance` is still a stub. Their
presence is not evidence that an import-time third-party registration recipe
works.

**Rule:** do not add entry-point auto-discovery or mutate an internal table in
consumer code. Whole-pass compute lanes also never use a registry.

## 12 · Trajectory parameterisations

**Use when:** you are implementing a stand-alone numerical mapping between
sampled values and a lower-dimensional basis (for example a cosine basis).
The current Protocol owns only seed projection and expansion
`q_traj_seed -> z -> q_traj`; custom robot task
integration additionally needs a separately reviewed manifold retraction,
local Jacobian, and feasible-bound contract:

The class below is an **interface sketch**; its three method bodies are the
format-specific numerical implementation:

```python
# my_package/parameterizations/log_basis.py
import torch

class LogBasisTrajectory:
    def init(self, q_traj_seed: torch.Tensor) -> torch.Tensor: ...
    def expand(self, z: torch.Tensor, *, T: int, nq: int) -> torch.Tensor: ...
    def tangent_dim_per_step(self) -> int: ...
```

The shipped numerical implementations are `KnotTrajectory` (identity) and
`BSplineTrajectory` (Euclidean cubic basis). Robot `solve_trajopt` currently
accepts only `KnotTrajectory`: the richer custom-parameterisation contract for
manifold-safe interpolation/retraction and bounds is not implied by this
Protocol and remains deferred. See {doc}`/concepts/tasks`.

## 13 · Asset resolvers

**Use when:** you have a custom mesh-storage scheme (ROS package map, S3
bucket, embedded asset bundle, in-process cache).

```python
from pathlib import Path

class MappingResolver:
    def __init__(self, assets: dict[str, Path]) -> None:
        self.assets = dict(assets)

    def resolve(self, uri: str) -> Path:
        try:
            path = self.assets[uri].resolve()
        except KeyError as exc:
            raise FileNotFoundError(uri) from exc
        if not path.exists():
            raise FileNotFoundError(path)
        return path

from better_robot.io import build_model
from better_robot.io.parsers import parse_urdf

resolver = MappingResolver({"asset://arm.stl": Path("meshes/arm.stl")})
ir = parse_urdf("robot.urdf", resolver=resolver)
model = build_model(ir)
```

The exact `AssetResolver` Protocol has one method:
`resolve(uri: str) -> Path`. Missing assets raise `FileNotFoundError`; there is
no `exists` method or per-call `base_path` keyword. `better_robot.load` does
not currently accept `resolver=`, so callers that need a parse-time resolver
call `parse_urdf` or `parse_mjcf` and then `build_model` as above. The resolver
is preserved in `model.meta` for mesh rendering.

Concrete resolvers shipped in core: `FilesystemResolver`,
`PackageResolver`, `CompositeResolver`, `CachedDownloadResolver`.

## 14 · Actuator design (not a shipped seam)

BetterRobot does not currently ship a ``Muscle`` Protocol, actuator registry,
or ``human`` extra. The sketch below records how a future or external
non-rigid actuator could contribute to joint torque; it is not importable API.

A `Muscle` Protocol composes with `dynamics/`:

```python
class Muscle(Protocol):
    name: str
    nu: int
    def compute_force(self, q, v, u) -> Tensor:    # (B..., 1)
    def moment_arm(self, model, q) -> Tensor:      # (B..., nv)
```

There is no actuator dispatch hook in `rnea` or `aba`. External code can form
generalized actuator torque as
`sum(force(q, v, u) * moment_arm(model, q))` and add it to the `tau` passed to
`aba`. `rnea` instead returns the generalized torque required for prescribed
motion, so an external actuator model can compare or subtract its contribution
afterward. The existing `fext` argument is specifically a per-body local
spatial wrench, not a generic actuator callback.

No OpenSim/DeGrooteFregly implementation or sibling
``better_robot_human`` package is provided by this repository. Core
BetterRobot does not import ``chumpy``, SMPL, or OpenSim.

## 15 · What is **not** pluggable (deliberately)

| Thing | Why not |
|-------|---------|
| `Model` / `Data` schemas | Would break every algorithm. Extend via `Model.meta` dict or a wrapper class. |
| The SE(3) representation `[tx,ty,tz,qx,qy,qz,qw]` | Every algorithm depends on it. Change requires a major version. |
| `Problem` / `Values` structure | Freeze the solver-facing named-block contract so solvers are interchangeable. |
| Layer DAG | If you want to import from a higher layer, refactor instead. |
| `IRModel` shape | Internal parser/build boundary; re-parse assets after upgrades. |

## 16 · Pre-merge checklist for an extension

Every new extension PR:

1. **Uses the owning seam's explicit construction or documented registry.**
2. **Passes the `Protocol` check when that seam exposes one** — verify
   `isinstance(instance, Protocol)` for the runtime-checkable object seams;
   parser callables and reserved contributor sketches are not Protocols.
3. **Ships a unit test** that exercises the happy path on a toy model
   and one failure mode (bad input, missing method).
4. **Updates exactly one cross-cutting doc** if the extension is
   generally useful, or ships its own doc.
5. **Keeps the top-level API compact.** Pre-1.0 additions still require an
   intentional contract and documentation update.
