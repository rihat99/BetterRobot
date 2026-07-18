# Architecture

The library is organised as a strict layered DAG. Arrows point from a
dependent layer to the one it depends on; nothing ever points
backwards. The DAG is enforced by
`tests/contract/test_layer_dependencies.py`, which AST-walks every
file in `src/` and fails on any import that violates the order.

```
tasks → optim → residuals ─┬→ dynamics → kinematics ─┐
                           └───────────→ kinematics  │
io ───────────────────────────────────→ data_model ←┘
                                              ↓
                                           spatial → lie

ModelStructure + ModelValues feed whole-pass Torch or opt-in kernels.
```

## Why the layers fall out this way

The starting point is the math: everything above the `lie/` layer
needs to manipulate SE(3) and SO(3) elements. So `lie/` sits at the
bottom and provides direct Torch tensor operations. `spatial/` builds on
`lie/` to add the 6D value types
(`Motion`, `Force`, `Inertia`) that dynamics needs.

Above the math, the `data_model/` layer holds the Pinocchio-style
`Model` (frozen topology) and `Data` (mutable workspace). It depends
on `spatial/` because body inertias live there, and on `lie/` because
joint placements are SE(3) elements. It depends on nothing higher.
The same layer owns the whole-pass seam: `ModelStructure` mirrors static
topology as Python tuples and device tables, while `ModelValues` carries the
differentiable tensor pytree. Torch raw passes and eligible opt-in kernels
consume that shared contract.

`kinematics/` and `dynamics/` have the same contract rank and both work on
`Model` plus `Data`. The current dependency is one-way: dynamics reuses raw FK
and validation helpers from kinematics, while kinematics does not import
dynamics. A whole-pass kernel lives beside its Torch counterpart in the owning
package; it is not a new dependency layer. Purely kinematic IK therefore does
not pull dynamics into its import or compile path.

The optimization layer has one construction contract. A named-block `Problem`
owns `VarSpec`s, least-squares residual items, and evaluation-local providers.
Providers may reach directly to lower layers such as kinematics, while user
residuals consume a read-only context. Residual-owned `TemporalPattern` values
remain below `optim`; the optimizer consumes them to build block-banded normal
systems without creating a reverse dependency.

`tasks/` is the topmost user-facing facade. `solve_ik` builds a named-block
`Problem` with a `RobotConfig` variable and provider-backed built-in
residuals. `solve_trajopt` adapts an explicit sequence of `ResidualItem`
values into one time-annotated `RobotConfig` block and uses route-aware LM.
Direct problem callers use the same named-block solvers' `run` methods.

`io/` and `viewer/` sit alongside the main spine, not above it. `io/`
reads from `data_model/` only — the URDF parser does not invoke
kinematics. `viewer/` is at the very top: nothing imports from it.
That sequencing is what allows `import better_robot` to skip importing viser,
trimesh, mujoco, and yourdfpy. Viewer, direct mesh, and MJCF support are gated
by extras; the core URDF dependency is still imported lazily at the parser
boundary.

## The dependency rule, in code

```
tasks → optim → residuals → dynamics → kinematics → data_model → spatial → lie
                  ├────────────────→ kinematics
                  └→ collision ────────────────────→ data_model
io → data_model          (io reads nothing from optim or tasks)
viewer → tasks           (topmost; no-one imports from viewer)
```

Stated differently: when you sit in any module under `src/`, you may look down
to a lower-ranked layer. Equal-rank cross-package imports are also allowed;
that is how dynamics reuses kinematics today. You may never import a
higher-ranked layer. The contract test parses each file's imports and fails
with the offending file and line number if that rule breaks.

## What each layer owns

| Layer | Owns | Forbidden imports |
|-------|------|-------------------|
| `lie` | Direct Torch SE3 / SO3 ops, typed `SE3` / `SO3` / `Pose` wrappers | anything above itself |
| `spatial` | `Motion`, `Force`, `Inertia` value types | anything above `lie` |
| `data_model` | `Model`, `Data`, `JointModel`s, structure/value/execution seam | `kinematics` / `dynamics` / above |
| `kinematics` | FK, frame updates, Jacobians, local whole-pass kernels | `dynamics` / `residuals` / above |
| `dynamics` | RNEA / ABA / CRBA / centroidal algorithms and local whole-pass kernels | `residuals` / above |
| `residuals` | Pure residual functions | `optim` / `tasks` / `io` / `viewer` |
| `optim` | Named-block `Problem` evaluation, LM/GN, a `torch.optim` adapter, linear solvers, robust kernels, and temporal structure | `tasks` / `io` / `viewer` |
| `collision` | Reserved primitive, pair-dispatch, and robot-decomposition surfaces (computation is stubbed) | `tasks` / `io` / `viewer` |
| `io` | Parsers, IR, builders | `tasks` / `viewer` |
| `tasks` | `solve_ik`, `solve_trajopt`, `solve_contact_forces`, and trajectory types | `viewer` |
| `viewer` | viser bindings | — |

## The package layout

```
src/better_robot/
├── __init__.py                    # small top-level convenience API
├── _typing.py                     # jaxtyping-style shape annotations
│
├── lie/                           # SE3 / SO3 functional + typed wrappers
│   ├── alignment.py                 # weighted batched Umeyama fit
│   ├── se3.py
│   ├── so3.py
│   ├── tangents.py                # Jr / Jl, hat / vee, BCH helpers
│   ├── types.py                   # SE3 / SO3 / Pose dataclasses (around tensors)
│   └── _impl.py                   # direct pure-Torch implementation
│
├── spatial/                       # 6D value types
│   ├── motion.py
│   ├── force.py
│   ├── inertia.py
│   ├── symmetric3.py
│   └── ops.py                     # ad, Ad, cross, act
│
├── data_model/                    # Model / Data / Joints / Bodies / Frames
│   ├── model.py
│   ├── data.py
│   ├── model_structure.py         # immutable Python + device topology mirrors
│   ├── model_values.py            # differentiable tensor pytree
│   ├── execution_batch.py         # flat-E broadcast ABI for whole-pass kernels
│   ├── joint_dispatch.py          # shared built-in joint-kind dispatch
│   ├── joint.py
│   ├── joint_models/              # one file per joint family
│   ├── frame.py
│   ├── body.py
│   └── topology.py
│
├── kinematics/
│   ├── forward.py                 # forward_kinematics, update_frame_placements
│   ├── jacobian.py                # compute_joint_jacobians, get_joint/frame_jacobian
│
├── dynamics/
│   ├── rnea.py
│   ├── aba.py
│   ├── crba.py
│   ├── centroidal.py
│   ├── derivatives.py
│   ├── state_manifold.py
│   └── integrators.py
│
├── residuals/                     # residual classes composed explicitly
│   ├── pose.py                    # PoseResidual / PositionResidual / OrientationResidual
│   ├── limits.py
│   ├── smoothness.py              # temporal velocity / acceleration
│   ├── structure.py               # optimizer-independent TemporalPattern
│   ├── regularization.py
│   └── contact.py
│
├── optim/
│   ├── manifolds.py               # manifolds, robot configurations, bounds
│   ├── variables.py               # Values / VarSpec
│   ├── problem.py                 # builder, evaluation context, Jacobians
│   ├── providers.py               # lazy provider memo and robot state
│   ├── lm.py                      # LM/GN lifecycle and route decisions
│   ├── first_order.py             # thin torch.optim adapter
│   ├── temporal.py                # block-banded analysis and assembly
│   ├── implicit.py                # guarded implicit differentiation
│   ├── solvers.py                 # Cholesky / BandedCholesky
│   └── kernels.py                 # L2 / Huber / Cauchy / Tukey / GemanMcClure
│
├── tasks/
│   ├── ik.py                      # solve_ik
│   ├── trajopt.py                 # solve_trajopt
│   ├── contact_forces.py          # solve_contact_forces
│   ├── smoothing.py               # quaternion / SE3 kernel smoothing
│   ├── trajectory.py              # Trajectory dataclass
│   └── parameterization.py        # Knot / BSpline
│
├── collision/
│   ├── geometry.py
│   ├── pairs.py
│   ├── robot_collision.py
│   └── closest_pts.py
│
├── io/
│   ├── ir.py                      # internal IRModel dataclasses
│   ├── build_model.py             # IR → Model factory
│   ├── parsers/                   # urdf, mjcf, programmatic
│   ├── builders/                  # smpl_like example
│   └── assets.py                  # AssetResolver Protocol
│
├── viewer/
│   ├── visualizer.py
│   ├── scene.py
│   ├── trajectory_player.py
│   ├── render_modes/
│   ├── overlays/
│   └── renderers/
```

## The public API contract

The top-level `better_robot.__init__` exports a deliberately small set of
common entry points:

```python
__all__ = [
    # data_model (7)
    "Model", "ModelStructure", "ModelValues", "Data", "Frame", "Joint", "Body",
    # io (2)
    "load", "ModelBuilder",
    # lie (1)
    "SE3",
    # kinematics (5)
    "forward_kinematics", "update_frame_placements",
    "compute_joint_jacobians", "get_joint_jacobian", "get_frame_jacobian",
    # dynamics (5)
    "rnea", "aba", "crba", "center_of_mass", "compute_centroidal_map",
    # tasks (4)
    "solve_ik", "solve_trajopt", "solve_contact_forces", "Trajectory",
]
```

The set is not frozen before 1.0. The contract test pins a required core,
checks every listed symbol resolves, and rejects duplicate entries without
turning the current symbol count into an API promise. Promotion remains
evidence-driven: a symbol earns top-level status when examples show that the
qualified path is unnecessary friction.

Submodule-only public symbols are reachable from their qualified
import path and covered by the same contract suite, even though they
are not in the top-level `__all__`:

```python
from better_robot.lie         import SO3, Pose
from better_robot.spatial     import Motion, Force, Inertia, Symmetric3
from better_robot.kinematics  import (
    forward_kinematics_raw, frame_placements_raw, joint_jacobians_raw,
)
from better_robot.dynamics    import rnea_raw, aba_raw, crba_raw, ccrba_raw
from better_robot.tasks.ik    import IKResult, IKCostConfig, OptimizerConfig

from better_robot.optim import (
    Bounds, Euclidean, SO3Manifold, SE3Manifold, RobotConfig,
    Values, VarSpec, Problem, ResidualItem, TemporalPattern,
    BlockBandedMatrix, LinearizationDecision, RobotStateProvider,
    FirstOrderResult, run_first_order, LevenbergMarquardt, GaussNewton,
    detach_values,
)
```

The `better_robot.spatial` package is also reachable as `br.spatial` after
`import better_robot as br`; like the other qualified namespaces it is not a
top-level wildcard export.

The manifold suffixes are part of the contract: `better_robot.SE3` and
`better_robot.lie.SE3` are typed Lie-group pose wrappers, whereas
`better_robot.optim.SE3Manifold` is a retraction/difference policy for a
variable block. The block construction names stay qualified under
`better_robot.optim`; they are not added to the root `better_robot.__all__`.

`tests/contract/test_submodule_public_imports.py` enforces those
paths so a refactor cannot silently move them.

## Extension seams

Growth uses several explicit seams. Residuals, joints, solver lifecycles, linear
solvers, render modes, trajectory parameterizations, and asset resolvers have
structural or class contracts; parser discovery is a suffix registry plus
loader function. Collision and actuator surfaces are reserved sketches rather
than usable third-party seams. The exact live and deferred catalogue lives in
{doc}`/conventions/extension`.

Whole-pass compute lanes are the deliberate exception: they are internal
performance integrations selected by an explicit branch in the owning pass,
not a public plugin Protocol or process-wide registry.

## Why this shape works

- **`lie` + `spatial` split.** `lie/` is the algebraic machinery;
  `spatial/` is the 6D value-type layer that dynamics consumes.
  Mirrors Pinocchio's separation of `liegroups.hpp` from
  `spatial/`.
- **`data_model/joint_models/` one-file-per-joint.** Adding a new
  joint kind is an isolated change. See
  {doc}`joints_bodies_frames`.
- **One Jacobian boundary.** Named-block `Problem` evaluation assembles
  per-residual, per-variable tangent blocks and uses `torch.func`
  forward/reverse AD when an analytic block is absent. See {doc}`kinematics`
  and {doc}`solver_stack`.
- **Residuals never reach into solvers.** Residuals are structural consumers
  of declared context names; evaluation-local providers own shared FK or other
  expensive lower-layer work. This preserves the downward dependency rule.
  `tasks/` remains the top user-facing facade.
- **`io` and `viewer` siblings of `tasks`, not ancestors.** `load()`
  never constructs a `Task`; it returns a `Model`. The viewer is
  outside the spine because nothing should depend on it.

## Where to look next

The remaining chapters walk down the spine starting from the user
side: {doc}`model_and_data` is what `load` returns, {doc}`kinematics`
is the first thing most users call, and {doc}`tasks` is what most
production code actually uses.
