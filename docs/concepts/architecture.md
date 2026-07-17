# Architecture

The library is organised as a strict layered DAG. Arrows point from a
dependent layer to the one it depends on; nothing ever points
backwards. The DAG is enforced by
`tests/contract/test_layer_dependencies.py`, which AST-walks every
file in `src/` and fails on any import that violates the order.

```
io ─────────────┐
                ▼
tasks → optim → residuals → kinematics ↴
                              │         dynamics ↴
                              ▼                   ▼
                             data_model ──── spatial ──── lie
                                  │
                                  └── ModelStructure + ModelValues
                                      feed whole-pass Torch or opt-in kernels
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

`kinematics/` and `dynamics/` are siblings: both work on `Model` plus
`Data`, neither imports the other. A whole-pass kernel lives beside its Torch
counterpart in the owning package; it is not a new dependency layer. Forward kinematics and Jacobians
do not need to know about RNEA; RNEA does not need to know about
Jacobian assembly. Splitting them apart is what lets a user build
purely kinematic IK without dragging dynamics code through compile.

The optimization layer currently contains two contracts during the M2
migration. The legacy path keeps `residuals/` as functions of `(model, data,
variables)`, composes them in `optim/CostStack`, and hands a
`LeastSquaresProblem` to an optimizer. The old `better_robot.costs` import path
is a forwarding compatibility package, not a dependency layer. The named-block
path lives in `optim/blocks/`: a `Problem` owns `VarSpec`s, structural residual
items, scalar objective items, and a provider DAG. Providers may reach directly
to lower layers such as kinematics, while user residuals consume only the
read-only context names they declare.

`tasks/` is the topmost user-facing facade. `solve_ik` builds a named-block
`Problem` with a `RobotConfig` variable and provider-backed built-in
residuals. `solve_trajopt` still builds an optimizer-owned `CostStack`, wraps
it in a `LeastSquaresProblem`, and dispatches to the legacy optimizer stack
until M5. Legacy callers invoke an optimizer's `minimize` method directly;
named-block problems use the named-block solvers' `run` methods.

`io/` and `viewer/` sit alongside the main spine, not above it. `io/`
reads from `data_model/` only — the URDF parser does not invoke
kinematics. `viewer/` is at the very top: nothing imports from it.
That sequencing is what allows `import better_robot` to skip viser,
trimesh, mujoco, and yourdfpy (the viewer + parsers are the only
paths that pull them, and they are gated by extras).

## The dependency rule, in code

```
lie → spatial → data_model → (kinematics, dynamics) → residuals → optim → tasks
                                                       ↑                                  │
                                                       └── collision ─────────────────────┘
io → data_model          (io reads nothing from optim or tasks)
viewer → tasks           (topmost; no-one imports from viewer)
```

Stated differently: when you sit in any module under `src/`, you may
look down and across at modules in lower or earlier layers; you may
never look up. The contract test parses each file's imports and
fails the build with the offending file and line number if the rule
breaks.

## What each layer owns

| Layer | Owns | Forbidden imports |
|-------|------|-------------------|
| `lie` | Direct Torch SE3 / SO3 ops, typed `SE3` / `SO3` / `Pose` wrappers | anything above itself |
| `spatial` | `Motion`, `Force`, `Inertia` value types | anything above `lie` |
| `data_model` | `Model`, `Data`, `JointModel`s, structure/value/execution seam | `kinematics` / `dynamics` / above |
| `kinematics` | FK, frame updates, Jacobians, local whole-pass kernels | `dynamics` / `residuals` / above |
| `dynamics` | RNEA / ABA / CRBA / centroidal algorithms and local whole-pass kernels | `residuals` / above |
| `residuals` | Pure residual functions | `optim` / `tasks` / `io` / `viewer` |
| `optim` | Named-block `Problem` evaluation plus legacy `CostStack` / `LeastSquaresProblem`, optimizers, linear solvers, kernels, and damping | `tasks` / `io` / `viewer` |
| `collision` | Geometry, SDF pairs | `tasks` / `io` / `viewer` |
| `io` | Parsers, IR, builders | `tasks` / `viewer` |
| `tasks` | `solve_ik`, `solve_trajopt`, and trajectory types | `viewer` |
| `viewer` | viser bindings | — |

## The package layout

```
src/better_robot/
├── __init__.py                    # small top-level convenience API
├── _typing.py                     # jaxtyping-style shape annotations
│
├── lie/                           # SE3 / SO3 functional + typed wrappers
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
│   └── jacobian_strategy.py       # JacobianStrategy enum
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
│   ├── smoothness.py              # 5-point FD velocity / accel
│   ├── manipulability.py
│   ├── collision.py
│   ├── regularization.py
│   ├── reference_trajectory.py
│   └── contact.py
│
├── optim/
│   ├── cost_stack.py              # legacy flat-residual CostStack
│   ├── problem.py                 # LeastSquaresProblem
│   ├── blocks/                    # named Problem / VarSpec / manifolds / providers
│   ├── state.py                   # SolverState
│   ├── optimizers/                # LM / GN / Adam / LBFGS / MultiStage
│   ├── solvers/                   # dense batched Cholesky / LSTSQ
│   ├── kernels/                   # L2 / Huber / Cauchy / Tukey
│   └── strategies/                # legacy Constant / Adaptive
│
├── costs/                         # forwarding compatibility package
│   └── stack.py                   # re-exports optim.cost_stack identities
│
├── tasks/
│   ├── ik.py                      # solve_ik
│   ├── trajopt.py                 # solve_trajopt
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
    # data_model (5)
    "Model", "Data", "Frame", "Joint", "Body",
    # io (2)
    "load", "ModelBuilder",
    # lie (1)
    "SE3",
    # kinematics (6)
    "forward_kinematics", "update_frame_placements",
    "compute_joint_jacobians", "get_joint_jacobian", "get_frame_jacobian",
    "JacobianStrategy",
    # dynamics (5)
    "rnea", "aba", "crba", "center_of_mass", "compute_centroidal_map",
    # optim (2)
    "CostStack", "LeastSquaresProblem",
    # tasks (3)
    "solve_ik", "solve_trajopt", "Trajectory",
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
from better_robot.kinematics  import ReferenceFrame
from better_robot.optim.state import SolverState
from better_robot.tasks.ik    import IKResult, IKCostConfig, OptimizerConfig

from better_robot.optim import (
    Bounds, Euclidean, SO3Manifold, SE3Manifold, RobotConfig,
    Values, VarSpec, Problem, ResidualItem, ObjectiveItem,
    RobotStateProvider, detach_values,
)
```

The manifold suffixes are part of the contract: `better_robot.SE3` and
`better_robot.lie.SE3` are typed Lie-group pose wrappers, whereas
`better_robot.optim.SE3Manifold` is a retraction/difference policy for a
variable block. The block construction names stay qualified under
`better_robot.optim`; they are not added to the root `better_robot.__all__`.

`tests/contract/test_submodule_public_imports.py` enforces those
paths so a refactor cannot silently move them.

## Extension seams

Growth happens at `Protocol`-shaped seams — every place a user might
want to plug in their own implementation is documented as a
structural type. The complete catalogue (residuals, joints,
optimisers, robust kernels, damping strategies, linear solvers,
collision primitives, render modes, parsers, trajectory
parameterisations, asset resolvers, actuators) lives in
{doc}`/conventions/extension`. Core layers import only the Protocol,
not concrete classes; this keeps the DAG stable as the extension set
grows.

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
- **Two Jacobian boundaries with different jobs.** The legacy task solver calls
  the unified kinematics Jacobian dispatch, which selects an analytic Jacobian
  or the unbatched central-finite-difference fallback. Named-block `Problem`
  evaluation instead assembles per-residual, per-variable tangent blocks and
  uses `torch.func` forward/reverse AD when an analytic block is absent. See
  {doc}`kinematics` and {doc}`solver_stack`.
- **Residuals never reach into solvers.** Legacy built-ins live above
  kinematics and compose through optimizer-owned `CostStack`. Named-block
  residuals are structural consumers of declared context names;
  evaluation-local providers own shared FK or other expensive lower-layer
  work. Both routes preserve the downward dependency rule while they coexist.
  `tasks/` remains the top user-facing facade.
- **`io` and `viewer` siblings of `tasks`, not ancestors.** `load()`
  never constructs a `Task`; it returns a `Model`. The viewer is
  outside the spine because nothing should depend on it.

## Where to look next

The remaining chapters walk down the spine starting from the user
side: {doc}`model_and_data` is what `load` returns, {doc}`kinematics`
is the first thing most users call, and {doc}`tasks` is what most
production code actually uses.
