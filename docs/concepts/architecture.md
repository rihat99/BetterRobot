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
                             data_model ──── spatial ──── lie ──── backends
                                                                    │
                                                                    ▼
                                                               (torch_native | warp)
```

## Why the layers fall out this way

The starting point is the math: everything above the `lie/` layer
needs to manipulate SE(3) and SO(3) elements. So `lie/` sits at the
bottom, with `backends/` below it providing the actual tensor
kernels. `spatial/` builds on `lie/` to add the 6D value types
(`Motion`, `Force`, `Inertia`) that dynamics needs.

Above the math, the `data_model/` layer holds the Pinocchio-style
`Model` (frozen topology) and `Data` (mutable workspace). It depends
on `spatial/` because body inertias live there, and on `lie/` because
joint placements are SE(3) elements. It depends on nothing higher.

`kinematics/` and `dynamics/` are siblings: both work on `Model` plus
`Data`, neither imports the other. Forward kinematics and Jacobians
do not need to know about RNEA; RNEA does not need to know about
Jacobian assembly. Splitting them apart is what lets a user build
purely kinematic IK without dragging dynamics code through compile.

The optimisation stack stacks above. `residuals/` are pure functions
of `(model, data, variables)` returning a residual vector; they
depend on `kinematics/` (to compose pose-error Jacobians, to read
`frame_pose_world`), and optionally on `dynamics/` (residuals like
`min_torque` reach down to `rnea`). `costs/` is one layer up — it
composes residuals into a weighted concatenation. `optim/` depends on
`costs/` (the residual / Jacobian interface) but knows nothing about
specific residuals; that is what makes the LM solver work for IK and
for trajopt with the same code.

`tasks/` is the topmost user-facing facade. `solve_ik` and
`solve_trajopt` build a `CostStack`, wrap it in a
`LeastSquaresProblem`, hand it to an `Optimizer`, and return a clean
result type.

`io/` and `viewer/` sit alongside the main spine, not above it. `io/`
reads from `data_model/` only — the URDF parser does not invoke
kinematics. `viewer/` is at the very top: nothing imports from it.
That sequencing is what allows `import better_robot` to skip viser,
trimesh, mujoco, and yourdfpy (the viewer + parsers are the only
paths that pull them, and they are gated by extras).

## The dependency rule, in code

```
backends → lie → spatial → data_model → (kinematics, dynamics) → residuals → costs → optim → tasks
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
| `backends` | Backend Protocol; per-backend kernel implementations | anything above |
| `lie` | SE3 / SO3 group ops, typed `SE3` / `SO3` / `Pose` wrappers | anything above `backends` |
| `spatial` | `Motion`, `Force`, `Inertia` value types | anything above `lie` |
| `data_model` | `Model`, `Data`, `JointModel`s | `kinematics` / `dynamics` / above |
| `kinematics` | FK, frame updates, Jacobians | `dynamics` / `residuals` / above |
| `dynamics` | RNEA / ABA / CRBA / centroidal / action models | `residuals` / above |
| `residuals` | Pure residual functions | `costs` / `optim` / `tasks` / `io` / `viewer` |
| `costs` | `CostStack` | `optim` / `tasks` / `io` / `viewer` |
| `optim` | `LeastSquaresProblem`, optimisers, linear solvers, kernels, damping | `tasks` / `io` / `viewer` |
| `collision` | Geometry, SDF pairs | `tasks` / `io` / `viewer` |
| `io` | Parsers, IR, builders | `tasks` / `viewer` |
| `tasks` | `solve_ik`, `solve_trajopt`, `retarget` facades | `viewer` |
| `viewer` | viser bindings | — |

## The package layout

```
src/better_robot/
├── __init__.py                    # 26 public symbols (frozen)
├── _typing.py                     # jaxtyping-style shape annotations
│
├── backends/
│   ├── __init__.py                # default_backend(), set_backend(), get_backend()
│   ├── protocol.py                # Backend / LieOps / KinematicsOps / DynamicsOps
│   ├── torch_native/              # default backend
│   └── warp/                      # experimental
│
├── lie/                           # SE3 / SO3 functional + typed wrappers
│   ├── se3.py
│   ├── so3.py
│   ├── tangents.py                # Jr / Jl, hat / vee, BCH helpers
│   ├── types.py                   # SE3 / SO3 / Pose dataclasses (around tensors)
│   └── _torch_native_backend.py   # the kernels routed by backends/
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
│   ├── joint.py
│   ├── joint_models/              # one file per joint family
│   ├── frame.py
│   ├── body.py
│   ├── topology.py
│   └── indexing.py
│
├── kinematics/
│   ├── forward.py                 # forward_kinematics, update_frame_placements
│   ├── jacobian.py                # compute_joint_jacobians, get_joint/frame_jacobian
│   ├── jacobian_strategy.py       # JacobianStrategy enum
│   └── chain.py                   # subtree / chain helpers
│
├── dynamics/
│   ├── rnea.py
│   ├── aba.py
│   ├── crba.py
│   ├── centroidal.py
│   ├── derivatives.py
│   ├── action/                    # Crocoddyl-style 3-layer
│   ├── state_manifold.py
│   └── integrators.py
│
├── residuals/                     # registry + 17 residual classes
│   ├── pose.py                    # PoseResidual / PositionResidual / OrientationResidual
│   ├── limits.py
│   ├── smoothness.py              # 5-point FD velocity / accel
│   ├── manipulability.py
│   ├── collision.py
│   ├── regularization.py
│   ├── reference_trajectory.py
│   ├── contact.py
│   └── registry.py                # @register_residual
│
├── costs/
│   ├── stack.py                   # CostStack
│   └── factory.py
│
├── optim/
│   ├── problem.py                 # LeastSquaresProblem
│   ├── state.py                   # SolverState
│   ├── optimizers/                # LM / GN / Adam / LBFGS / MultiStage
│   ├── linear_solvers/            # Cholesky / LSTSQ / CG / SparseCholesky
│   ├── kernels/                   # L2 / Huber / Cauchy / Tukey
│   ├── damping/                   # Constant / Adaptive / TrustRegion
│   └── jacobian_spec.py           # ResidualSpec
│
├── tasks/
│   ├── ik.py                      # solve_ik
│   ├── trajopt.py                 # solve_trajopt
│   ├── retarget.py
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
│   ├── ir.py                      # IRModel + schema_version
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
│
└── utils/
    ├── batching.py
    ├── broadcasting.py
    ├── logging.py
    └── testing.py
```

## The public API contract — 26 symbols

The top-level `better_robot.__init__` exports exactly **26 symbols**:

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
    # residuals (1)
    "register_residual",
    # costs (1)
    "CostStack",
    # optim (1)
    "LeastSquaresProblem",
    # tasks (4)
    "solve_ik", "solve_trajopt", "retarget", "Trajectory",
]
```

The set is **frozen** under
`tests/contract/test_public_api.py::EXPECTED`. Adding or removing a
symbol requires updating `EXPECTED` in the same PR — the audit is the
diff, not a magic number. Promotion is evidence-driven: a symbol
earns top-level status when example code or tutorials show that the
qualified path is friction.

Submodule-only public symbols are reachable from their qualified
import path and covered by the same contract suite, even though they
are not in the top-level `__all__`:

```python
from better_robot.lie         import SO3, Pose
from better_robot.spatial     import Motion, Force, Inertia, Symmetric3
from better_robot.kinematics  import ReferenceFrame
from better_robot.optim.state import SolverState
from better_robot.tasks.ik    import IKResult, IKCostConfig, OptimizerConfig
```

`tests/contract/test_submodule_public_imports.py` enforces those
paths so a refactor cannot silently move them.

## Extension seams

Growth happens at `Protocol`-shaped seams — every place a user might
want to plug in their own implementation is documented as a
structural type. The complete catalogue (residuals, joints,
optimisers, robust kernels, damping strategies, linear solvers,
collision primitives, render modes, parsers, backends, trajectory
parameterisations, asset resolvers, actuators) lives in
{doc}`/conventions/extension`. Core layers import only the Protocol,
not concrete classes; this keeps the DAG stable as the extension set
grows.

## Why this shape works

- **`lie` + `spatial` split.** `lie/` is the algebraic machinery;
  `spatial/` is the 6D value-type layer that dynamics consumes.
  Mirrors Pinocchio's separation of `liegroups.hpp` from
  `spatial/`.
- **`data_model/joint_models/` one-file-per-joint.** Adding a new
  joint kind is an isolated change. See
  {doc}`joints_bodies_frames`.
- **One Jacobian entry point.** `kinematics/jacobian_strategy.py` is
  a tiny dispatch module that picks analytic, autodiff, or
  functional. There is **one** Jacobian function the solver calls;
  see {doc}`kinematics`.
- **`residuals` above `kinematics`.** Every residual reaches down
  through FK; none reaches into solvers. `costs/` is a pure
  composition layer above residuals. `optim/` knows about
  `LeastSquaresProblem` and solvers but not about specific
  residuals. `tasks/` is the top user-facing facade.
- **`io` and `viewer` siblings of `tasks`, not ancestors.** `load()`
  never constructs a `Task`; it returns a `Model`. The viewer is
  outside the spine because nothing should depend on it.

## Where to look next

The remaining chapters walk down the spine starting from the user
side: {doc}`model_and_data` is what `load` returns, {doc}`kinematics`
is the first thing most users call, and {doc}`tasks` is what most
production code actually uses.
