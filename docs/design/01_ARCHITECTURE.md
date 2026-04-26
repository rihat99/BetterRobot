# 01 · Architecture

## Layered design

BetterRobot is organised as a strict dependency DAG. Arrows point from
dependent to dependency; nothing may ever point backwards.

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

## Target directory layout

The skeleton has landed: every top-level submodule below already exists
in `src/better_robot/`. Files marked **(planned)** are tracked in
[UPDATE_PHASES.md](../UPDATE_PHASES.md). Today's `__all__` has 25
symbols; the 26-symbol target below adds `SE3` (P2) and `ModelBuilder`
(P5).

```
src/better_robot/
├── __init__.py                    # public API (25 today → 26)
├── _typing.py                     # Shape-annotated type aliases (jaxtyping-style)
│
├── backends/
│   ├── __init__.py                # default_backend() / set_backend() / get_backend()
│   ├── protocol.py                # (planned, P1) Backend / LieOps / KinematicsOps / DynamicsOps Protocols
│   ├── torch_native/              # default after P1
│   │   ├── __init__.py            # TorchNativeBackend, BACKEND
│   │   ├── lie_ops.py             # (planned, P1) routes to lie/_torch_native_backend.py
│   │   ├── kinematics_ops.py      # (planned, P1) forward_kinematics, jacobian assembly
│   │   └── dynamics_ops.py        # (planned, P11) rnea/aba/crba (stubs until D2-D4)
│   └── warp/                      # optional future backend (experimental)
│       ├── __init__.py
│       ├── bridge.py              # wp.from_torch / wp.to_torch + autograd.Function wrappers
│       ├── kernels/
│       └── graph_capture.py       # CUDA graph capture, exposed as @graph_capture
│
├── lie/                           # lie groups + tangent algebra
│   ├── __init__.py
│   ├── types.py                   # (planned, P2) SE3 / SO3 / Pose typed dataclasses (.tensor field)
│   ├── se3.py                     # SE3: compose/inverse/exp/log/act/adjoint (functional)
│   ├── so3.py                     # SO3: compose/inverse/exp/log/act/adjoint (functional)
│   ├── tangents.py                # Jr/Jl Jacobians, hat/vee, BCH helpers (pure-torch)
│   ├── _torch_native_backend.py   # (planned, P10/L-A) pure-PyTorch SE3/SO3 kernels — becomes default after L-C
│   └── _pypose_backend.py         # current default; reachable via BR_LIE_BACKEND=pypose after L-C; deleted in L-D
│
├── spatial/                       # 6D spatial algebra (Pinocchio-style)
│   ├── __init__.py
│   ├── motion.py                  # Motion value type (6D twist)
│   ├── force.py                   # Force value type (6D wrench)
│   ├── inertia.py                 # Inertia: mass + com + symmetric3
│   ├── symmetric3.py              # 3×3 symmetric
│   └── ops.py                     # ad, Ad, cross(Motion, Motion), cross(Motion, Force)
│
├── data_model/                    # pinocchio-style Model / Data
│   ├── __init__.py                # Model, Data, Frame, Joint, Body, build_model()
│   ├── model.py                   # frozen Model dataclass
│   ├── data.py                    # mutable per-query Data
│   ├── joint.py                   # Joint enum + metadata tables
│   ├── joint_models/              # one file per joint family (see 02 DATA_MODEL)
│   │   ├── __init__.py
│   │   ├── base.py                # JointModel protocol
│   │   ├── fixed.py
│   │   ├── revolute.py            # RX/RY/RZ/unaligned
│   │   ├── prismatic.py           # PX/PY/PZ/unaligned
│   │   ├── spherical.py           # SO3 ball joint
│   │   ├── free_flyer.py          # SE3 floating base
│   │   ├── planar.py
│   │   ├── translation.py
│   │   ├── helical.py
│   │   ├── composite.py
│   │   └── mimic.py
│   ├── frame.py                   # Frame struct (name, parent_joint, placement, type)
│   ├── body.py                    # Body (link) struct + inertia
│   ├── topology.py                # topo sort, parent/child/subtree helpers
│   └── indexing.py                # name→id lookup tables
│
├── kinematics/
│   ├── __init__.py
│   ├── forward.py                 # forward_kinematics(), update_frame_placements()
│   ├── jacobian.py                # compute_joint_jacobians(), get_joint_jacobian(), get_frame_jacobian()
│   ├── jacobian_strategy.py       # ANALYTIC | AUTODIFF | FUNCTIONAL dispatch
│   └── chain.py                   # subtree / chain helpers
│
├── dynamics/                      # skeleton only in v1
│   ├── __init__.py
│   ├── rnea.py                    # inverse dynamics
│   ├── aba.py                     # forward dynamics
│   ├── crba.py                    # joint-space inertia
│   ├── centroidal.py              # ccrba, com, jacobian_of_com
│   ├── derivatives.py             # rnea_derivatives, aba_derivatives, crba_derivatives
│   ├── action/                    # crocoddyl-style 3-layer split (later)
│   │   ├── differential.py
│   │   ├── integrated.py
│   │   └── action.py
│   └── integrators.py             # euler, rk4, symplectic
│
├── residuals/                     # pure functions → CostStack
│   ├── __init__.py                # @register_residual decorator
│   ├── pose.py                    # pose, position, orientation
│   ├── limits.py                  # joint/velocity/acceleration bounds
│   ├── smoothness.py              # 5-point finite-diff velocity/accel/jerk
│   ├── manipulability.py          # Yoshikawa index
│   ├── collision.py               # self/world collision margins
│   ├── regularization.py          # rest/null-space regularizer
│   └── registry.py
│
├── costs/
│   ├── __init__.py                # CostStack
│   ├── stack.py                   # named, weighted, individually activatable
│   └── factory.py                 # Cost.factory(residual_fn)
│
├── optim/
│   ├── __init__.py                # LeastSquaresProblem, solve()
│   ├── problem.py                 # LeastSquaresProblem; .residual(), .jacobian(); .gradient() (P6.4, matrix-free)
│   ├── state.py                   # SolverState (per-iter state shared between optimizer/strategy/solver)
│   ├── optimizers/
│   │   ├── base.py                # Optimizer Protocol
│   │   ├── gauss_newton.py
│   │   ├── levenberg_marquardt.py
│   │   ├── adam.py                # P6.4: switches to problem.gradient(x); never materialises J
│   │   ├── lbfgs.py               # same
│   │   ├── composite.py           # current LM+LBFGS two-stage; renamed/generalised in P6.5
│   │   ├── multi_stage.py         # (planned, P6.5) MultiStageOptimizer + OptimizerStage
│   │   └── lm_then_lbfgs.py       # (planned, P6.5) backward-compat thin wrapper
│   ├── linear_solvers/            # (planned, P6.2) LinearSolver Protocol implementations
│   │   ├── cholesky.py
│   │   ├── qr.py
│   │   ├── lsqr.py
│   │   ├── cg.py
│   │   └── block_cholesky.py      # for sparse trajopt (per ResidualSpec)
│   ├── kernels/                   # RobustKernel Protocol — Identity/Huber/Cauchy/Tukey
│   │   ├── identity.py
│   │   ├── huber.py
│   │   ├── cauchy.py
│   │   └── tukey.py
│   ├── damping.py                 # (planned, P6.3) DampingStrategy Protocol — Constant/Adaptive/TrustRegion
│   └── jacobian_spec.py           # ResidualSpec (output_dim, structure, time_coupling, affected_knots, dynamic_dim)
│
├── tasks/
│   ├── __init__.py                # solve_ik, solve_trajopt, retarget, Trajectory
│   ├── ik.py                      # thin facade: builds problem → optim.solve; honours every OptimizerConfig knob
│   ├── trajopt.py                 # solve_trajopt — accepts parameterization=
│   ├── retarget.py                # stub — reduces to trajopt
│   ├── trajectory.py              # Trajectory dataclass — accepts (T,nq) and (*B,T,nq)
│   └── parameterization.py        # (planned, P7) TrajectoryParameterization Protocol; KnotTrajectory; BSplineTrajectory
│
├── collision/
│   ├── __init__.py
│   ├── geometry.py                # Sphere, Capsule, Box, HalfSpace, Plane
│   ├── pairs.py                   # pairwise SDF dispatch table
│   ├── robot_collision.py         # RobotCollision (optional layer)
│   └── closest_pts.py             # segment-segment, point-capsule, etc.
│
├── io/
│   ├── __init__.py                # load(path) suffix-dispatched; register_parser
│   ├── ir.py                      # IRModel; schema_version landed in P0
│   ├── build_model.py             # IR → Model factory; raises IRSchemaVersionError on mismatch
│   ├── assets.py                  # (planned, P5) AssetResolver Protocol + Filesystem/Package/Composite/CachedDownload
│   ├── parsers/
│   │   ├── urdf.py                # parses URDF → IRModel; AssetResolver-aware after P5 (lazy yourdfpy import)
│   │   ├── mjcf.py                # parses MJCF (lazy mujoco import)
│   │   └── programmatic.py        # ModelBuilder DSL — named helpers land in P5 (add_revolute_z, add_free_flyer_root)
│   └── builders/
│       └── smpl_like.py           # example: SMPL-like body with fixed shape
│
├── viewer/
│   ├── __init__.py
│   ├── visualizer.py              # viser wrapper
│   └── helpers.py
│
└── utils/
    ├── batching.py                # leading-batch helpers
    ├── broadcasting.py
    ├── logging.py
    └── testing.py                 # assert_close_manifold, fk_regression, …
```

## Dependency rule

```
backends → lie → spatial → data_model → (kinematics, dynamics) → residuals → costs → optim → tasks
                                                       ↑                                  │
                                                       └── collision ─────────────────────┘
io → data_model          (io reads nothing from optim or tasks)
viewer → tasks (topmost; no-one imports from viewer)
```

A linter test (`tests/test_layer_dependencies.py`) will parse each file's
imports and fail the suite if any arrow points backwards.

## Public API contract

```python
# better_robot/__init__.py — EVERY public symbol listed explicitly.
from .data_model import Model, Data, Frame, Joint, Body
from .io import load, ModelBuilder
from .lie.types import SE3
from .kinematics import (
    forward_kinematics,
    update_frame_placements,
    compute_joint_jacobians,
    get_joint_jacobian,
    get_frame_jacobian,
    JacobianStrategy,
)
from .dynamics import rnea, aba, crba, center_of_mass, compute_centroidal_map
from .residuals import register_residual
from .costs import CostStack
from .optim import LeastSquaresProblem
from .tasks import solve_ik, solve_trajopt, retarget, Trajectory

__all__ = [
    # Robot model & data (7)
    "Model", "Data", "Frame", "Joint", "Body",
    "load", "ModelBuilder",
    # Geometric value types — top-level (1)
    "SE3",
    # Kinematics (5)
    "forward_kinematics", "update_frame_placements",
    "compute_joint_jacobians", "get_joint_jacobian", "get_frame_jacobian",
    # Dynamics (5)
    "rnea", "aba", "crba", "center_of_mass", "compute_centroidal_map",
    # Optimisation primitives (3)
    "JacobianStrategy", "CostStack", "LeastSquaresProblem",
    # Tasks (4)
    "solve_ik", "solve_trajopt", "retarget", "Trajectory",
    # Extension points (1)
    "register_residual",
]
```

**26 symbols** — audited, not capped. The discipline lives in
`tests/contract/test_public_api.py` (frozen `EXPECTED` set) and
`tests/contract/test_submodule_public_imports.py` (asserts
intentionally-non-top-level symbols stay reachable). Promotion is
evidence-driven: a symbol earns top-level when example code or tutorials
demonstrate the qualified path is friction.

**Submodule-only public symbols** (reachable, not promoted):

```python
from better_robot.lie         import SO3, Pose
from better_robot.spatial     import Motion, Force, Inertia, Symmetric3
from better_robot.kinematics  import ReferenceFrame
from better_robot.optim.state import SolverState
from better_robot.tasks.ik    import IKResult, IKCostConfig, OptimizerConfig
```

These are *fully public* — covered by SemVer, contract-tested for
reachability — they simply are not in `import better_robot as br; br.<TAB>`.

## What each layer owns

| Layer | Owns | Forbidden imports |
|-------|------|-------------------|
| `backends` | primitive ops | anything above |
| `lie` | SE3/SO3 exp/log/act | anything above `backends` |
| `spatial` | Motion/Force/Inertia value types | anything above `lie` |
| `data_model` | `Model`, `Data`, `JointModel`s | `kinematics`/`dynamics`/above |
| `kinematics` | FK, frame updates, Jacobians | `dynamics`/`residuals`/above |
| `dynamics` | RNEA/ABA/CRBA stubs | `residuals`/above |
| `residuals` | Pure residual functions | `costs`/`optim`/`tasks`/`io`/`viewer` |
| `costs` | `CostStack` | `optim`/`tasks`/`io`/`viewer` |
| `optim` | Problem interface, solvers, kernels | `tasks`/`io`/`viewer` |
| `collision` | Geometry, SDF pairs | `tasks`/`io`/`viewer` |
| `io` | Parsers, IR, builders | `tasks`/`viewer` |
| `tasks` | `solve_ik`, `solve_trajopt`, `retarget` facades | `viewer` |
| `viewer` | viser bindings | — |

## Why this shape

- **`lie` + `spatial` split** mirrors Pinocchio's separation of
  `liegroups.hpp` from `spatial/`. `lie` is the algebraic machinery; `spatial`
  is the 6D value-type layer used by dynamics.
- **`data_model/joint_models/`** one-file-per-joint makes adding a new joint
  (say `CoupledJoint` with a splined coordinate) an isolated change. See
  [15_EXTENSION.md §2](../conventions/15_EXTENSION.md).
- **`kinematics/jacobian_strategy.py`** is a *tiny* dispatch module that picks
  analytic, autodiff, or functional — there is **one** Jacobian entry point;
  see 05 KINEMATICS.
- **`residuals` is above `kinematics`** because every residual reaches down
  through FK but not into solvers. `costs` is a pure composition layer above
  residuals. `optim` knows about `LeastSquaresProblem` and solvers but not
  about specific residuals. `tasks` is the top user-facing facade.
- **`io` and `viewer` are siblings of `tasks`**, not ancestors. `load()` never
  constructs a `Task`; it returns a `Model`.

## Extension seams

Growth happens at **`Protocol`-shaped seams** — every place a user might
want to plug in their own implementation is documented as a structural
type. The complete catalogue — residuals, joints, optimisers, robust
kernels, damping strategies, linear solvers, collision primitives, render
modes, parsers, backends — lives in [15_EXTENSION.md](../conventions/15_EXTENSION.md).
Core layers import *only* the Protocol, not concrete classes; this keeps
the DAG stable as the extension set grows.

## Testing story

The test tree is spec'd fully in [16_TESTING.md](../conventions/16_TESTING.md). Headlines:

- `tests/contract/test_layer_dependencies.py` — AST-parses imports and enforces the DAG.
- `tests/contract/test_backend_boundary.py` — only `lie/`, `kinematics/`, `dynamics/` cross the backend boundary; nothing else imports a backend implementation.
- `tests/contract/test_public_api.py` — `__all__` matches the frozen `EXPECTED` set (26 symbols).
- `tests/contract/test_submodule_public_imports.py` — symbols documented as submodule-only stay reachable from their qualified path.
- `tests/contract/test_skeleton_signatures.py` — imports every symbol, introspects its signature, checks it matches the plan.
- `tests/contract/test_naming.py` — grep-enforces the rename table in
  [13_NAMING.md](../conventions/13_NAMING.md) (no pinocchio cryptic names in new code).
- `tests/contract/test_hot_path_lint.py` — enforces the perf
  anti-patterns in [14_PERFORMANCE.md §3](../conventions/14_PERFORMANCE.md).
- `tests/contract/test_cache_invariants.py` — `Data._kinematics_level` enforced; in-place mutation documented as out-of-scope.
- `tests/contract/test_shape_annotations.py` — jaxtyping coverage on the public surface (advisory until Stage 3).
- `tests/contract/test_no_legacy_strings.py` — no `reference="world|local|local_world_aligned"` literals in `src/`.
- `tests/contract/test_optional_imports.py` — `import better_robot` does not pull `yourdfpy`/`mujoco`/`viser`/`warp`/`sphinx`/`robot_descriptions`/`pinocchio`.
- `tests/kinematics/test_fk_regression.py` — frozen FK oracle (`fk_reference.npz`); fp64 ulp tolerance.
- `tests/kinematics/`, `tests/dynamics/`, `tests/tasks/` — numerical tests; each
  file exercises one concept.
- `tests/bench/` — benchmarks gated against the budgets in [14_PERFORMANCE.md](../conventions/14_PERFORMANCE.md) under the advisory-then-blocking ladder ([16_TESTING.md §gate-promotion ladder](../conventions/16_TESTING.md)).
- `tests/examples/` — every runnable script under `examples/` is imported from
  a test (PyRoki pattern).
