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

```
src/better_robot/
├── __init__.py                    # Small public API (≤25 exports)
├── _typing.py                     # Shape-annotated type aliases (jaxtyping-style)
│
├── backends/
│   ├── __init__.py                # current_backend() / set_backend()
│   ├── torch_native/              # default; pure torch kernels
│   │   ├── __init__.py
│   │   └── ops.py                 # primitive ops (matmul, scan, gather, …)
│   └── warp/                      # optional future backend
│       ├── __init__.py
│       ├── bridge.py              # wp.from_torch / wp.to_torch + autograd.Function wrappers
│       ├── kernels/
│       └── graph_capture.py       # CUDA graph capture, exposed as @graph_capture
│
├── lie/                           # lie groups + tangent algebra
│   ├── __init__.py
│   ├── se3.py                     # SE3: compose/inverse/exp/log/act/adjoint
│   ├── so3.py                     # SO3: compose/inverse/exp/log/act/adjoint
│   ├── tangents.py                # Jr/Jl Jacobians, hat/vee, BCH helpers
│   └── _pypose_backend.py         # private — holds every pypose import today
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
│   ├── problem.py                 # LeastSquaresProblem interface
│   ├── optimizers/
│   │   ├── base.py
│   │   ├── gauss_newton.py
│   │   ├── levenberg_marquardt.py
│   │   ├── adam.py
│   │   └── lbfgs.py
│   ├── solvers/                   # linear solvers
│   │   ├── cholesky.py
│   │   ├── lstsq.py
│   │   ├── cg.py
│   │   └── sparse_cholesky.py
│   ├── kernels/                   # robust kernels
│   │   ├── l2.py
│   │   ├── huber.py
│   │   └── cauchy.py
│   ├── strategies/                # damping / trust region / schedule
│   │   ├── constant.py
│   │   ├── adaptive.py
│   │   └── trust_region.py
│   └── jacobian_spec.py           # ResidualSpec (shape, sparsity hints)
│
├── tasks/
│   ├── __init__.py                # solve_ik, solve_trajopt, retarget
│   ├── ik.py                      # thin facade: builds problem → optim.solve
│   ├── trajopt.py                 # Trajectory dataclass + solve_trajopt
│   ├── retarget.py                # stub
│   └── trajectory.py              # Trajectory dataclass (B, T, …)
│
├── collision/
│   ├── __init__.py
│   ├── geometry.py                # Sphere, Capsule, Box, HalfSpace, Plane
│   ├── pairs.py                   # pairwise SDF dispatch table
│   ├── robot_collision.py         # RobotCollision (optional layer)
│   └── closest_pts.py             # segment-segment, point-capsule, etc.
│
├── io/
│   ├── __init__.py                # load(path) suffix-dispatched
│   ├── ir.py                      # intermediate representation
│   ├── parsers/
│   │   ├── urdf.py
│   │   ├── mjcf.py
│   │   └── programmatic.py        # builder DSL
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
from .io import load
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
from .optim import LeastSquaresProblem, solve
from .tasks import solve_ik, solve_trajopt, retarget, Trajectory

__all__ = [
    "Model", "Data", "Frame", "Joint", "Body",
    "load",
    "forward_kinematics", "update_frame_placements",
    "compute_joint_jacobians", "get_joint_jacobian", "get_frame_jacobian",
    "JacobianStrategy",
    "rnea", "aba", "crba", "center_of_mass", "compute_centroidal_map",
    "register_residual", "CostStack",
    "LeastSquaresProblem", "solve",
    "solve_ik", "solve_trajopt", "retarget", "Trajectory",
]
```

That's **25** symbols — the non-negotiable ceiling.

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
  [15_EXTENSION.md §2](15_EXTENSION.md).
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
modes, parsers, backends — lives in [15_EXTENSION.md](15_EXTENSION.md).
Core layers import *only* the Protocol, not concrete classes; this keeps
the DAG stable as the extension set grows.

## Testing story

The test tree is spec'd fully in [16_TESTING.md](16_TESTING.md). Headlines:

- `tests/contract/test_layer_dependencies.py` — AST-parses imports and enforces the DAG.
- `tests/contract/test_public_api.py` — verifies `__all__` matches spec, counts
  exports, and checks each public symbol has a docstring.
- `tests/contract/test_skeleton_signatures.py` — imports every symbol, introspects its
  signature, checks it matches the plan.
- `tests/contract/test_naming.py` — grep-enforces the rename table in
  [13_NAMING.md](13_NAMING.md) (no pinocchio cryptic names in new code).
- `tests/contract/test_hot_path_lint.py` — enforces the perf
  anti-patterns in [14_PERFORMANCE.md §3](14_PERFORMANCE.md).
- `tests/kinematics/`, `tests/dynamics/`, `tests/tasks/` — numerical tests; each
  file exercises one concept.
- `tests/bench/` — benchmarks gated against the budgets in [14_PERFORMANCE.md](14_PERFORMANCE.md).
- `tests/examples/` — every runnable script under `examples/` is imported from
  a test (PyRoki pattern).
