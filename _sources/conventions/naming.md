# Naming

> **Status:** normative. Every symbol in `src/better_robot/` follows the
> conventions on this page.

Names are the smallest decisions a library makes and the ones users
interact with most. A field called `oMi` saves three keystrokes and
costs every reader a context switch into Pinocchio's `aMb` notation.
Multiply that across a 5,000-line library, a docs site, every code
review, and every Stack Overflow answer that gets pasted into a new
project, and you can feel why Pinocchio's storage names became a tax on
adoption rather than a feature for experts.

We kept the parts of Pinocchio's vocabulary that are universal in the
literature — `rnea`, `aba`, `crba`, `SE3`, `Motion`, `Force`, `Inertia`
— because every textbook on rigid-body dynamics uses them, and renaming
those would force readers to look up the equivalence every time they
opened a paper. We replaced everything *not* universal — every storage
field, every dataclass attribute, every user-facing tensor — with a
self-describing identifier. `oMi` becomes `joint_pose_world`. `nle`
becomes `bias_forces`. `Ag` becomes `centroidal_momentum_matrix`. None
of those expansions cost speed; all of them make the call site
self-documenting.

The discipline is enforced by `tests/contract/test_naming.py`, which
greps `src/` for the cryptic names and fails the suite if a new caller
slips one in. The rest of this document is the canonical table the
test points at.

## 1 · Convention

### 1.1 Storage names

> *Storage fields (attributes on `Data`, `Model`, `IKResult`, etc.) are
> self-documenting nouns. Abbreviations are forbidden except for universally
> standardised math notation.*

The pattern is **`<entity>_<quantity>_<frame>`**:

| Part | Meaning | Examples |
|------|---------|----------|
| `<entity>` | What the tensor is about | `joint`, `frame`, `body`, `com`, `link` |
| `<quantity>` | The physical quantity | `pose`, `velocity`, `acceleration`, `inertia`, `jacobian`, `momentum` |
| `<frame>` | The coordinate frame it is expressed in | `world`, `local`, `body`, `com` |

Read left to right: `joint_velocity_world` ≡ "the velocity of each
joint, expressed in the world frame." No glossary required.

### 1.2 Function names

> *Functions named after published algorithms keep their canonical acronym
> (lowercased, snake_case). Functions named by what they compute use full
> English verb phrases.*

| Style | Rule | Example |
|-------|------|---------|
| Algorithm acronym | Use the acronym from the paper or textbook | `rnea`, `aba`, `crba`, `ccrba` |
| Compute verb | `compute_<noun>` returns a new tensor and may fill `Data` | `compute_joint_jacobians`, `compute_centroidal_map` |
| Get verb | `get_<noun>` reads from `Data`, cheap, no allocation | `get_joint_jacobian`, `get_frame_jacobian` |
| Update verb | `update_<noun>` writes one field of `Data` in place | `update_frame_placements` |
| Top-level façade | English imperative, no prefix | `forward_kinematics`, `solve_ik`, `solve_trajopt`, `retarget` |

### 1.3 Math notation that stays

Some symbols are so universal in rigid-body and Lie-group literature
that renaming costs more than it adds. These keep their symbols:

| Symbol | Meaning | Why we keep it |
|--------|---------|----------------|
| `q` | generalised configuration | Featherstone, Siciliano, Pinocchio |
| `v` | generalised velocity (= `dq/dt` for Euclidean joints) | Same |
| `a` | generalised acceleration | Same |
| `tau` | generalised torque/force | Same |
| `nq` | config-space dim | Same |
| `nv` | tangent-space dim | Same |
| `SE3` / `SO3` | Lie groups | Universal |
| `Jr`, `Jl`, `Jr_inv`, `Jl_inv` | right/left Jacobians of `exp` | Chirikjian, Barfoot |
| `hat`, `vee` | isomorphisms 𝔰𝔬(3)↔ℝ³, 𝔰𝔢(3)↔ℝ⁶ | Standard |
| `ad`, `Ad` | adjoint (algebra / group) | Distinguish in docstrings |
| `exp`, `log` | group exp and log | Universal |

**Rule of thumb:** if the symbol appears identically in Featherstone's
*Rigid Body Dynamics Algorithms* or Siciliano's *Robotics: Modelling,
Planning and Control*, keep it. Otherwise, expand it.

## 2 · Rename table

### 2.1 Per-joint kinematic state (`Data`)

| Old (Pinocchio) | New (BetterRobot) | Notes |
|-----|-----|-------|
| `liMi` | `joint_pose_local` | Joint placement in parent-joint frame. `(B..., njoints, 7)` |
| `oMi` | `joint_pose_world` | Joint placement in world frame. `(B..., njoints, 7)` |
| `oMf` | `frame_pose_world` | Frame placement in world frame. `(B..., nframes, 7)` |
| `v_joint` | `joint_velocity_local` | Joint twist in joint-local frame. `(B..., njoints, 6)` |
| `ov` | `joint_velocity_world` | Joint twist in world frame. `(B..., njoints, 6)` |
| `a_joint` | `joint_acceleration_local` | `(B..., njoints, 6)` |
| `oa` | `joint_acceleration_world` | `(B..., njoints, 6)` |

### 2.2 Jacobians (`Data`)

| Old | New | Notes |
|-----|-----|-------|
| `J` | `joint_jacobians` | Full stack of per-joint spatial Jacobians. `(B..., njoints, 6, nv)` |
| `dJ` | `joint_jacobians_dot` | Time derivative. `(B..., njoints, 6, nv)` |

### 2.3 Dynamics (`Data`)

| Old | New | Notes |
|-----|-----|-------|
| `M` | `mass_matrix` | `(B..., nv, nv)` |
| `C` | `coriolis_matrix` | `(B..., nv, nv)` |
| `g` | `gravity_torque` | `(B..., nv)` |
| `nle` | `bias_forces` | `C(q,v)·v + g(q)`. "Bias" is the canonical Featherstone term. `(B..., nv)` |
| `ddq` | `ddq` (kept) | `q̈`. Universal math notation. `(B..., nv)` |

### 2.4 Centroidal (`Data`)

| Old | New | Notes |
|-----|-----|-------|
| `Ag` | `centroidal_momentum_matrix` | `(B..., 6, nv)` |
| `hg` | `centroidal_momentum` | `(B..., 6)` |
| `com` | `com_position` | `(B..., 3)` |
| `vcom` | `com_velocity` | `(B..., 3)` |
| `acom` | `com_acceleration` | `(B..., 3)` |
| `Ycrb` | `composite_inertia` | `(B..., njoints, 10)` |
| `Jcom` | `com_jacobian` | `(B..., 3, nv)` |

### 2.5 Model topology (`Model`)

`Model` attributes are already readable. The two contractions that stay:

| Name | Meaning | Keep because |
|------|---------|--------------|
| `nq`, `nv`, `njoints`, `nbodies`, `nframes` | dimensions | Every rigid-body text uses these |
| `idx_qs`, `idx_vs` | per-joint start indices into `q` / `v` | The verbose alternative (`q_start_indices`) is worse and the name is documented |

Everything else on `Model` (`joint_placements`, `body_inertias`,
`lower_pos_limit`, `topo_order`, `parents`, `children`, `subtrees`,
`supports`) is already self-documenting.

### 2.6 Lie / spatial layer

Lie-algebra symbols (`Jr`, `hat`, `vee`, `ad`, `Ad`, `exp`, `log`) are
kept as-is but become **functions**, never bare identifiers on a
tensor:

```python
from better_robot import lie
J = lie.right_jacobian_se3(xi)
```

Internal implementations may still use the short form in tight
algebraic blocks; the public API uses the English form so IDE
autocomplete and help text are readable. The short form is also
available as an alias:

```python
from better_robot.lie import Jr, Jr_inv
```

### 2.7 Optim / Tasks

The user-facing names already follow the conventions above:

| Name | Purpose |
|----------|---------|
| `IKResult` | Result of `solve_ik` |
| `TrajOptResult` | Result of `solve_trajopt` |
| `IKCostConfig` | User-facing knobs for the IK cost stack |
| `OptimizerConfig` | User-facing knobs for the optimiser |
| `SolverState` | Per-iteration state shared between `Optimizer`, `DampingStrategy`, `LinearSolver` |
| `ResidualSpec` | Structural metadata returned from `Residual.spec(state)` |
| `OptimizerStage` / `MultiStageOptimizer` | Composite of stages |
| `TrajectoryParameterization` | Protocol; concrete `KnotTrajectory`, `BSplineTrajectory` |

### 2.8 Enums replacing string literals

| Enum | Purpose | Members |
|------|---------|---------|
| `ReferenceFrame` (in `kinematics`) | Replaces `reference="..."` strings on `get_*_jacobian` | `WORLD`, `LOCAL`, `LOCAL_WORLD_ALIGNED` |
| `KinematicsLevel` (in `data_model`) | Tracks how far FK has been computed on a `Data` | `NONE` (0), `PLACEMENTS` (1), `VELOCITIES` (2), `ACCELERATIONS` (3) |
| `JacobianStrategy` (in `kinematics`) | Selects analytic / autodiff / FD | `ANALYTIC`, `AUTODIFF`, `FUNCTIONAL`, `FINITE_DIFF`, `AUTO` |

All three subclass `str` (`int` for `KinematicsLevel`) so user code that
still compares to a string literal continues to work.

### 2.9 Shape annotations — `_typing.py` aliases

Public functions use `jaxtyping`-style annotations. The canonical
aliases live in `src/better_robot/_typing.py`:

```python
from jaxtyping import Float, Int
from torch import Tensor

# Single SE3 / SO3 storage tensors
SE3Tensor       = Float[Tensor, "*B 7"]
SO3Tensor       = Float[Tensor, "*B 4"]
Quaternion      = Float[Tensor, "*B 4"]      # (qx, qy, qz, qw)
TangentSE3      = Float[Tensor, "*B 6"]
TangentSO3      = Float[Tensor, "*B 3"]
# Per-joint and per-frame stacks
JointPoseStack  = Float[Tensor, "*B njoints 7"]
FramePoseStack  = Float[Tensor, "*B nframes 7"]
# Configurations and tangents
ConfigTensor    = Float[Tensor, "*B nq"]
VelocityTensor  = Float[Tensor, "*B nv"]
# Jacobians
JointJacobian      = Float[Tensor, "*B 6 nv"]
JointJacobianStack = Float[Tensor, "*B njoints 6 nv"]
```

Coverage is enforced by `tests/contract/test_shape_annotations.py`
(advisory until the typing migration completes; blocking thereafter).

## 3 · Glossary

| Term | Meaning |
|------|---------|
| **Model** | Immutable kinematic tree: joints, bodies, frames, inertias, limits. One per robot, shared across queries, moved with `.to(device, dtype)`. |
| **Data** | Mutable per-query workspace. Holds `q`, lazy kinematic/dynamic caches. One per batch × time evaluation. |
| **Frame** | Any named coordinate frame on the robot — joint frames, body frames, user-declared operational frames. |
| **Joint** | A single degree of articulation. "Joint 0" is always the universe (world). |
| **Body** | A rigid link. Bodies are 1:1 with joints except joint 0; inertia is attached via `joint_placements`. |
| **joint_pose_local** | SE(3) transform from a joint's parent joint to itself. |
| **joint_pose_world** | SE(3) transform from world origin to a joint. |
| **frame_pose_world** | SE(3) transform from world origin to a frame. |
| **joint_velocity_world** | 6D twist of each joint, linear first, in world axes. |
| **spatial Jacobian** | 6 × nv Jacobian relating `v` to twist; `linear rows | angular rows`. |
| **body-frame Jacobian** | Spatial Jacobian with the origin's twist re-expressed in the body's axes. |
| **LOCAL_WORLD_ALIGNED** | Jacobian of a point on the body, *translated* to the world origin but *rotated* with world axes. Default for `get_frame_jacobian`. |
| **Residual** | A differentiable function `r(model, data, …) -> (B..., dim)` — the quantity the optimiser drives toward zero. |
| **CostStack** | Weighted concatenation of residuals; returns a single flat residual vector. |
| **LeastSquaresProblem** | `(cost_stack, x0, bounds, jacobian_strategy)` — a fully specified optimisation problem. |
| **Optimizer** | A `Protocol` that minimises a `LeastSquaresProblem` (LM, GN, Adam, LBFGS, …). |
| **bias_forces** | `C(q, q̇) q̇ + g(q)` — the generalised force present even at zero input torque. |
| **centroidal momentum** | 6D momentum of the robot around its centre of mass. |
| **mass matrix** | Joint-space inertia `M(q)`. |
| **coriolis matrix** | `C(q, q̇)` such that `C(q, q̇) q̇` is the Coriolis/centrifugal torque. |
| **gravity torque** | `g(q)` — joint-space generalised gravity. |
| **Capsule** | Sphere-swept line segment; the default collision primitive for the robot surface. |
| **Gizmo** | Draggable SE(3) widget in the viewer used to set IK targets. |

## 4 · Renames at a glance

```
Pinocchio           ->  BetterRobot
─────────────────────────────────────────────────────────
oMi                 ->  joint_pose_world
oMf                 ->  frame_pose_world
liMi                ->  joint_pose_local
ov                  ->  joint_velocity_world
oa                  ->  joint_acceleration_world
v_joint             ->  joint_velocity_local
a_joint             ->  joint_acceleration_local
M                   ->  mass_matrix
C                   ->  coriolis_matrix
g                   ->  gravity_torque
nle                 ->  bias_forces
Ag                  ->  centroidal_momentum_matrix
hg                  ->  centroidal_momentum
com                 ->  com_position
vcom                ->  com_velocity
acom                ->  com_acceleration
Ycrb                ->  composite_inertia
Jcom                ->  com_jacobian
J                   ->  joint_jacobians
dJ                  ->  joint_jacobians_dot
```

## 5 · Enforcement

`tests/contract/test_naming.py` greps `src/` for any forbidden cryptic
identifier and fails the suite if one slips in. The check is
deliberately mechanical: a regex sweep is faster than a code review and
never gets tired. Every residual, solver, and task docstring uses the
new vocabulary; every contributor sees the discipline before they ship.

## 6 · For contributors

When you add a new field to `Data`, a new attribute to `Model`, or a
new tensor that a user will read:

1. **Read §1.** Pick a name that follows `<entity>_<quantity>_<frame>`
   or the function-naming patterns.
2. **Check §1.3.** If the quantity has a universal math symbol, use it
   as a variable *inside* a function but expose the verbose name in any
   public-facing docstring or attribute.
3. **Add to §3.** Any genuinely new concept gets one row in the
   glossary.
4. **If the name is cryptic, it is wrong.** The reviewer's first
   question will be "what does this mean?" — if the answer is not
   obvious from the identifier, rename before merging.
