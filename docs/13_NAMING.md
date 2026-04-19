# 13 · Naming & Glossary

> **Status:** normative. Every symbol in `src/better_robot/` must follow the
> conventions here. Where existing code violates them, the name is carried as
> a deprecated alias for exactly **one** release and then removed.

The library stole vocabulary from Pinocchio because the mathematics are
the same. But Pinocchio's storage fields are notation-heavy (`oMi`,
`oMf`, `liMi`, `nle`, `Ycrb`) and opaque to readers who haven't internalised
the `aMb` convention. The cost compounds — every docstring has to re-explain
`oMi`, every code review requires the reviewer to decode it, every new
contributor faces the same ramp-up.

This document fixes that. We **keep canonical algorithm names** (`rnea`,
`aba`, `crba`, `se3.exp`, `Motion`) because those are universal in the
literature. We **rename every storage field, dataclass attribute, and
user-facing tensor** to a self-describing identifier.

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
| `<frame>` | The coordinate frame it's expressed in | `world`, `local`, `body`, `com` |

Read left-to-right: `joint_velocity_world` ≡ "the velocity of each joint,
expressed in the world frame." No glossary required.

### 1.2 Function names

> *Functions named after published algorithms keep their canonical acronym
> (lowercased, snake_case). Functions named by what they compute use full
> English verb phrases.*

| Style | Rule | Example |
|-------|------|---------|
| Algorithm acronym | Use the acronym from the paper/book | `rnea`, `aba`, `crba`, `ccrba` |
| Compute verb | `compute_<noun>` returns a new tensor, may fill `Data` | `compute_joint_jacobians`, `compute_centroidal_map` |
| Get verb | `get_<noun>` reads from `Data`, cheap, no allocation | `get_joint_jacobian`, `get_frame_jacobian` |
| Update verb | `update_<noun>` writes one field of `Data` in place | `update_frame_placements` |
| Top-level façade | English imperative, no prefix | `forward_kinematics`, `solve_ik`, `solve_trajopt`, `retarget` |

### 1.3 Math notation that stays

Some symbols are so universal in rigid-body / Lie-group literature that
renaming costs readability. These **keep their symbols**:

| Symbol | Meaning | Why we keep it |
|--------|---------|----------------|
| `q` | generalised configuration | Featherstone, Siciliano, Pinocchio — standard |
| `v` | generalised velocity (= `dq/dt` for Euclidean joints) | Same |
| `a` | generalised acceleration | Same |
| `tau` | generalised torque/force | Same |
| `nq` | config-space dim | Same |
| `nv` | tangent-space dim | Same |
| `SE3` / `SO3` | Lie groups | Universal |
| `Jr`, `Jl`, `Jr_inv`, `Jl_inv` | right/left Jacobians of exp | Chirikjian, Barfoot — keep with docstring |
| `hat`, `vee` | isomorphisms 𝔰𝔬(3)↔ℝ³, 𝔰𝔢(3)↔ℝ⁶ | Standard |
| `ad`, `Ad` | adjoint (algebra / group) | Keep; distinguish in docstrings |
| `exp`, `log` | group exp and log | Universal |

**When in doubt**: if the symbol appears identically in Featherstone's
*Rigid Body Dynamics Algorithms* or Siciliano's *Robotics: Modelling,
Planning and Control*, keep it. Otherwise, expand it.

## 2 · Rename table

The **left column** is what currently exists in `src/better_robot/`. The
**right column** is the target. Rows marked *alias* get a deprecated alias
for one release; all others are hard renames (skeleton-only fields, no
external users).

### 2.1 Per-joint kinematic state (`Data`)

| Old | New | Notes |
|-----|-----|-------|
| `liMi` | `joint_pose_local` | *alias.* Joint placement in parent-joint frame. `(B..., njoints, 7)` |
| `oMi` | `joint_pose_world` | *alias.* Joint placement in world frame. `(B..., njoints, 7)` |
| `oMf` | `frame_pose_world` | *alias.* Frame placement in world frame. `(B..., nframes, 7)` |
| `v_joint` | `joint_velocity_local` | Joint twist in joint-local frame. `(B..., njoints, 6)` |
| `ov` | `joint_velocity_world` | Joint twist in world frame. `(B..., njoints, 6)` |
| `a_joint` | `joint_acceleration_local` | `(B..., njoints, 6)` |
| `oa` | `joint_acceleration_world` | `(B..., njoints, 6)` |

Two alias shims deliberately: `liMi`, `oMi`, `oMf` appear in *every*
Pinocchio-derived example on the internet. Keeping them one release makes
ports painless; dropping them in v1.1 makes the surface clean.

### 2.2 Jacobians (`Data`)

| Old | New | Notes |
|-----|-----|-------|
| `J` | `joint_jacobians` | Full stack of per-joint spatial Jacobians. `(B..., njoints, 6, nv)` |
| `dJ` | `joint_jacobians_dot` | Time derivative. `(B..., njoints, 6, nv)` |
| — | `frame_jacobians` *(new, optional cache)* | Per-frame Jacobians, populated on demand |

### 2.3 Dynamics (`Data`)

| Old | New | Notes |
|-----|-----|-------|
| `M` | `mass_matrix` | `(B..., nv, nv)` |
| `C` | `coriolis_matrix` | `(B..., nv, nv)` |
| `g` | `gravity_torque` | `(B..., nv)` |
| `nle` | `bias_forces` | `C(q,v)·v + g(q)`. "Bias" is the canonical term in Featherstone. `(B..., nv)` |
| `ddq` | keep | `q̈`. Universal math. `(B..., nv)` |

### 2.4 Centroidal (`Data`)

| Old | New | Notes |
|-----|-----|-------|
| `Ag` | `centroidal_momentum_matrix` | `(B..., 6, nv)` |
| `hg` | `centroidal_momentum` | `(B..., 6)` |
| `com` | `com_position` | `(B..., 3)` |
| `vcom` | `com_velocity` | `(B..., 3)` |
| `acom` | `com_acceleration` | `(B..., 3)` |
| `Ycrb` *(future)* | `composite_inertia` | `(B..., njoints, 10)` |
| `Jcom` *(future)* | `com_jacobian` | `(B..., 3, nv)` |

### 2.5 Model topology (`Model`)

`Model` attributes are already readable. The two contractions we keep:

| Name | Meaning | Keep because |
|------|---------|--------------|
| `nq`, `nv`, `njoints`, `nbodies`, `nframes` | dimensions | Every rigid-body text uses these |
| `idx_qs`, `idx_vs` | per-joint start indices into `q` / `v` | Verbose alternative (`q_start_indices`) is worse. Documented. |

Everything else on `Model` (`joint_placements`, `body_inertias`,
`lower_pos_limit`, `topo_order`, `parents`, `children`, `subtrees`,
`supports`) already reads well — no change.

### 2.6 Lie / spatial layer

Lie-algebra symbols (`Jr`, `hat`, `vee`, `ad`, `Ad`, `exp`, `log`) are
kept as-is but become **functions**, never bare identifiers on a tensor:

```python
from better_robot import lie
J = lie.right_jacobian_se3(xi)     # "Jr" in code → right_jacobian_se3 as the callable
```

Internal implementations may still use the short form in tight algebraic
blocks, but the **public API uses the English form** so IDE autocomplete
and help-text are readable. The short form is available as an alias:

```python
from better_robot.lie import Jr, Jr_inv    # allowed; documented
```

### 2.7 Optim / Tasks

Already readable; one addition for consistency:

| New name | Purpose |
|----------|---------|
| `IKResult` | keep |
| `TrajOptResult` | keep |
| `IKCostConfig` | rename from old `IKCostConfig` — already matches `<verb><noun>Cfg` pattern |
| `OptimizerConfig` | keep |
| `SolverState` | **new**, standardised per-iteration state object passed between `Optimizer`, `DampingStrategy`, `LinearSolver` (see [07_RESIDUALS_COSTS_SOLVERS.md](07_RESIDUALS_COSTS_SOLVERS.md)) |

## 3 · Glossary (in plain English)

| Term | Meaning | Appears in |
|------|---------|------------|
| **Model** | Immutable kinematic tree: joints, bodies, frames, inertias, limits. One per robot, shared across queries, moved with `.to(device, dtype)` | 02, 04 |
| **Data** | Mutable per-query workspace. Holds `q`, lazy kinematic/dynamic caches. One per batch × time evaluation | 02, 05, 06 |
| **Frame** | Any named coordinate frame on the robot — joint frames, body frames, user-declared operational frames | 02 |
| **Joint** | A single degree of articulation (revolute, prismatic, free-flyer, spherical, …). "Joint 0" is always the universe (world) | 02 |
| **Body** | A rigid link. Bodies are 1:1 with joints except joint 0; inertia is attached via `joint_placements` | 02 |
| **joint_pose_local** | SE(3) transform from a joint's parent joint to itself (was `liMi`) | 02, 05 |
| **joint_pose_world** | SE(3) transform from world origin to a joint (was `oMi`) | 02, 05 |
| **frame_pose_world** | SE(3) transform from world origin to a frame (was `oMf`) | 02, 05 |
| **joint_velocity_world** | 6D twist of each joint, linear first, in world axes (was `ov`) | 02, 06 |
| **spatial Jacobian** | 6 × nv Jacobian relating `v` to twist; `linear rows | angular rows` | 05 |
| **body-frame Jacobian** | Spatial Jacobian with the origin's twist re-expressed in the body's axes | 05 |
| **LOCAL_WORLD_ALIGNED** | Jacobian of a point on the body, *translated* to the world origin but *rotated* with world axes. Default for `get_frame_jacobian` | 05 |
| **Residual** | A differentiable function `r(model, data, …) -> (B..., dim)` — the quantity the optimizer drives toward zero | 07 |
| **CostStack** | Weighted concatenation of residuals; returns a single flat residual vector | 07 |
| **LeastSquaresProblem** | `(cost_stack, x0, bounds, jacobian_strategy)` — a fully specified optimization problem | 07 |
| **Optimizer** | A `Protocol` that minimises a `LeastSquaresProblem` (LM, GN, Adam, LBFGS, …) | 07 |
| **Rollout** | A callable that turns an action sequence into a trajectory + cost; for trajopt only | 08 |
| **bias_forces** | `C(q, q̇) q̇ + g(q)` — the generalised force present even at zero input torque (was `nle`) | 06 |
| **centroidal momentum** | 6D momentum of the robot around its centre of mass (was `hg`) | 06 |
| **mass matrix** | Joint-space inertia `M(q)` (was `M`) | 06 |
| **coriolis matrix** | `C(q, q̇)` such that `C(q, q̇) q̇` is the Coriolis/centrifugal torque (was `C`) | 06 |
| **gravity torque** | `g(q)` — joint-space generalised gravity (was `g`) | 06 |
| **Capsule** | Sphere-swept line segment; the default collision primitive for the robot surface | 09 |
| **Gizmo** | Draggable SE(3) widget in the viewer used to set IK targets | 12 |

## 4 · Renames at a glance (cheat sheet)

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

## 5 · Migration plan

The renames land in the skeleton phase (see [11_SKELETON_AND_MIGRATION.md §Rename sprint](11_SKELETON_AND_MIGRATION.md)), before any dynamics
body is written. Sequence:

1. **Data field rename.** Add new fields next to the old on `Data`; populate
   both in the kinematics layer. Tests still pass against old names.
2. **Downstream rewire.** Move every internal read to the new field names.
   The old names become `@property` shims that return the new tensor.
3. **Deprecation warning.** Wrap the old properties with
   `warnings.warn(..., DeprecationWarning)`.
4. **Release note + example update.** All `examples/*.py` switch to new
   names in the same release the warnings appear.
5. **Removal in v1.1.** One release window; no support thereafter.

### 5.1 Acceptance criteria

- Every file under `src/better_robot/` uses only the new names (grep).
- No external-looking identifier (top-level function, public attribute,
  public class name) begins with a lowercase `o`/`l` followed by a capital
  unless it is documented Lie-algebra notation (`oMi`, `liMi` etc. are
  **out**).
- Every residual, solver, and task docstring uses the new vocabulary.
- `docs/13_NAMING.md` (this file) contains no `[TODO]` marker.

## 6 · For contributors

When you add a new field to `Data`, a new attribute to `Model`, or a new
tensor that a user will read:

1. **Read §1.** Pick a name that follows `<entity>_<quantity>_<frame>` or
   the function-naming patterns.
2. **Check §1.3.** If the quantity has a universal math symbol, use it as a
   variable *inside* a function but expose the verbose name in any
   public-facing docstring or attribute.
3. **Add to §3.** Any genuinely new concept gets one row in the glossary.
4. **If the name is cryptic, it's wrong.** The reviewer's first question
   will be "what does this mean?" — if the answer is not obvious from the
   identifier, rename before merging.
