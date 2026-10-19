# Roadmap

This page says what BetterRobot does today, what it will do next, and what it
deliberately does not do. It has two parts: a checked inventory of the
operations that still raise `NotImplementedError`, and a plan per theme.

## How to read the plan

Each theme table lists one *brick* (a self-contained piece of work) per row,
why it is wanted, what must exist first, and when it is planned:

- **now**: landed in this version (listed below, with the contract or ADR that
  records it);
- **next**: the following version; the design is settled;
- **later**: wanted, but waits for its prerequisite or a real in-tree caller;
- **maybe**: kept open; it starts only if a concrete need appears.

No brick starts without an owner decision. The API is not frozen during alpha,
so a brick may change a public signature; the [changelog](changelog.md) records it.

### Landed in this version (now)

- Identifiers after `(model, state)` are keyword-only everywhere, including
  terms; joint ids accept `int | str`; an unknown name raises
  `UnknownNameError` ([C-ERR-2](../contracts/06-validation-errors.md)).
- `io.parse` returns the IR, where composition lives; URDF/MJCF joint
  damping, friction and armature reach `ModelValues`; MJCF import is marked
  experimental with its known gaps ([model loading](../concepts/io.md)).
  `BackendNotAvailableError` is reserved for a future kernel brick
  ([dependencies and layering](../contracts/07-dependencies-layering.md)).
- `Trajectory`, a minimal frozen dataclass (`q (T, nq)`, `times (T,)`,
  optional `v (T, nv)`), and the SE(3) vocabulary placement / pose / transform
  ([C-NAME-5](../contracts/01-naming.md), [glossary](glossary.md)).
- Optimizer diagnostics in `State.extra`, `Result.diagnostics` and
  `Result.state`, warm start through `solve(..., state=)` / `run(..., state=)`,
  a warning when a `time=` problem falls back to dense, the normal-equation
  sink seam, and reserved `equality`/`inequality` term kinds
  ([extension points](../contracts/08-extension-points.md),
  [optimization](../concepts/optimization.md)).
- Compile-safe placement caches with `fullgraph=True` tests for FK, RNEA and
  CRBA; the derivatives contract
  ([ADR 0007](../contracts/decisions/0007-derivatives-contract.md)); the
  documented `_memo` keys ([C-STATE-4](../contracts/03-state-mutability.md));
  the `to_euler` gimbal-lock fix; a benchmark smoke test.
- The robot-geometry brick decision
  ([ADR 0008](../contracts/decisions/0008-geometry-brick.md));
  `self_collision_pairs(..., ignore=)`, `RobotSpheres.to()`, batched sphere
  offsets and the SDF outside-grid rule ([collision](../concepts/collision.md)).
- Viewer: optional per-layer sample `times` on one integer-step timeline
  ([C-NAME-4](../contracts/01-naming.md)), `Viewer.on_pointer` /
  `clear_pointer`, the picking helper `viewer.tools.picking.nearest_on_ray`,
  viser as the only backend with `server.gui` as the studio API
  ([ADR 0004](../contracts/decisions/0004-viser-viewer.md),
  [viewer](../concepts/viewer.md)).

## Complete explicit-raise inventory

Paths are relative to the repository root. Keep this block sorted; the
contract test reports both missing source files and stale documentation
entries.

<!-- not-implemented-inventory:start -->
- `src/better_robot/io/build_model.py`
- `src/better_robot/spatial/force.py`
<!-- not-implemented-inventory:end -->

## Current limits

### Dynamics and spatial algebra

`center_of_mass(model, state)` gives the position only; the centre-of-mass velocity is the linear part of
`centroidal_momentum(model, state, v=v)` divided by the total mass. Centre-of-mass acceleration is not provided.

`Force.cross_motion` raises because that operation is not a standard spatial
primitive. Use the documented motion/force cross operation instead.

### Model building

A mimic relationship needs the concrete motion type of its target, such as a
revolute, prismatic, or helical joint. A direct zero-width `mimic` joint
kind does not contain enough motion information and is rejected. Use a
supported scalar joint together with the mimic source, multiplier, and
offset.

### Collision today

Robots are approximated by spheres (`collision.RobotSpheres`) or capsules
(`collision.RobotCapsules`, from a table). Sphere-to-sphere, sphere-to-capsule,
capsule-to-capsule, sphere-to-box, sphere-to-half-space and sphere-to-SDF
distances are live, as are self-distances over `self_collision_pairs`; the
`terms.self_clearance` and `terms.sphere_clearance` terms build on them. Mesh-to-mesh
distance and convex decomposition are not provided.

## What is implemented

The following surfaces are live:

- `forward_kinematics_state(model, q)` returning an immutable `KinematicState`,
  and batched analytic joint and frame Jacobians and their time variation in
  all three references;
- inverse dynamics, the mass matrix, forward dynamics, bias forces, the
  centroidal map and momentum, and the center of mass, all reading the state;
- `optim.Problem` over named variables with plain-function terms; LM,
  Gauss--Newton, L-BFGS and a `torch.optim` adapter; dense, banded and arrow
  normal equations; staged solves with `Phase`; `solve_layer` and
  `solve_constrained`;
- fixed-base and floating-base IK (`terms.solve_ik`) and knot-based trajectory
  optimization; and
- URDF, MJCF, and programmatic model construction (`ModelBuilder`).

The [API reference](api/better_robot.md) is the exact signature source.

## Plan by theme

### Optimization

| Brick | Why | Prerequisite | When |
|---|---|---|---|
| Constraint term kinds (`equality`, `inequality`) on `Problem` | QP-IK, SQP, limits as hard constraints | vocabulary reserved now | next |
| Batch compaction: step only unfinished elements | multi-start IK | diagnostics in `State.extra` (now) | next |
| `BlockSparseSink` with block Cholesky / PCG in pure torch | multi-body fitting, long problems | sink seam (now) | next |
| One flattened batch axis inside solvers | simpler solver code, compile | batch-axis rule documented (now) | later |
| Schur complement beyond the arrow border; matrix-free normal route | trajectories too long for banded Cholesky | block-sparse sink | later |
| QP subproblem and SQP | constrained IK and trajectories | constraint term kinds | later |
| Shooting, DDP/iLQR and MPC | trajectory optimization, control | warm start (now), dynamics derivatives | later |
| Fixed-shape, `torch.compile`-friendly step | real-time MPC | compile-clean kinematics (now) | maybe |
| CHOLMOD or another external sparse backend | very large problems | block-sparse sink; a measured need | maybe |
| Forward-mode Jacobians when the tangent is narrower than the residual | point-heavy terms | a benchmark that beats reverse mode, which stays the default | maybe |

### Kinematics and dynamics

| Brick | Why | Prerequisite | When |
|---|---|---|---|
| `<quantity>_derivatives` through autograd | trajectory optimization, DDP | ADR 0007 (now) | next |
| Energy, Coriolis matrix, centre-of-mass velocity and acceleration, `compute_minverse`, centroidal derivatives | controllers, human dynamics | none | next |
| Named model-parameter accessors such as link mass | identification, editing | none | next |
| Contact Jacobians, including offset contact points with their `r × f` moment | legged robots, human contact | contact frames on the model | next |
| Analytic RNEA/ABA derivatives (Carpentier–Mansard) behind the same signatures | speed | autograd derivatives and parity tests | later |
| O(n) ABA; loop closures | long chains, musculoskeletal models | none | later |
| Force-plate / ground-reaction input feeding `fext` | gait biomechanics | contact Jacobians | later |
| Warp kernel backend behind `BackendNotAvailableError` | throughput | a measured bottleneck | maybe |
| Constrained dynamics (KKT forward dynamics, impulses) and its derivatives | contact-implicit trajectory optimization | contact Jacobians | far future |

The target is batch-first; compile-clean code is the route to fast batch-1 calls.
Biomechanical joints (for example coupled knee or scapula joints) live in
BetterHuman as extensions; BetterRobot stays generic.

### Collision

| Brick | Why | Prerequisite | When |
|---|---|---|---|
| `collision.RobotGeometry` built from the IR, meshes through lazy `trimesh` | geometry for every new robot | ADR 0008 (now) | next |
| Capsule robots read from the IR (`RobotCapsules` takes a table today) | humanoids, thin links | `RobotGeometry` | next |
| `fit_spheres(geometry)`; sampled collision matrix and SRDF import | new robots without hand-made spheres | `RobotGeometry` | next |
| Closest point and normal as separate functions; generic contact for dynamics | contact terms and contact Jacobians | none | next |
| `terms.swept_clearance` and `collision.Scene` | trajectories, planning | closest-point functions | next |
| Attached objects; ESDF from a mesh; active-set pairs | pick-and-place, speed | `collision.Scene` | later |
| GJK/EPA in torch; a fused GPU kernel | exact contact, throughput | a measured need | maybe |

### Viewer

| Brick | Why | Prerequisite | When |
|---|---|---|---|
| Tool manager for pointer tools | studios with several tools | `Viewer.on_pointer` (now) | next |
| Panel brick (`Viewer.add_panel`), lazy layer sources, layer parenting | human studio, long clips | none | next |
| Lines, labels and a wrench layer | dynamics users | none | next |
| Follow camera, source comparison, clip browser, measurement tool | demos, ground truth against estimate | panel brick | later |
| Scene export to `.npz` (the only export format) | sharing results | none | later |
| Video panel with reprojection, synced through per-layer `times` | video reconstruction | per-layer `times` (now) | later |

viser stays the only backend ([ADR 0004](../contracts/decisions/0004-viser-viewer.md)); no own three.js client is planned.

### Model building and tools

| Brick | Why | Prerequisite | When |
|---|---|---|---|
| IR editing: attach, lock joints, add frame | grippers, tools, collision | `io.parse` returns the IR (now) | next |
| SRDF import | collision matrices, groups | IR editing | next |
| MJCF import beyond experimental (gaps listed in [model loading](../concepts/io.md)) | MuJoCo models | parity tests per element | later |
| Manifold-safe, bounds-aware B-splines and retiming | smooth trajectories | a designed B-spline contract | later |
| URDF/MJCF writers | round-trips | IR editing | later |
| Control and simulation helpers | RL, closed loop | dynamics derivatives | later |
| External benchmarks against cuRobo-class libraries; CI on push and a GPU runner | trust, regressions | benchmark smoke test (now) | later |
| Calibration | real robots | IR editing | maybe |

## Finishing an entry

1. Define the public shape, dtype, device, batching, and gradient behavior.
2. Implement the operation without changing an unrelated signature.
3. Add focused success, failure, and parity tests.
4. Remove the explicit raise and its file from the marker block in the same
   patch.
5. Update the relevant concept, reference page, and changelog.
