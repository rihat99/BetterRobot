# Roadmap

This page lists public or importable operations that still raise
`NotImplementedError`. The file list is checked against the source tree.

The inventory is file-level: one file may contain several unfinished
operations. A file's absence means only that it has no explicit raise of that
type. It does not promise every imaginable feature in that area.

## Complete explicit-raise inventory

Paths are relative to the repository root. Keep this block sorted; the
contract test reports both missing source files and stale documentation
entries.

<!-- not-implemented-inventory:start -->
- `src/better_robot/collision/closest_pts.py`
- `src/better_robot/collision/geometry.py`
- `src/better_robot/collision/pairs.py`
- `src/better_robot/collision/robot_collision.py`
- `src/better_robot/dynamics/centroidal.py`
- `src/better_robot/io/build_model.py`
- `src/better_robot/spatial/force.py`
<!-- not-implemented-inventory:end -->

## Collision

Collision primitives are usable as tensor containers. Distance evaluation,
closest-point helpers, robot decomposition, collision penalties, and
collision-aware tasks are unfinished. See
{doc}`collision_and_geometry` for the exact boundary.

## Dynamics and spatial algebra

`center_of_mass` computes position and, when velocity is supplied, velocity.
Passing acceleration is not supported.

`Force.cross_motion` raises because that operation is not a standard spatial
primitive. Use the documented motion/force cross operation instead.

## Model building

A mimic relationship needs the concrete motion type of its target, such as a
revolute, prismatic, or helical joint. A direct zero-width `mimic` joint
kind does not contain enough motion information and is rejected. Use a
supported scalar joint together with the mimic source, multiplier, and
offset.

## Residuals

Every exported residual is live. Third-derivative smoothness, task-space
nullspace regularization, and angular contact consistency remain future
design work rather than placeholder API. See
{doc}`/concepts/residuals_costs_and_solvers` for the supported families.

## Tasks

`solve_ik` does not expose L-BFGS because its batched line-search and history
semantics are not implemented. Use LM, Gauss--Newton, Adam, or the supported
LM-then-Adam sequence.

Trajectory optimization is knot-based. A B-spline parameterization remains
deferred because component interpolation is not manifold-safe and cannot
enforce robot state bounds correctly.

`solve_contact_forces` applies each fitted force at the selected joint origin
as `[force, torque=0]`. It does not model an arbitrary offset contact point;
an offset `r` would contribute the additional moment `r × force`.

## What is implemented

The following nearby surfaces are live:

- FK, frame placement, analytic spatial Jacobians, and batched execution;
- RNEA, ABA, CRBA, CCRBA, centroidal momentum, and center-of-mass position;
- variable-based least-squares problems, LM, Gauss--Newton, a
  `torch.optim` adapter, and dense or declared temporal linearization;
- fixed-base and floating-base IK;
- knot-based trajectory optimization; and
- URDF, MJCF, and programmatic model construction.

The generated {doc}`api/better_robot/better_robot` reference is the exact
signature source.

## Deferred directions

Larger directions that are deliberately not in progress. Each entry names its
precondition; none is started without an owner decision.

- **Differentiable optimization as a module.** A `TheseusLayer`-style
  `nn.Module` wrapping a whole solve, with backward-mode selection (unrolled,
  implicit, truncated). The object-owned variables, detached `optimize()`, and
  guarded implicit mode are the prepared substrate.
- **Residual vectorization.** Grouping structurally identical residual
  instances into one batched evaluation instead of N Python calls. Only
  worthwhile for problems with many small residuals; measure first.
- **Warp kernels beyond FK and RNEA.** Frame Jacobians, integrate/difference,
  and whole-formulation kernels; a persistent CUDA-graph-captured solver
  driver. Each kernel lands only with parity, gradcheck, and timing evidence.
- **Optimizer extensions.** Matrix-free normal route for trajectories too long
  for banded Cholesky; banded/operator implicit backward; Schur elimination
  for trajectory-plus-shared blocks; batched per-element line-search L-BFGS.
  Each waits for a demonstrated in-tree need.
- **Dynamics derivatives and accessors.** `compute_minverse`, the Coriolis
  matrix, centroidal derivatives, analytic RNEA/ABA derivatives
  (Carpentier--Mansard), offset contact points with their `r × f` moment,
  angular contact consistency, and named model-parameter accessors such as
  link mass.
- **Residual families.** Yoshikawa manipulability, nullspace regularization,
  jerk smoothness, and acceleration limits; each needs its contract designed
  first.
- **Collision.** The package is a stub. The pending decision is to port a real
  implementation or cut the package; a Torch oracle must exist before any
  collision kernel.
- **Trajectory representations.** Manifold-safe, bounds-aware B-splines. No
  B-spline task surface ships until that contract is designed.
- **External benchmarks and infrastructure.** Comparisons against
  cuRobo-class libraries; CI on push and a GPU CI runner; viewer extras such
  as recording and overlay traces.

## Finishing an entry

1. Define the public shape, dtype, device, batching, and gradient behavior.
2. Implement the operation without changing an unrelated signature.
3. Add focused success, failure, and parity tests.
4. Remove the explicit raise and its file from the marker block in the same
   patch.
5. Update the relevant concept, reference page, and changelog.
