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
- `src/better_robot/dynamics/centroidal.py`
- `src/better_robot/io/build_model.py`
- `src/better_robot/spatial/force.py`
<!-- not-implemented-inventory:end -->

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

## Collision

Robots are approximated by spheres (`collision.RobotSpheres`). Sphere-to-sphere,
sphere-to-capsule, sphere-to-box, sphere-to-half-space and sphere-to-SDF
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

## Deferred directions

Larger directions that are deliberately not in progress. Each entry names its
precondition; none is started without an owner decision.

- **Optimizer extensions.** A matrix-free normal route for trajectories too
  long for banded Cholesky, and Schur elimination beyond the arrow border.
  Each waits for a demonstrated in-tree need.
- **Dynamics derivatives and accessors.** `compute_minverse`, the Coriolis
  matrix, centroidal derivatives, analytic RNEA/ABA derivatives
  (Carpentier--Mansard), offset contact points with their `r × f` moment,
  and named model-parameter accessors such as link mass.
- **Trajectory representations.** Manifold-safe, bounds-aware B-splines. No
  B-spline surface ships until that contract is designed.
- **External benchmarks and infrastructure.** Comparisons against
  cuRobo-class libraries; CI on push and a GPU CI runner.

## Finishing an entry

1. Define the public shape, dtype, device, batching, and gradient behavior.
2. Implement the operation without changing an unrelated signature.
3. Add focused success, failure, and parity tests.
4. Remove the explicit raise and its file from the marker block in the same
   patch.
5. Update the relevant concept, reference page, and changelog.
