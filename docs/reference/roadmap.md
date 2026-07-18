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
- `src/better_robot/residuals/contact.py`
- `src/better_robot/residuals/regularization.py`
- `src/better_robot/residuals/smoothness.py`
- `src/better_robot/spatial/force.py`
- `src/better_robot/tasks/ik.py`
- `src/better_robot/tasks/trajopt.py`
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

These residual features remain unfinished:

- angular contact consistency;
- `NullspaceResidual`; and
- `JerkResidual`.

Use linear contact consistency, explicit regularization, and
`AccelerationResidual` for the supported cases. See
{doc}`/concepts/residuals_costs_and_solvers`.

## Tasks

`solve_ik` does not expose L-BFGS because its batched line-search and history
semantics are not implemented. Use LM, Gauss--Newton, Adam, or the supported
LM-then-Adam sequence.

`solve_trajopt` accepts `KnotTrajectory`. `BSplineTrajectory` remains a
Euclidean numerical utility; component interpolation is not a manifold-safe
robot trajectory and cannot enforce robot state bounds correctly.

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

## Finishing an entry

1. Define the public shape, dtype, device, batching, and gradient behavior.
2. Implement the operation without changing an unrelated signature.
3. Add focused success, failure, and parity tests.
4. Remove the explicit raise and its file from the marker block in the same
   patch.
5. Update the relevant concept, reference page, and changelog.
