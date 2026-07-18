# Collision and geometry status

`better_robot.collision` currently provides tensor containers and reserved
function signatures. It does not yet provide collision-distance computation
or collision-aware IK.

This page is a status reference so callers can distinguish usable data types
from unfinished operations. The generated API reference contains the exact
signatures.

## Available containers

The package exports five frozen dataclasses:

| Type | Fields |
|---|---|
| `Sphere` | `center (..., 3)`, `radius (...,)` |
| `Capsule` | endpoints `a (..., 3)`, `b (..., 3)`, and `radius (...,)` |
| `Box` | `center (..., 3)`, `half_extents (..., 3)`, `rotation (..., 4)` |
| `HalfSpace` | `normal (..., 3)`, `offset (...,)` |
| `Plane` | the same fields as `HalfSpace` |

These classes store tensors but do not validate shape, dtype, device,
normalization, or broadcasting at construction. They can be used as data
containers only.

## Unfinished operations

The following public operations raise `NotImplementedError`:

- `distance(a, b)`;
- `colldist_from_sdf(d, margin)`;
- `RobotCollision.from_model(...)`;
- `RobotCollision.world_capsules(data)`;
- `RobotCollision.self_distances(data)`; and
- `RobotCollision.world_distances(data, world)`.

The closest-point helpers `point_to_segment` and
`segment_to_segment` are also unfinished.

`register_pair(type_a, type_b)` does store a decorated function in the
package's private table. `distance` does not use that table yet. Registration
therefore does not make a working third-party distance implementation. It
stores only the exact ordered type pair and does not infer symmetry.

## Robot collision container

`RobotCollision` is an importable mutable dataclass with:

- frame identifiers;
- local capsule endpoints and radii;
- a self-pair index tensor; and
- an allowed-pair mask.

Its constructors and query methods are unfinished. It does not inspect
`Model.meta`, fit capsules from meshes, or update itself from FK data.

## What is not shipped

There is no broad phase, BVH, spatial hash, mesh fitting, collision-specific
asset loader, self-collision residual, world-collision residual, or task
integration. `solve_ik` does not accept collision geometry.

URDF and MJCF parsing may preserve geometry metadata for other consumers. The
viewer can resolve visual meshes, but the collision package does not consume
those meshes.

The complete file-level list of explicit unfinished operations is in
{doc}`roadmap`. Implemented residuals and solvers are described in
{doc}`/concepts/residuals_costs_and_solvers`.
