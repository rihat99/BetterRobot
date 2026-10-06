# Collision and geometry

`better_robot.collision` is a minimal, functional collision layer: signed
distances between a sphere and simple shapes, and robots approximated by
spheres attached to frames. Everything is batched, differentiable, and torch
only. The generated API reference contains the exact signatures.

## Distance functions

All functions return a signed distance with leading batch axes; negative
values mean penetration.

| Function | Obstacle |
|---|---|
| `sphere_sphere(center_a, radius_a, center_b, radius_b)` | another sphere |
| `sphere_capsule(center, radius, start, end, capsule_radius)` | a capsule around a segment (zero length allowed) |
| `sphere_halfspace(center, radius, normal, offset)` | the solid `{x : normal · x <= offset}` |
| `sphere_box(center, radius, box_center, box_rotation, half_extents)` | an oriented box; inside, the distance to the nearest face |
| `sphere_sdf(center, radius, grid, origin, voxel_size)` | a sampled signed distance field, trilinear lookup, clamped at the grid border |

Gradients are finite at contact. They are exactly zero in measure-zero
configurations (a centre on a capsule axis, two coincident centres, a centre
at the middle of a cube) and outside an SDF grid, where the value is held at
the border.

## Robot spheres

`RobotSpheres.from_frames(model, {frame: [(offset, radius), ...]})` attaches
spheres to frames; `spheres.centers(state)` returns their world centres
`(B..., S, 3)` from a `KinematicState`. `self_collision_pairs(model, spheres)`
lists the sphere pairs that can move relative to each other (spheres on one
rigid body — frames joined only by fixed joints — and on bodies joined by one
moving joint are excluded); `self_distances(spheres, state, pairs)` evaluates
them.

## Optimization terms

`better_robot.terms.sphere_clearance` and `self_clearance` turn these
distances into residuals for `better_robot.optim2`: a hinge
`max(margin - d, 0)` for penalties, or the plain `margin - d` as an inequality
for `solve_constrained`. See `examples/optim2/06_obstacle_trajectory.py`.

## What is not shipped

There is no mesh collision, broad phase, BVH, capsule fitting from meshes, or
collision-specific asset loader. The viewer can resolve visual meshes, but the
collision package does not consume them.
