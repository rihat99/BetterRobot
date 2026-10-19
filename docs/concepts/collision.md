# Collision is signed distance between spheres and shapes

After this page you can attach spheres or capsules to a robot's frames, compute signed distances to obstacles
and between the robot's own links, and add clearance terms to an optimization problem. The code lives in
`better_robot.collision` (geometry) and `better_robot.terms` (residual terms built on it).

## The robot is a set of spheres fixed to frames

`RobotSpheres` approximates the robot by `S` spheres. Sphere `s` sits at `offsets[s]` in the
coordinates of frame `frame_ids[s]` and has radius `radii[s]`: `frame_ids (S,)`, `offsets (S, 3)`,
`radii (S,)`. Build it with `RobotSpheres.from_frames`, which takes frame names or ids (see
[Joints, bodies and frames](joints-bodies-frames.md)) and matches the model's dtype and device;
`spheres.to(device=..., dtype=...)` moves it (frame ids stay `long`). `offsets` may also carry batch axes,
`(B..., S, 3)`, for per-sample sphere placement (for example a body shape per batch row): they broadcast
against the state's batch axes in `centers`, while `frame_ids` and `radii` stay unbatched.

```python
import torch
from better_robot import collision, optim, terms
from better_robot.io import ModelBuilder, build_model
from better_robot.kinematics import forward_kinematics_state

builder = ModelBuilder("arm", gravity=[0.0, 0.0, -9.81])
for body in ("base", "link1", "link2"):
    builder.add_body(body, mass=1.0)
builder.add_revolute_z("j1", parent="base", child="link1")
builder.add_revolute_y("j2", parent="link1", child="link2",
                       origin=torch.tensor([0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0]))
builder.add_frame("tool", parent_body="link2",
                  placement=torch.tensor([0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 1.0]))
model = build_model(builder.finalize(), dtype=torch.float64)

spheres = collision.RobotSpheres.from_frames(model, {
    "body_base": [((0.0, 0.0, 0.05), 0.06)],
    "body_link2": [((0.0, 0.0, 0.12), 0.04)],
    "tool": [((0.0, 0.0, 0.0), 0.03)],
})
state = forward_kinematics_state(model, torch.tensor([0.0, 0.5], dtype=torch.float64))
centers = spheres.centers(state)
print(centers.shape)   # torch.Size([3, 3]): (B..., S, 3), in WORLD
```
```output
torch.Size([3, 3])
```

`centers(state)` reads the frame poses from a `KinematicState`, so it batches and differentiates like
forward kinematics (see [Kinematics and Jacobians](kinematics-and-jacobians.md)).

## Distances are signed

Every distance function returns one signed scalar per query `(...)`: positive is the gap between
surfaces, zero is touching, negative is penetration depth. Gradients stay finite when two centres
coincide.

| Function | Distance from a sphere to |
|----------|---------------------------|
| `sphere_sphere(center_a, radius_a, *, center_b, radius_b)` | another sphere |
| `sphere_capsule(center, radius, *, start, end, capsule_radius)` | a capsule around a segment |
| `sphere_halfspace(center, radius, *, normal, offset)` | the solid `{x : normal . x <= offset}` |
| `sphere_box(center, radius, *, box_center, box_rotation, half_extents)` | an oriented box |
| `sphere_sdf(center, radius, *, grid, origin, voxel_size)` | a sampled signed distance field |

`sphere_sdf` interpolates the grid trilinearly. A centre outside the grid is clamped to the boundary, so the
distance there is the boundary value and its gradient with respect to the centre is zero: a clearance term
cannot push a sphere back into the grid from outside. Make the grid cover the whole workspace.

```python
floor = collision.sphere_halfspace(centers, spheres.radii,
                                   normal=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64), offset=0.0)
print(floor)   # tensor([-0.0100, 0.3653, 0.4894]): the base sphere dips 1 cm below z = 0
```
```output
tensor([-0.0100,  0.3653,  0.4894], dtype=torch.float64)
```

## Self-collision skips pairs that touch by construction

`self_collision_pairs(model, spheres)` lists the sphere pairs `(P, 2)` that can move relative to each
other. It drops pairs on the same rigid body (frames joined only by fixed joints) and pairs on bodies
joined by one moving joint, which touch at the joint by design. `ignore=[(frame_a, frame_b), ...]` (names
or ids, either order) drops more frame pairs, such as links whose spheres overlap at every reachable `q`.
`self_distances` returns `(B..., P)`.

```python
pairs = collision.self_collision_pairs(model, spheres)
print(pairs)   # tensor([[0, 1], [0, 2]]): base vs. link2 and base vs. tool
print(collision.self_distances(spheres, state, pairs=pairs))
```
```output
tensor([[0, 1],
        [0, 2]])
tensor([0.2599, 0.3945], dtype=torch.float64)
```

The sphere on `body_link2` and the `tool` sphere are on the same body, so they form no pair. The base
and `link1` are joined by `j1`, so they would form none either.

## Capsules cover long links with few primitives

`RobotCapsules` approximates the robot by `C` capsules: capsule `c` is every point within `radii[c]` of the
segment from `starts[c]` to `ends[c]`, both in the coordinates of frame `frame_ids[c]` (`frame_ids (C,)`,
`starts, ends (C, 3)`, `radii (C,)`); equal end points make a sphere. `segments(state)` returns the end points
in `WORLD`, `(B..., C, 3)` each. `capsule_capsule(start_a, end_a, *, radius_a, start_b, end_b, radius_b)` is the
signed distance between two capsules from the closest points of their segments (Ericson's algorithm without
branches; parallel and zero-length segments included). Its gradient is exact wherever the closest points are
unique: the distance is stationary in the segment parameters there, so they are detached. `self_collision_pairs`
takes capsules with the same rules as spheres, and `self_capsule_distances` returns `(B..., P)`.

```python
capsules = collision.RobotCapsules.from_frames(model, {
    "body_base": [((0.0, 0.0, 0.0), (0.0, 0.0, 0.1), 0.05)],
    "body_link2": [((0.0, 0.0, 0.05), (0.0, 0.0, 0.25), 0.03)],
})
capsule_pairs = collision.self_collision_pairs(model, capsules)
print(collision.self_capsule_distances(capsules, state, pairs=capsule_pairs))   # base top to link2's lower end
```
```output
tensor([0.1651], dtype=torch.float64)
```

## Clearance terms go into a problem

`terms.sphere_clearance(spheres, distance_fn, margin=...)` builds a residual `margin - distance` for
every sphere and obstacle; `terms.self_clearance(spheres, pairs=pairs, margin=...)` does the same for sphere
pairs. Both are hinged at zero by default, so a clear configuration costs nothing. With `hinge=False`
the residual is the inequality `margin - distance <= 0` for `optim.solve_constrained`. Both read the
cached kinematic state named `"state"`.

The example below swings `j1` from -0.9 to 0.9 rad. The straight path drives the tool through an
obstacle; a trajectory problem bends it around. [Optimization](optimization.md) explains `Problem`,
`cached`, `window` and the solvers; `examples/06_obstacle_trajectory.py` is the full Panda version with
both soft and hard constraints.

```python
knots, dt, margin = 12, 0.1, 0.02
q_mid = torch.tensor([0.0, 0.5], dtype=torch.float64)
obstacle = forward_kinematics_state(model, q_mid).frame_translation[model.frame_id("tool")].clone()
obstacle_radius = torch.tensor(0.05, dtype=torch.float64)

def to_obstacle(centers, radii):
    return collision.sphere_sphere(centers, radii, center_b=obstacle, radius_b=obstacle_radius)

def clearance(q):
    return float(to_obstacle(spheres.centers(forward_kinematics_state(model, q)), spheres.radii).min())

alpha = torch.linspace(0, 1, knots, dtype=torch.float64)[:, None]
straight = (1 - alpha) * torch.tensor([-0.9, 0.5]) + alpha * torch.tensor([0.9, 0.5])
end_weight = torch.zeros(knots, 1, dtype=torch.float64)
end_weight[0] = end_weight[-1] = 1e3

problem = optim.Problem(
    {"q": straight[None].clone()},
    manifolds={"q": terms.RobotManifold(model)},
    data={"ends": straight.clone()[None]},
    time=("q",),
)
problem.cached(terms.kinematic_state(model), name="state")
problem.residual("ends", lambda q, ends: q - ends, weight=end_weight, window=1)
problem.residual("velocity", terms.finite_difference(model, order=1, dt=dt), weight=1e-2)
problem.residual("clearance", terms.sphere_clearance(spheres, to_obstacle, margin=margin),
                 weight=1e2, window=1)

result = optim.solve(problem, optim.LevenbergMarquardt(), iters=100)
print(f"straight {clearance(straight):+.3f} m, optimized {clearance(result.x['q'][0]):+.3f} m")
# straight -0.070 m, optimized +0.019 m
assert clearance(result.x["q"][0]) > clearance(straight)
```
```output
straight -0.070 m, optimized +0.019 m
```

The clearance before optimization is negative: the tool sphere passes through the obstacle. After the
solve it sits just under the 0.02 m margin, because the soft weight trades clearance against smoothness. Raise the weight or switch to the
hard inequality to enforce it exactly.

<figure markdown="span"><video autoplay loop muted playsinline preload="metadata" poster="../../clips/obstacle_trajectory.png" width="960"><source src="../../clips/obstacle_trajectory.webm" type="video/webm"></video><figcaption>The Panda, modelled as spheres, follows a trajectory that a clearance term bends around a spherical obstacle (<code>examples/06_obstacle_trajectory.py</code>).</figcaption></figure>

## Extending collision

A new obstacle type is a new `distance_fn(centers: (B..., S, 3), radii: (S,)) -> (B..., S)` or `(B..., S, K)`
for `sphere_clearance`; the term and the solvers do not change (contract
[08, collision section](../contracts/08-extension-points.md)). Robot geometry built from the IR (capsules read
from a model file, shapes fitted to meshes), closest points and contact follow
[ADR 0008: geometry is a separate brick](../contracts/decisions/0008-geometry-brick.md).

## Where next

- [Optimization](optimization.md): how the clearance residual joins a `Problem` and which solver runs it.
- [Hard constraints](../guides/hard-constraints.md): enforce the margin exactly with `solve_constrained`.
- [Viewer](viewer.md): show spheres, obstacles and the optimized trajectory.
