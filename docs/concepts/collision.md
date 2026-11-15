# Collision is signed distance between spheres and shapes

After this page you know how a robot becomes spheres and capsules on its frames, what the sign of a distance
means, which link pairs self-collision skips, and how distances become clearance terms. The code lives in
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
from better_robot import ModelBuilder, collision, forward_kinematics, optim, terms
from better_robot.io import build_model

builder = ModelBuilder("arm", gravity=[0.0, 0.0, -9.81])
for body in ("base", "link1", "link2"):
    builder.add_body(body, mass=1.0)
builder.add_revolute("j1", axis=(0.0, 0.0, 1.0), parent="base", child="link1")
builder.add_revolute("j2", axis=(0.0, 1.0, 0.0), parent="link1", child="link2",
                       origin=torch.tensor([0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0]))
builder.add_frame("tool", parent="link2",
                  placement=torch.tensor([0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 1.0]))
model = build_model(builder.finalize(), dtype=torch.float64)

spheres = collision.RobotSpheres.from_frames(model, {
    "base": [((0.0, 0.0, 0.05), 0.06)],
    "link2": [((0.0, 0.0, 0.12), 0.04)],
    "tool": [((0.0, 0.0, 0.0), 0.03)],
})
state = forward_kinematics(model, torch.tensor([0.0, 0.5], dtype=torch.float64))
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
| `sphere_sphere` | another sphere |
| `sphere_capsule` | a capsule around a segment |
| `sphere_halfspace` | the solid `{x : normal . x <= offset}` |
| `sphere_box` | an oriented box |
| `sphere_sdf` | a sampled signed distance field |

Their signatures and shapes are in the [collision reference](../reference/api/collision.md).

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

The sphere on `link2` and the `tool` sphere are on the same body, so they form no pair. The base
and `link1` are joined by `j1`, so they would form none either.

## Sphere sets for the Panda and the G1 ship with the package

`load_spheres(path, model)` reads a sphere file: JSON whose `"spheres"` maps a frame name to rows
`[x, y, z, radius]` (the centre in that frame's coordinates) and whose `"ignore"` lists the frame pairs to pass to
`self_collision_pairs`, those whose spheres overlap at the neutral configuration beyond the pairs the function drops
itself. `SPHERE_SETS` is the folder of the files that ship with the package: `panda.json` (42 spheres) and `g1.json`
(91 spheres, for the G1 with 29 joints).

```python
import torch
from robot_descriptions import panda_description
from better_robot import collision, forward_kinematics, load

panda = load(panda_description.URDF_PATH, dtype=torch.float64)
spheres, ignore = collision.load_spheres(collision.SPHERE_SETS / "panda.json", panda)
pairs = collision.self_collision_pairs(panda, spheres, ignore=ignore)
distances = collision.self_distances(spheres, forward_kinematics(panda, panda.q_neutral), pairs=pairs)
print(spheres, tuple(pairs.shape), ignore[0])
print(bool((distances > 0).all()))   # every pair kept is clear at the neutral configuration
```
```output
RobotSpheres(42 spheres, float64, cpu) (535, 2) ('panda_hand', 'panda_link5')
True
```

The spheres of a body hold all its collision shapes: no surface point lies more than 0.5 mm outside them. The G1's
feet and hands hold their display meshes instead, since its description gives the feet four 5 mm contact spheres and
the hands no collision shape. Holding the shapes costs overshoot: the spheres' surface lies on average 8 to 43 mm
beyond the shapes, depending on the body. So pairs that are close at the neutral configuration overlap and are
ignored. On the Panda these are `panda_link5` with `panda_link7`, the hand and each finger, and the two fingers
(the meshes of link 5, link 7 and the hand touch there, as do the closed fingers); on the G1 they include the
shoulder and elbow links with the torso, whose meshes are 9 to 37 mm apart there. Pass your own `ignore` to keep
such a pair. `tools/fit_spheres.py` fits a file for another robot or sphere count; its
docstring says how.

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
    "base": [((0.0, 0.0, 0.0), (0.0, 0.0, 0.1), 0.05)],
    "link2": [((0.0, 0.0, 0.05), (0.0, 0.0, 0.25), 0.03)],
})
capsule_pairs = collision.self_collision_pairs(model, capsules)
print(collision.self_capsule_distances(capsules, state, pairs=capsule_pairs))   # base top to link2's lower end
```
```output
tensor([0.1651], dtype=torch.float64)
```

## Clearance terms turn distances into residuals

`terms.sphere_clearance` and `terms.self_clearance` turn these distances into residuals `margin - distance`,
hinged at zero so a clear configuration costs nothing, or left unhinged as an inequality for
`optim.solve_constrained`. The hinge has no jump in its slope: for a gap `margin - distance` between 0 and `margin`
the residual is `gap² / (2 margin)`, beyond it `gap - margin / 2`, so the Gauss-Newton model does not switch on at
once when a sphere enters the margin. [Keep a trajectory clear of obstacles](../guides/avoid-collisions.md) bends a
path around an obstacle with them.

Both terms have analytic Jacobians: the gradient of each distance with respect to its sphere centre times
`kinematics.point_jacobian`. For `self_clearance` the gradient is the unit vector between the two centres; for
`sphere_clearance` it is the reverse-mode derivative of `distance_fn` alone, so any shape gets it, including a
sampled SDF and a function of your own. A robot with many spheres has thousands of pairs:
`self_clearance(..., closest=k)` keeps the `k` nearest pairs of each configuration, a fixed number of rows.

## Extending collision

A new obstacle type is a new `distance_fn(centers: (B..., S, 3), radii: (S,)) -> (B..., S)` or `(B..., S, K)`
for `sphere_clearance`, entry `s` reading centre `s` only and transformable by `torch.func`; the term and the
solvers do not change (contract [08, collision section](../contracts/08-extension-points.md), which also gives the
autodiff route for any other function). Robot geometry built from the IR (capsules read
from a model file, shapes fitted to meshes), closest points and contact follow
record [0016: geometry is a value, not a field of `Model`](../contracts/decisions/0016-geometry-out-of-model.md).

## Where next

- [Optimization](optimization.md): how the clearance residual joins a `Problem` and which solver runs it.
- [Keep a trajectory clear of obstacles](../guides/avoid-collisions.md): a clearance term in a trajectory problem.
- [Hard constraints](../guides/hard-constraints.md): enforce the margin exactly with `solve_constrained`.
- [Viewer](viewer.md): show spheres, obstacles and the optimized trajectory.
