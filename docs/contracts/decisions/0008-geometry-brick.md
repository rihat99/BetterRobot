# 0008 Geometry is a separate brick

**Status:** Accepted · **Date:** 2026-10-08

## Context
`collision` today approximates the robot by `RobotSpheres` built by hand, and returns scalar signed distances.
The URDF and MJCF loaders already parse visual and collision geometry into the IR (`IRBody.visual_geoms`,
`IRBody.collision_geoms`, `IRGeom`), and the built `Model` keeps the IR in `model.meta["ir"]`, where only the
viewer reads it. Planned work (capsule robots, mesh-fitted primitives, contact for dynamics, BetterHuman bodies)
needs one place that owns robot geometry, so it does not grow inside `Model`, the viewer or each caller.

## Decision
We will add geometry as its own brick, in the style of Pinocchio's `GeometryModel` next to `Model`:

- `collision.RobotGeometry` is built from the IR (`RobotGeometry.from_ir(ir, model)` or from a `Model` that
  carries one), separate from `Model`; kinematics and dynamics never import it.
- Spheres and capsules are the robot primitives, each attached to a frame with a fixed placement. Boxes,
  half-spaces and SDF grids stay environment shapes.
- Mesh to primitive fitting may import `trimesh` lazily at that one boundary (`C-DEP-2`); nothing else needs it.
- Distances stay scalar (`(B..., P)` signed distances). Closest points and normals are separate `*_closest`
  functions with the same arguments, so the scalar path stays cheap and differentiable.
- Contact for dynamics (contact frames, normals and penetration for contact-aware RNEA/ABA or constraints) lives
  in BetterRobot and is generic; human-specific contact stays in BetterHuman.
- Until `RobotGeometry` lands, the geometry schema in `model.meta["ir"]` (`IRGeom` fields and the
  `visual_geoms`/`collision_geoms` lists) is frozen: loaders may fill more geometry, not rename or reshape it.

## Consequences
- `RobotSpheres` remains the supported robot model until `RobotGeometry` replaces or wraps it; its
  `frame_ids`/`offsets`/`radii` layout is what a sphere-only `RobotGeometry` lowers to.
- The viewer keeps reading `meta["ir"]` meanwhile; when `RobotGeometry` lands it reads that instead.
- Extending distances (new primitive pairs, closest points) is an additive change in `collision.distance`.

## Alternatives considered
- Geometry inside `Model` (MuJoCo style): couples topology, values and shapes; every `Model` copy carries meshes.
- Meshes as the primary representation: distances are not smooth and need a mesh library in the core.
- Closest points from every distance call: doubles the cost of the common scalar path.

## Enforcement
- `C-DEP-1` layering (`tests/contract/test_layer_dependencies.py`): kinematics and dynamics do not import
  `collision`.
- `C-DEP-2`: `trimesh` stays a lazy import inside the fitting function.
- [08 Extension points](../08-extension-points.md), collision section: the `distance_fn` protocol.
