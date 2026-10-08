# P1 collision: adversarial architecture review

Scope: `src/better_robot/collision` (207 lines), `src/better_robot/terms/collision.py` (54 lines), geometry in
`io` and `data_model`, `docs/concepts/collision.md`, `docs/reference/roadmap.md` (Collision). Worktree head `23070f4`.
Probe script: `/tmp/claude-1003/-home-rikhat-akizhanov-better/d9220856-907e-4a64-b1a1-f6cd72ec6355/scratchpad/p1_coll_probe.py`.

## Current state (facts, with paths)

- Robot geometry is spheres only: `collision.RobotSpheres` is a frozen dataclass holding `frame_ids (S,)`,
  `offsets (S, 3)` and `radii (S,)`, built by hand with `from_frames(model, {frame: [(xyz, r), ...]})`
  (`collision/robot_spheres.py:18-55`). `centers(state)` gathers frame poses from a `KinematicState`.
- Distances are five free functions of the form "sphere vs X": `sphere_sphere`, `sphere_capsule`,
  `sphere_halfspace`, `sphere_box`, `sphere_sdf` (`collision/distance.py`). They return a signed scalar `(...)`, use
  a safe norm (`clamp_min(1e-24)`) and broadcast over leading axes. There is no closest point, normal or witness output.
- `self_collision_pairs` drops pairs on the same rigid body and on parent/child bodies, nothing else; it is a Python
  O(S^2) loop (`robot_spheres.py:64-88`). `self_distances` evaluates every listed pair densely.
- Obstacles have no representation. `terms.sphere_clearance` takes a closure
  `distance_fn(centers (B..., S, 3), radii (S,)) -> (B..., S) | (B..., S, K)` (`terms/collision.py:15-35`).
  Residual is `margin - distance`, hinged with `clamp_min(0)` or left raw for `solve_constrained`.
- Parsers already extract collision geometry into the IR: URDF via yourdfpy (`io/parsers/urdf.py:40-125, 199-208`),
  MJCF sphere/capsule/cylinder/box only (`io/parsers/mjcf.py:93-111`; mesh, plane, ellipsoid, hfield skipped;
  `contype`/`conaffinity`/`<exclude>` ignored). `IRBody.collision_geoms` (`io/ir.py:53-70`).
- The typed `Model` does not carry that geometry. It survives only as `model.meta["ir"]` (untyped), which the viewer
  reads to draw meshes (`viewer/robot.py:165-182`). `data_model/body.py:21` has an unused `visual_geom: int | None`.
  `io/assets.py:8-9` says "Collision computation is currently stubbed and does not consume resolved meshes."
- Probe on the Panda URDF: `meta` keys are `['asset_resolver', 'ir']`; 9 of 11 links have only `mesh` collision
  geometry (fingers have 4 boxes each); `Model` has no attribute containing "geom" or "coll".
- Roadmap states mesh-to-mesh distance and convex decomposition are not provided (`docs/reference/roadmap.md:37-43`).

## How references evolved (with sources)

- **Pinocchio / hpp-fcl -> coal.** Geometry is a separate pair `GeometryModel` / `GeometryData`, parallel to
  `Model`/`Data`, built from the same URDF (`references/kin_dyn/pinocchio/include/pinocchio/multibody/geometry.hpp`,
  `geometry-object.hpp`, `geometry-object-filter.hpp`). Collision pairs live in the `GeometryModel` and are pruned
  from SRDF `disable_collisions`. hpp-fcl was renamed coal in v3 (2024) and pinocchio 3 moved to it
  (https://github.com/coal-library/coal/blob/devel/CHANGELOG.md). Lesson: kinematic model stays geometry-free;
  geometry is an attachable brick keyed by frame/joint.
- **cuRobo v1 -> v2.** v1 needed hand-written YAML spheres and collision-ignore lists; v2 added `RobotBuilder`
  with `fit_collision_spheres()` (MorphIt on link meshes) and a three-pass `compute_collision_matrix()`
  (adjacent, overlapping at retract pose, never-colliding over Halton samples) (`references/design/curobo.md:225-240`).
  Self-collision moved from one thread block to map-reduce kernels once a 48-DoF humanoid hit 162k pairs vs 818 for
  Franka (`curobo.md:214-223`). Scene is typed: Cuboid/Sphere/Capsule/Mesh/VoxelGrid (`curobo.md:21`), attached
  objects are re-fit to spheres (`curobo.md:385`), ESDF decoupled from TSDF resolution (`curobo.md:489`).
- **PyRoki.** Typed geometry pytrees (`HalfSpace, Sphere, Capsule, Box, Heightmap`) with a pair dispatch table
  `_get_coll_func` / `collide` / `pairwise_collide` (`references/kin_dyn/pyroki/src/pyroki/collision/_collision.py:41-81`).
  Robot geometry is one capsule per link fitted from the URDF mesh (`_geometry.py:243 Capsule.from_trimesh`), with
  `user_ignore_pairs` and `ignore_immediate_adjacents` (`_robot_collision.py:43-55`). Swept-volume constraints use
  capsules between consecutive sphere positions (`_geometry.py:319 from_sphere_pairs`; `design/pyroki.md:153`).
  Cost uses a smooth activation `colldist_from_sdf` (`_collision.py:133`, from arXiv 2310.17274).
- **MuJoCo / mujoco_warp.** Per-geom `contype`/`conaffinity` bitmasks, `<exclude>` body pairs, parent-child
  filtering, `margin`/`gap`; narrowphase spread over primitive, convex (GJK/EPA), SDF and flex files
  (`references/design/mujoco_warp.md:18`). Output is a contact list (point, normal, frame, dist), not a scalar.
- **Newton.** Shapes are columns of the Model (`shape_transform, shape_body, shape_flags, shape_type`, materials)
  and a pre-allocated `Contacts` buffer is filled by `model.collide()` (`references/design/newton.md:66,78,106-108`).
- **Drake SceneGraph.** Geometry is registered with roles (proximity, perception, illustration); queries go through
  a `QueryObject`; FCL-backed proximity engine with BVH, SDF and hydroelastic contact
  (`references/design/drake.md:30,40-45`). Roles map well to our visual vs collision IR split.

Common path: all started with "hand-authored primitives on links" and later had to add (1) automatic generation of
the collision representation from the robot file, (2) a richer pair filter, (3) a typed scene, (4) contact output
for dynamics. Pinocchio and Drake keep geometry out of the kinematic model; Newton puts it in. Our layering
(`collision` above `data_model`) already matches the Pinocchio split.

## Findings (ranked, F1..Fn: severity, evidence, consequence)

**F1 high: parsed collision geometry is dropped from the typed model.** Evidence: `IRBody.collision_geoms` is filled
by both parsers, but only `viewer/robot.py:170-182` reads it, through `model.meta["ir"]`. Consequence: there is no
path "URDF -> collision query". Every user (and BH) hand-authors sphere offsets; `meta["ir"]` is an untyped side
channel that the viewer and a future collision builder would both depend on, so its shape is a de facto contract.

**F2 high: no generation of spheres from geometry.** Evidence: Panda has 9 mesh-only links (probe). cuRobo needed
`fit_collision_spheres` to scale past hand-tuned arms; pyroki fits capsules from meshes. Consequence: collision is
practically unusable on a new robot without a manual authoring session; humanoids (BH) are worse.

**F3 high: self-collision filter is topology-only.** Evidence: `robot_spheres.py:64-88` removes same-body and
parent/child pairs only; no ignore list, no SRDF, MJCF `contype/conaffinity/exclude` not parsed. Probe: 160 spheres
on 10 frames -> 8448 pairs, all evaluated every iteration. Consequence: links two joints apart that overlap at rest
(common on Panda wrist, humanoid pelvis/thigh) give permanent nonzero residuals that fight the task; the only
workaround is filtering the `(P, 2)` tensor by hand.

**F4 medium: obstacles are closures, there is no scene.** Evidence: `distance_fn` callable in
`terms/collision.py:15-35`. Consequence: the viewer cannot draw what a closure collides with, scenes cannot be saved
or batched per environment, and "attach an object to the gripper" has no home. The callable protocol itself is a good
LEGO seam and should stay as the contract that a `Scene` satisfies.

**F5 medium: no continuous (swept) collision for trajectories.** Evidence: clearance is evaluated at knots only
(`examples/06_obstacle_trajectory.py`). Consequence: thin obstacles tunnel between knots. The fix is cheap with
existing bricks: `sphere_capsule` between a sphere's centre at `t` and `t+1` (pyroki's approach).

**F6 medium: distance functions return only a scalar.** Evidence: `distance.py` signatures. Consequence: dynamics
(contact Jacobians, friction cones), visualization of contact points and BH ground-contact all need witness points
and normals; adding them later means a parallel function family unless the closest-point form is the core.

**F7 medium: `sphere_sdf` gives zero gradient outside the grid.** Probe: centre at x=3 m outside a 0.8 m grid gives
`d=0.9`, gradient `[0, 0, 0]` (index clamp kills it). Also one unbatched grid `(X, Y, Z)` and a Python-float
`voxel_size`. Consequence: an optimizer starting outside the mapped volume gets no signal; per-env maps impossible.
Not documented in `collision.md`.

**F8 medium: "sphere vs X" naming does not scale to new robot primitives.** Evidence: every function takes a sphere
first. A capsule robot (pyroki, cheaper for humanoids) needs `capsule_capsule`, `capsule_box`, ... ; mesh/convex
needs GJK. Consequence: N x M growth without a pair dispatch. Not a wall today; keep `<a>_<b>(a..., *, b...)` naming
so a dispatch table can be added later without renaming.

**F9 low: dense all-pairs evaluation, no broadphase or active set.** Probe: pair build 5 ms at S=160 (one-time,
fine); evaluation is O(P) per iteration. cuRobo needed map-reduce at 162k pairs. Consequence: fine for arms, slow for
a 48-DoF humanoid with ~1000 spheres (~500k pairs).

**F10 low: `RobotSpheres` has no `.to(device, dtype)` and fixed per-robot offsets.** Evidence: dataclass with only
`from_frames` and `centers`. Batched `offsets (B..., S, 3)` broadcast through `centers` by accident (`rotation @
offsets[..., None]`) but this is undocumented. Consequence: BH shape-dependent spheres (radii as a function of
betas) work only by luck; device moves need manual rebuild.

**F11 low: non-smooth points.** `sphere_box` at the exact box centre returns gradient 0 (probe; `abs` at 0); hinge
residual is C0 (squared cost is C1, acceptable for LM). Smooth activation (pyroki/cuRobo) is a nicer default for L-BFGS.

## Roadmap (table: brick | why | prerequisite | when)

| brick | why (user) | prerequisite in architecture | when |
|---|---|---|---|
| `ignore=` frame-name pairs on `self_collision_pairs` | any arm user with overlapping wrist links | none | now |
| Document `meta["ir"]` + `distance_fn` protocol as extension points; ADR "geometry stays out of `Model`" | contributors, BH | ADR file in `docs/contracts/decisions/` | now |
| Document SDF outside-grid behaviour and batched-offsets support | users of `sphere_sdf`, BH | none | now |
| `collision.RobotGeometry` (typed per-geom: frame id, placement `(G, 7)`, kind, params) built from IR via `from_model(model, role="collision")` | everyone with URDF/MJCF robots; viewer can use it too | stable IR geom schema (`IRGeom.kind/params/placement`) | next |
| `fit_spheres(geometry)`: exact for sphere/capsule, rule-based for box/cylinder, mesh via lazy trimesh | new robots, humanoids | `RobotGeometry` | next |
| Sampling-based collision matrix (adjacent / at-rest / never-colliding over samples), SRDF `disable_collisions` import, MJCF `contype/conaffinity/exclude` into IR | arms, MJCF robots | `ignore=` keyword; IR carries filter fields | next |
| `terms.swept_clearance` (capsule between consecutive sphere centres) | trajectory optimisation users | time-window terms already exist | next |
| `collision.Scene`: frozen dataclass of batched primitive tensors with `distance(centers, radii) -> (B..., S, K)` and viewer layer | planning users, viewer, per-env batching | distance_fn protocol (exists) | next |
| Closest-point variants `(distance, point_a, point_b, normal)`; scalar functions become wrappers | contact dynamics, BH foot-ground contact | F6 decision on return type | later |
| Smooth activation (`colldist`-style) option on clearance terms | L-BFGS users | none | later |
| Capsule robot geometry + pair dispatch table `collide(a, b)` | humanoids, BH limbs | `RobotGeometry`, `<a>_<b>` naming | later |
| Attached objects (re-fit spheres onto a frame at grasp) | pick-and-place | `RobotSpheres` concat + `Scene` | later |
| ESDF from mesh / depth, batched grids, outside-grid extrapolation | perception-driven planning | `Scene` | later |
| Active set / top-k pairs, blocked reduction | 48-DoF humanoids | `RobotGeometry` pair matrix | later |
| Convex-convex GJK/EPA in torch; mesh-mesh via lazy coal binding | precise contact, sim | closest-point API | maybe |
| GPU kernel for fused FK -> spheres -> distances | throughput | profiling in `benchmarks/` shows it matters | maybe |

## Prep changes recommended now (each <= 1 day, non-breaking)

1. `self_collision_pairs(model, spheres, *, ignore: Iterable[tuple[str | int, str | int]] = ())`: drop pairs whose
   frames (or their rigid bodies) match an ignored pair. Default keeps current behaviour. One test.
2. ADR in `docs/contracts/decisions/`: "Collision geometry is a separate brick keyed by frame ids (Pinocchio
   GeometryModel pattern); `Model` stays geometry-free; `model.meta["ir"]` is the interim source and its
   `IRBody.collision_geoms` / `IRGeom(kind, params, placement)` schema is frozen until `RobotGeometry` lands."
   Also mark `data_model/body.py:21 visual_geom` as unused or tie it to this ADR.
3. `docs/concepts/collision.md`: state that `sphere_sdf` clamps outside the grid (zero gradient there), that
   `offsets`/`radii` may carry leading batch axes (shape-dependent spheres for BH), and name the `distance_fn`
   protocol as the obstacle extension point (also in `docs/contracts/08-extension-points.md`).
4. Add `RobotSpheres.to(device=None, dtype=None)` returning a new instance; matches the dtype/device contract.
5. MJCF parser: keep `contype`, `conaffinity` and `<exclude>` in `IRGeom.params` / IR (parse only, no behaviour), so
   the information is not lost before the collision matrix brick exists.

## Open questions for the owner

1. Geometry home: separate `collision.RobotGeometry` built from the model (Pinocchio, Drake; fits our DAG) or
   columns on `Model` (Newton)? Recommendation: separate brick.
2. Robot primitive: stay sphere-only (cuRobo; simplest, uniform kernels) or add capsules as a first-class robot
   primitive (pyroki; fewer primitives for humanoids)? Recommendation: spheres now, capsules `later`.
3. Mesh fitting: allow a lazy optional `trimesh` import in `collision` (viewer already uses it) for `fit_spheres`?
4. Should distance functions grow a closest-point/normal return now (breaking later otherwise), or stay scalar with
   separate `*_closest` functions added later? Recommendation: separate functions, scalar stays.
5. Is contact for dynamics (ground contact, friction) in scope for BR, or does BH own it for human feet?
