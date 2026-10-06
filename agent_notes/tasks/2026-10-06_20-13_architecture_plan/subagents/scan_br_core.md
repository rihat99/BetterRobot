# scan_br_core — BetterRobot foundation layers audit (2026-10-06)

Scope: `src/better_robot/{lie,spatial,data_model,io,collision}`, top-level `__init__.py`,
`_typing.py`, `_validation.py`, `exceptions.py`, docs/concepts/{architecture,model_and_data,design_decisions}.md,
tests/contract/. Read-only. All paths below are relative to `src/better_robot/` unless absolute.
Scope total: 8260 lines (wc -l). Measured numbers come from
`/tmp/claude-1003/-home-rikhat-akizhanov-better/d9220856-907e-4a64-b1a1-f6cd72ec6355/scratchpad/bench_val.py`.

## 1. Module map

| package | files (lines) | role |
|---|---|---|
| lie | so3 306, se3 216, tangents 220, types 182, alignment 124, __init__ 16 | functional SO3/SE3 on raw tensors + `SO3`/`SE3` dataclass wrappers + umeyama |
| spatial | inertia 346, mesh 282, motion 115, force 79, symmetric3 66, ops 60, __init__ 30 | Pinocchio-style Motion/Force/Inertia/Symmetric3 value classes + triangle-mesh face orientation |
| data_model | model_structure 493, model 315, data 214, execution_batch 179, model_values 160, topology 76, reduced_coordinates 74, joint_dispatch 64, joint 61, frame 28, body 21, _kinematics_level 23 | Model/Data, topology, mimic reduction, batch broadcasting |
| data_model/joint_models | revolute 194, base 147, composite 116, prismatic 115, planar 88, fixed 81, helical 79, spherical 71, free_flyer 69, translation 54, mimic 46, __init__ 56 | one class per joint type (Protocol `JointModel`) |
| io | build_model 1174, parsers/programmatic 435, parsers/urdf 296, parsers/mjcf 228, builders/kinematic_tree 242, assets 152, __init__ 133, ir 98 | URDF/MJCF/programmatic → IR → Model |
| collision | robot_collision 73, geometry 66, pairs 40, closest_pts 36, __init__ 27 | stubs only (see §6) |
| root | exceptions 141, _typing 82, __init__ 80, _validation 34 | public surface, error types, `check_tensor` |

Top-level `__all__` = 27 names (`__init__.py`): Model, ModelStructure, ModelValues, Data, Frame, Joint, Body,
load, ModelBuilder, SE3, 8 kinematics fns, rnea/aba/crba/center_of_mass/compute_centroidal_map,
Trajectory, solve_ik, solve_trajopt, solve_contact_forces; plus submodules `io`, `spatial`, `exceptions`.
No `lie` functional API at top level (only the `SE3` wrapper, which hot paths never use: 0 hits for
`SE3(`/`lie.types` in kinematics/dynamics/residuals/optim/tasks).

Downstream deep imports (grep of BetterHuman/src + BVR/tools, counts):
- `better_robot.lie import so3/se3` 7+7 — the functional Lie API is effectively public but not top-level.
- `io.builders.kinematic_tree.build_kinematic_tree_model` (BetterHuman body building),
  `io.ir.IRFrame`, `io.build_model` (IR → Model as a public step),
  `data_model.reduced_coordinates.expand_configuration`, `data_model.model_structure.JOINT_KIND_CODES`,
  `spatial.mesh.orient_faces_by_component`, `kinematics.jacobian.compute_joint_jacobians`,
  `viewer.RenderContext`, `spatial.Inertia`.
→ The real public API is ~2x the declared one; IR, builder, kind codes and mimic expansion are de facto public.

## 2. Model / Data design

- `ModelStructure` (`data_model/model_structure.py:75`, frozen): counts, names, name→id dicts, parents/children/
  subtrees/supports/topo_order, per-joint nq/nv/idx tables (reduced AND `_full` pre-mimic variants, :102-109),
  `joint_models` tuple (:110), `joint_kind_codes` (:111), mimic, 8 `manifold_*_joint_ids` category tables
  (:117-124), and ~9 device tensors (`joint_kind_tensor`, `parents_tensor`, ... :126-134). So "structure"
  is not device-agnostic: it carries device tensors and must be moved with the model.
- `ModelValues` (`model_values.py:43`): 15 tensors — joint_placements, body_inertias (packed), frame_placements,
  4 limits, rotor_inertia, armature, friction, damping, gravity, mimic_multiplier/offset, q_neutral; has
  `tree_flatten/unflatten` (:133) and `validate` (:62). Values may carry batch dims (`_execution_batch_shape` :86).
- `Model` (`model.py:54`, frozen): `structure`, `values`, `reference_configurations: dict`, `meta: dict`.
  57 one-line forwarding properties (`return self.structure.X` / `self.values.X`, model.py:76-162) duplicate both
  sub-objects' surface. `meta` is an untyped dict holding `asset_resolver` and the whole `IRModel` as
  `meta["ir"]` (viewer reads geoms from it: `viewer/render_modes/urdf_mesh.py:139,157`). Frozen dataclass with
  mutable dict fields → not really immutable, not hashable by value.
- Parameter changes: `Model.with_values(joint_placements=, body_inertias=, frame_placements=)` only (model.py:261-267).
  Gravity, limits, armature, etc. need `replace(model, values=replace(model.values, gravity=g))`. BetterHuman uses
  `with_values` for shape-dependent bones/inertias (`bodies/mhr/deformation.py:61,96`, `smpl_family/shared.py:151`).
- `Data` (`data.py:72`, mutable dataclass): q, v, a, tau + 22 optional cache fields (joint/frame poses, velocities,
  accelerations, forces, jacobians, jacobians_dot, M, C, g, bias, ddq, CMM, h_g, com p/v/a) and
  `_kinematics_level`. `__setattr__` (:127-141) invalidates caches on q/v/a reassignment; `require(level)` raises
  on stale access. Contract tests pin this (`tests/contract/test_cache_invariants.py`), including the known hole
  `test_inplace_q_mutation_is_not_detected` (:89): `data.q[...] = x` silently leaves stale caches.
- Does mutable Data earn its place? Weakly. In a functional autograd library every call already recomputes from q
  (forward_kinematics returns a fresh Data). The cache-level machinery protects against stale reuse but (a) misses
  in-place mutation, (b) makes Data a grab-bag of 26 Optional fields with no type-level guarantee of what is filled,
  (c) only 7 `.require(` call sites exist in src. A typed, immutable result per algorithm (e.g. `FKResult(joint_pose,
  frame_pose)`, `Jacobians(...)`, `Dynamics(M, b)`) passed explicitly would be simpler and torch.compile-friendlier.
- Batch semantics: leading `B...` dims on q and on per-call Data fields; ModelValues may also be batched and is
  broadcast against q by `execution_batch.py` (`broadcast_execution_batch_shape` :59, `flatten_execution_batch`
  :145, `ExecutionInput.reduce_gradient` :33) — 179 lines and 42 call sites outside the file. That is the right
  idea (per-sample shapes for humans) but it is plumbing every algorithm must remember to call.

## 3. Lie + spatial

- Representation: plain tensors, `[tx,ty,tz,qx,qy,qz,qw]` (scalar-last), tangents linear-first
  (design_decisions.md "Quaternions scalar-last, tangents linear-first"). Wrappers `SO3`/`SE3` (`lie/types.py:32,100`)
  are thin dataclasses around `.tensor`; exported at top level but unused internally.
- Inventory: so3 {identity, compose, inverse, log, exp, act, adjoint, from/to_matrix, from/to_euler, from_axis_angle,
  normalize, slerp}; se3 {identity, compose, inverse, log, exp, act, adjoint, adjoint_inv, from/to_matrix,
  from_axis_angle, from_translation, normalize, apply_base, sclerp}; tangents {hat/vee so3/se3, left/right Jacobian and
  inverses for SO3 and SE3} (`lie/tangents.py:26-218`). Missing: Jacobians of compose/inverse/act w.r.t. arguments
  (pypose/theseus-style `jac_*`), batched `log` Jacobian helper (only via `right_jacobian_inv`). `alignment.umeyama`
  is a point-set registration algorithm, not Lie algebra.
- Numerical safety: good. so3.log flips to qw>=0, uses atan2 and a Taylor branch with a safe `where` input
  (`so3.py:137-151`); exp likewise (:154-166); se3.log uses Taylor 1/12+θ²/720 near 0 (`se3.py:49-69`);
  `_so3_jac_coefficients` has Taylor A,B (tangents.py ~88-103). The double-`where` pattern keeps gradients finite.
  Near θ=π se3.log `cot(θ/2)`→0 fine; no special π handling for log Jacobian inverse (coefficient grows but stays finite
  on [0,π]).
- Motion/Force/Inertia/Symmetric3 classes: bypassed by every hot path. Grep for `Motion(`/`Force(`/`Inertia(`/
  `Symmetric3(` in kinematics/dynamics/residuals/optim/tasks/collision: 0 hits. Users of the classes: io builders,
  mjcf parser, `ModelValues.body_inertia` accessor (`model_values.py:104-107`), viewer force overlay.
  `spatial/ops.py` (ad, ad_star, cross_mm, cross_mf, act_*) is imported nowhere outside spatial.
  `dynamics/rnea.py:76,91` re-implements `_cross_motion`/`_cross_motion_force` on raw tensors with docstrings saying
  "Matches Motion.cross_motion" — explicit duplication. `spatial/mesh.py` (282 lines, face orientation) is geometry
  processing, unrelated to spatial algebra.
- Net: spatial/ (~700 lines) is mostly a parallel, typed API nobody fast uses; lie/ is the real kernel layer.

## 4. Joint models

- Two parallel representations: (a) classes implementing Protocol `JointModel` (`joint_models/base.py:41`: kind, nq,
  nv, axis, joint_transform, joint_motion_subspace, joint_velocity, integrate, difference, random_configuration,
  neutral), and (b) integer kind codes `JOINT_KIND_CODES` (`model_structure.py:27-48`, 18 codes, axis-aligned
  revolute/prismatic split into rx/ry/rz/unaligned) + 8 manifold category id tables.
- Dispatch: torch FK uses kind codes through `data_model/joint_dispatch.py:34-61` (if-chains on code sets, falling
  back to `joint.joint_transform(q_slice)` :61); warp bridges switch on `joint_kind_tensor`; manifold ops in
  `ModelStructure.integrate` use category tables plus per-joint `self.joint_models[joint_id].integrate` for the
  fallback set (`model_structure.py:320`); composite joints loop in Python (`composite.py:52,63,87`).
- Per-joint Python loops survive in every eager algorithm: `kinematics/_fk_matrix.py:262`, `dynamics/rnea.py:186,234`,
  `aba.py:163,196,238`, `crba.py:75,87,99`, `centroidal.py:108` (`for i in structure.topo_order`). This is the source
  of the ~1 ms flat eager FK cost (measured below), not validation.
- Adding a joint type touches 6 files: a JointModel class, `JOINT_KIND_CODES`, `joint_dispatch.py`,
  `io/build_model.py:_kind_to_joint_model` (:381), `kinematics/_warp_bridge.py`/`dynamics/_warp_bridge.py` (+ kernels),
  and ModelStructure manifold tables. No registry.

## 5. io

- Parsers: URDF (`parsers/urdf.py`), MJCF (`parsers/mjcf.py`, subset), programmatic `ModelBuilder`
  (`parsers/programmatic.py:65`). All emit `IRModel` (`io/ir.py:82`; IRJoint/IRBody/IRGeom/IRFrame dataclasses).
  `build_model` (`io/build_model.py:581`, 1174-line file) does topo sort, mimic reduction (`_build_mimic_reduction`
  :202, ~180 lines), kind mapping, inertia packing, invariant checks — it is the real model compiler.
- `io/builders/kinematic_tree.py` (242) builds models from parent arrays/offsets (used by BetterHuman) and itself goes
  through ModelBuilder → IR → build_model (:23,25). Good single path.
- Entanglement: IR is stored on the model as `meta["ir"]` and the viewer reads `IRBody.visual_geoms` from it
  (`viewer/render_modes/urdf_mesh.py:21,102-157`); asset resolution (`io/assets.py`, includes `urllib` downloads
  :18,115) is forwarded via `meta["asset_resolver"]` (urdf.py:288, mjcf.py:219). So visual/collision geometry has
  no typed home in Model; viewer depends on io internals. io imports nothing from optim/tasks/viewer (good).

## 6. collision

- `geometry.py`: Sphere, Capsule, Box, HalfSpace, Plane dataclasses + `colldist_from_sdf` (1 NotImplementedError).
- `closest_pts.point_to_segment/segment_to_segment`: `raise NotImplementedError` (:23,36).
- `RobotCollision.from_model/world_capsules/self_distances/world_distances`: all NotImplementedError (:48,55,62).
- `pairs.distance` dispatch registry: stub (1 NotImplementedError).
- No residual, task or optim module imports collision (grep). Maturity: zero; it is a placeholder API (~240 lines),
  guarded only by `tests/contract/test_roadmap_stub_inventory.py`.

## 7. Validation / typing layer

- `_validation.check_tensor` (34 lines): isinstance + trailing-shape + dtype + device checks, pure Python metadata.
  FK validates q once (`kinematics/forward.py:68,222`); quaternion-norm check is opt-in (`check_quaternion_norm`,
  :238) because it syncs (`bool(...any())`, forward.py ~111). Contract test
  `test_boundary_validation_count.py:48` enforces "validate once at public boundary".
- Measured (Panda 7-DoF from theseus fixture, B=64): eager `forward_kinematics` 1039 us CPU, 1414 us CUDA;
  `_validate_q` 1 us on both. Validation is <0.1% of FK; it is not the overhead problem. The overhead is the Python
  per-joint loop + many small kernel launches (§4).
- `exceptions.py` (141): typed error hierarchy (ShapeError, DtypeMismatchError, DeviceMismatchError,
  QuaternionNormError, IRError...). Fine. `_typing.py` (82): aliases, low value but harmless.
- Contract tests (tests/contract/): layer DAG via AST (`test_layer_dependencies.py`), hot-path lint
  (`test_hot_path_lint.py:470`, forbids sync patterns unless tagged `bench-ok`), public API core, naming, docstrings,
  optional imports, protocols, cache invariants, submodule public imports, stub inventory. Strong governance for the size.

## 8. Top structural problems (ranked)

1. Per-joint Python dispatch in every eager algorithm (`_fk_matrix.py:262`, rnea/aba/crba/centroidal loops) with
   kind codes + class fallback; ~1 ms floor for a 7-DoF FK independent of batch. Needs level-batched (depth-wise)
   or packed-table vectorized passes, or a single kernel backend — one implementation.
2. Two joint representations (JointModel classes vs 18 kind codes + 8 manifold tables) kept in sync by hand across
   6 files; adding a joint is a cross-cutting change. Pick one: data-driven tables with a small registry per kind.
3. Model surface bloat: 57 forwarding properties on Model (`model.py:76-162`), `_full` and reduced duplicates of every
   index table, device tensors inside "structure". Structure vs values split is right; the facade is not.
4. Parameter editing is partial: `with_values` covers 3 of 15 value fields; gravity/limits need nested
   `dataclasses.replace`. Should be `model.replace(gravity=..., joint_placements=...)` over all ModelValues fields,
   with ModelValues registered as a pytree (it already has tree_flatten :133).
5. Mutable cache-tracking Data (26 Optional fields, `__setattr__` invalidation, known in-place hole
   `test_cache_invariants.py:89`) in a functional autograd library; replace with immutable typed results.
6. spatial/ typed classes and `spatial/ops.py` unused by hot paths; rnea duplicates them (`rnea.py:76,91`).
   Either make one raw-tensor `spatial` function module the single source (used by dynamics) or delete the classes.
   `spatial/mesh.py` and `lie/alignment.py` are misfiled.
7. Geometry has no typed home: `Model.meta["ir"]` / `meta["asset_resolver"]` untyped dict back-references couple
   viewer to io internals; collision cannot build on it. Need a `GeometryModel` (visual + collision geoms per body),
   Pinocchio-style, produced by io next to Model.
8. Declared public API (27 names) ≠ real API: downstream imports io.build_model, io.ir.IRFrame,
   builders.kinematic_tree, reduced_coordinates.expand_configuration, JOINT_KIND_CODES, lie.so3/se3. Plus
   collision is an all-stub package advertised as a layer. Make the real surface explicit; drop stubs until real.

What is good and should survive:
- Functional Lie on raw tensors with careful Taylor/double-where numerics and analytic left/right Jacobians
  (lie/so3.py, se3.py, tangents.py); scalar-last + linear-first conventions documented.
- Structure (static, hashable topology) vs Values (differentiable, batchable tensors) split; batched ModelValues
  broadcasting against q (execution_batch) — needed for per-subject human shapes.
- Single IR → build_model compiler path for URDF, MJCF, programmatic and kinematic-tree builders; mimic reduction.
- Cheap boundary validation (1 us) with opt-in syncing checks; typed exception hierarchy.
- Contract tests: AST layer DAG, hot-path sync lint, validate-once count — keep this governance in any redesign.

## 9. Proposed package layout (no back-compat)

```
better_robot/
  __init__.py        small explicit surface: Model, load, build, fk, jacobians, dynamics fns, lie ns, solve ns
  lie/               so3, se3, tangents (+ jac of compose/act/inverse); raw tensors only; SE3 wrapper dropped or optional
  spatial/           raw-tensor 6D algebra used by dynamics: cross_m/cross_f, act, inertia pack/unpack (no classes)
  model/             ModelStructure (pure-Python, hashable), ModelValues (pytree), Model.replace(**values), batching
    joints/          one table-driven joint registry: kind -> (nq, nv, transform, subspace, integrate, warp code)
  geometry/          GeometryModel (visual/collision geoms per body), primitives, mesh utils (from spatial/mesh)
  io/                parsers (urdf, mjcf, programmatic, kinematic_tree) -> ir -> compile.py (build_model split)
  kinematics/        fk, jacobians, frames; returns typed immutable results; one backend per algorithm
  dynamics/          rnea, aba, crba, centroidal on spatial/ functions
  collision/         distances on geometry/ primitives (only when implemented), feeds residuals
  algorithms/ (opt)  umeyama/alignment and other non-Lie utilities
  optim/, residuals/, tasks/   (covered by other agents)
  viewer/            reads Model + GeometryModel, never io internals
  _validation.py, exceptions.py   unchanged
```

## Open questions
- Is Data's cache-level API used by BVR/BetterHuman beyond `data.joint_pose_world`/`frame_pose_world`? (grep needed
  before replacing with result types.)
- Do any downstream users rely on `Model.meta["ir"]` directly (besides viewer)?
- Not verified: performance of level-batched FK vs current loop; warp-path FK timing (other agent).
