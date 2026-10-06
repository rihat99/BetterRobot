# BetterHuman architecture scan (2026-10-06)

Root: `/home/rikhat.akizhanov/better/dev/BetterHuman` (paths below relative to it,
`bh/` = `src/better_human/`). Read-only scan, ~45k token budget: core/ and the
SMPL-family bake were read in full; MHR, mass, studio, io were grepped, not read
line by line. Claims marked (grep) rest on signatures/greps only.

## 1. Module map (7,345 lines in src; studio alone ~3,160 = 43%)

| File | Lines | Role |
|---|---:|---|
| bh/__init__.py | ~10 | exports `SMPL, SMPLX, MHR, ParametricBody, load` |
| core/body.py | 209 | `BodyStructure` (static), `BodyValues` (tensors), `ParametricBody` (frozen dataclass: structure + values + `robot: br.Model` + two policy callables) |
| core/pipeline.py | 206 | pure tensor ops: `blend_shapes`, `regress_joints`, pose features (rotmat-I, MHR 6D-from-Euler), `linear_corrective`, `mlp_corrective`, `skinning_transforms(_from_inverse)`, `lbs` |
| core/classic.py | 124 | axis-angle<->quat, classic root <-> free-flyer, `broadcast_configuration` |
| core/reduction.py | 163 | slice shape basis to requested widths; drop terminal joints (partial bodies) |
| core/mass.py | 272 | part-mesh topology (boundary loops, caps; pure-Python on CPU lists), `body_inertias_from_shape` |
| core/contact_frames.py | 160 | vertex-anchored frames: parse json, normals, local placements |
| bodies/registry.py | 40 | `@register(name)` dict + `load(name, **cfg)` |
| bodies/config_body.py | 49 | `BodyConfig`: joint names, landmarks, axes, asset filenames |
| bodies/smpl_family/shared.py | 354 | SMPL-family shape bake, linear pose corrective, BR model build, mass binding |
| bodies/smpl_family/smpl.py / smplx.py | 184 / 404 | concrete classes, `from_classic`/`to_classic`, hand PCA, face/hands toggles |
| bodies/smpl_family/config_smpl*.py | 89 / 124 | static joint/landmark tables |
| bodies/mhr/{body,tree,deformation,archive,config}.py | 319/226/120/161/36 | MHR: archive reader, BR tree with mimic/reduced coords, bake (runs BR FK at neutral), MLP corrective |
| bodies/assets/{__init__,densities}.py | 71/130 | segmentation/density assets |
| io/load.py, schema.py, convert_chumpy.py | 116/290/174 | npz load + declarative schema validation + chumpy pkl conversion |
| viz/skinned_mesh.py | 87 | `SkinnedMeshMode`, a layer on `better_robot.viewer.RenderContext` |
| viz/studio/* (15 files) | ~3,160 | separate viser app `HumanStudio` that owns its own `viser.ViserServer` (app.py:60): layers (mesh, skeleton, frames, parts), gizmo, picking, posing, segment/frames editors, persistence |
| benchmarks/benchmark_bodies.py | 812 | one script: vertex streaming vs upstream smplx / MHR TorchScript, FK + RNEA torch vs warp |
| tools/convert/mhr/convert_mhr.py, tools/parity/* | 747 + ~800 | asset conversion and parity dumps |

Minimal user code (verified against examples/01, 03 and body.py):
```python
import better_human as bh, better_robot as br
body = bh.SMPLX(path, use_hands=True, use_face=True)          # compute_mass=True default?
body = body.with_shape(betas=betas_B10)                        # bakes v_shaped, joints, Model values, inertias
q    = body.from_classic(bh.bodies...SMPLXClassic(...))        # axis-angle -> q (..., nq)
verts = body.vertices(q)                                       # (B, V, 3)  = fk + correctives + LBS
data  = body.fk(q)                                             # br.Data; joints = data.joint_pose_world[..., :3]
keypts = body.landmark_positions(verts)                        # (B, L, 3)
tau  = br.rnea(body.robot, q, v, a)                            # body.robot is a plain br.Model (name per BR API)
I    = body.values.body_inertias                               # (B, nbodies, 10) packed BR inertias
```
Joints have two meanings: `values.rest_joints` (regressed, rest) and posed
`data.joint_pose_world`; there is no `body.joints(q)` helper. Note
`MHR.from_classic` returns `(MHR, q)` (mhr/body.py:225) while
`SMPL.from_classic` returns `q` (smpl.py:145): MHR classic carries shape.

## 2. Forward pipeline

Split in two phases; this is the central design decision and it is good.

Phase A, shape bake, `ParametricBody.with_shape` (core/body.py:115-146) ->
family policy, SMPL: `bake_regressed_pelvis_shape` (smpl_family/shared.py:105-157):
1. shape blend `v_template + blend_shapes(coeffs, shape_basis)` (shared.py:112).
2. joint regression `einsum("jv,...vc")` dense (J,V) regressor (pipeline.py:34; shared.py:113).
3. pelvis-centre, child offsets -> identity-rotation joint placements (shared.py:115-127).
4. per-part inertias from closed part meshes (shared.py:130-136 -> mass.py:232-262).
5. contact-frame placements re-baked from shaped vertices (shared.py:146-149).
6. `body.robot.with_values(joint_placements, body_inertias, frame_placements)` (shared.py:151).
MHR bake (mhr/deformation.py:31) instead runs `br.forward_kinematics` at
`q_neutral` (deformation.py:62) and uses authored inverse binds + joint scales.

Phase B, per pose, `vertices(q)` (body.py:156-179):
1. FK: `br.forward_kinematics(self.robot, q)` (body.py:154). No `use_warp`
   passthrough; the warp lane is only reachable by calling BR directly, as the
   benchmark does (benchmark_bodies.py:398-400).
2. pose corrective: SMPL `(R - I)` features @ posedirs (pipeline.py:37-46, 76-82);
   MHR 6D-Euler features -> 750->3000->V*3 MLP (pipeline.py:49-73, 113-125;
   docs/architecture.md:246-248).
3. skinning transforms `T_world * inv(T_rest)` (pipeline.py:128-157); the rest-pose
   inverse is recomputed every call from `rest_joints` (pipeline.py:139-142), cheap.
4. dense matrix-blend LBS: joints -> (J,12) affines, `einsum("vj,...jx")` with the
   dense (V,J) weight matrix, then per-vertex 3x4 apply (pipeline.py:176-191).

Batching: everything uses trailing event dims and broadcasting; shape batch
and pose batch broadcast together (body.py:140-144). Shape-dependent work is
cached by construction (bake once, stream poses). Remaining per-call work that
could be cached: identity-quaternion bind construction (pipeline.py:139-141);
`so3.to_matrix` of 7D poses (FK already produced rotations internally, so pose
-> quat -> matrix is a round trip, pipeline.py:44, 176).

Cost profile: FK eager torch is latency-bound (flat in batch: SMPL 2.5 ms @B=1
vs 1.6 ms @B=256), i.e. Python/kernel-launch overhead in BR FK, not FLOPs.
Correctives + LBS are bandwidth-bound: SMPL-X posedirs (486 x 3V, V=10,475) is
~61 MB fp32 read per call; MHR MLP output layer (3000 x 3V) is the largest
tensor (size depends on LOD, not verified). Mass bake is the slowest shape
path: Python loop over parts (mass.py:253) calling `Inertia.from_mesh` per part
with an index_select per boundary loop (mass.py:224).

Stored numbers (docs/contracts.md:176-196; BR rev 8c4e619, BH c8da7ec, dirty;
GPU not named in the excerpt read). Fixed-shape vertex streaming, ms:

| Body | B | upstream full | BH torch | BH warp-FK |
|---|---:|---:|---:|---:|
| SMPL | 1 | 1.55 | 3.93 | 2.20 |
| SMPL | 256 | 2.32 | 4.11 | 2.94 |
| SMPL-X | 1 | 1.71 | 2.72 | 1.32 |
| SMPL-X | 256 | 4.32 | 5.33 | 3.85 |
| MHR | 1 | 3.04 | 7.72 | 2.33 |
| MHR | 256 | 12.11 | 13.40 | 8.65 |

FK/RNEA, ms (torch / warp): SMPL B=1 FK 2.51/0.63, RNEA 11.6/1.05; SMPL-X B=1
FK 1.87/0.36, RNEA 13.0/0.63; MHR B=1 FK 6.39/0.90, RNEA 88.5/1.61; MHR B=256
RNEA 53.7/4.10. Reading: BH torch is 1.1-2.5x SLOWER than upstream smplx even
with shape baked out of timing, and that gap is almost all BR torch FK
overhead. Torch RNEA is 10-55x slower than warp. These are single-script,
median-of-30 numbers, no CPU rows, no backward-pass rows, no memory rows.

## 3. BetterHuman on BetterRobot

Full import list (grep, src only): `better_robot.{Data, Model, ModelBuilder}`,
`br.forward_kinematics` (body.py:154, mhr/deformation.py:62),
`better_robot.lie.{se3, so3}` (pipeline, classic, contact_frames, mass, mhr/body,
mhr/tree, studio), `better_robot.spatial.Inertia` (mass.py, studio/segment_editor),
`better_robot.spatial.mesh.orient_faces_by_component` (mass.py),
`better_robot.io.build_model` + `ModelBuilder` (mhr/tree.py),
`better_robot.io.builders.kinematic_tree.build_kinematic_tree_model` and
`better_robot.io.ir.IRFrame` (smpl_family/shared.py:12-13),
`better_robot.data_model.reduced_coordinates.expand_configuration` (mhr/deformation.py),
`better_robot.data_model.model_structure.JOINT_KIND_CODES` (studio/posing.py),
`better_robot.viewer.RenderContext` (viz/skinned_mesh.py:8).
Several are deep private-ish paths (io.builders.kinematic_tree, io.ir,
data_model.reduced_coordinates, model_structure.JOINT_KIND_CODES): BH depends on
BR internals, not a stable surface.

Body -> Model: SMPL family = free-flyer pelvis + spherical joints (SMPL-X: 56
joints incl. universe, nv 168 = 6 + 54x3? per contracts.md:191; SMPL nv 75);
built once by `build_family_robot` (shared.py:312-329). Shape changes only
`ModelValues` via `Model.with_values(joint_placements, body_inertias,
frame_placements)` (shared.py:151) with batched leading dims, so per-subject
link lengths are differentiable values on a fixed `ModelStructure`. This is the
right seam. MHR = 203 joints, nv 131: revolute/mimic reduced coordinates expanded
by BR (`expand_configuration`), rank-deficient for solve-based dynamics
(contracts.md:203-204).

Overlap / friction:
- One FK (BR), good. But skinning-side SE(3)->matrix conversion and bind
  inversion live in BH (pipeline.py:128-191); BR has no skinning, so no duplicate.
- Inertia: BH owns segmentation + caps + density, BR owns `Inertia.from_mesh`;
  clean split, but the per-part loop is in BH.
- Two viewers: `viz/skinned_mesh.py` is a proper BR viewer layer; `viz/studio`
  is a parallel viser application (own server, own mesh/skeleton/frames layers,
  gizmo, picking) that does not use BR's viewer. This is the main duplication.
- Warp: BR's opt-in FK/RNEA lanes are not exposed through `ParametricBody`.

## 4. Adding a body family

Registry = 40-line decorator dict (registry.py); only `smpl`, `smplx`, `mhr`
registered (smpl.py:47, smplx.py:91, mhr/body.py:41). SMPL-H, MANO, STAR,
SUPR, SMPL+H hands-only are NOT implemented; the README/docs vision mentions
the family but code has three. Extension points per family:
`BodyConfig` static tables, a loader building `BodyStructure`/`BodyValues`, a
`ShapeBakePolicy` and a `PoseCorrectivePolicy` (body.py:88-92), plus
`from_classic`/`to_classic` and a `*Classic` dataclass. SMPL-H/MANO would reuse
`bake_regressed_pelvis_shape` + `smpl_linear_pose_corrective` (~150-250 lines +
config table). STAR (sparse quaternion-feature correctives) and SUPR would need a
new corrective policy (~50 lines) in pipeline.py. MHR cost ~860 lines + 747-line
converter. Special cases: `getattr(body, "contact_frames", None)` duck-typing in
the generic bake (shared.py:145); SMPL-X expression is a *shape* group
(smplx.py:136,141), so per-frame expression forces a per-frame bake including
inertias and placements; MHR classic carries shape so `from_classic` returns a
new body.

## 5. Structural problems (ranked)

1. Studio is a second viewer framework (~3,160 lines, 43% of src) owning a
   viser server (studio/app.py:60) and re-implementing mesh/skeleton/frame
   layers already conceptually in BR viewer; only `skinned_mesh.py` follows
   the layer-on-RenderContext pattern. Target: studio = BR viewer + BH layers + editors.
2. BH reaches into BR internals (io.builders.kinematic_tree, io.ir,
   data_model.reduced_coordinates, model_structure.JOINT_KIND_CODES). BR needs a
   public "build tree from arrays" + joint-kind API, or BH breaks on every BR refactor.
3. Torch path is slower than upstream smplx (contracts.md:176-183) because BR
   eager FK is launch-bound; BH has no fused FK+LBS path and no `use_warp`
   passthrough (body.py:154). Any "fast human" claim depends on BR FK work.
4. Expression-as-shape couples per-frame face motion to the full shape bake
   (joint regression, inertia Python loop, placements). Needs a split between
   identity shape (bake) and per-frame blend groups (stream): expression, and
   for MHR any per-frame scale.
5. `ParametricBody` mixes data and behaviour through two private policy
   callables stored as dataclass fields plus `object.__new__` cloning
   (body.py:105-108, 199-206); subclasses add attributes (`contact_frames`) that
   the generic bake reads via getattr. Inheritance + policies + duck typing at once.
6. Inconsistent family APIs: `from_classic` returns `q` (SMPL/SMPL-X) vs
   `(body, q)` (MHR); no common `joints(q)` / `keypoints(q)` accessor; landmark
   naming ("frame_vertex_ids", OpenPose-style) lives in BodyConfig but contact
   frames are a separate JSON mechanism (contact_frames.py:54).
7. Mass topology built with pure-Python lists on CPU (mass.py:119-153) and a
   per-part loop at bake time (mass.py:253); fine once per subject, bad for
   per-frame bakes (problem 4) and for differentiable shape fitting at batch.
8. Missing pieces for the stated use cases: no retargeting (human->robot),
   no marker/virtual-marker sets beyond vertex landmarks, no muscles, no
   contact model beyond vertex-anchored frames, no ground/penetration terms,
   no human-specific residuals for BR optim (BVR builds these itself), no
   SMPL-H/MANO/STAR/SUPR, no pose priors (VPoser/GMM) or joint limits.

Belongs elsewhere: SE(3) dense LBS and `skinning_transforms` are generic and
could sit in BR spatial/kinematics if BR ever skins robot meshes; conversely
`io/schema.py` (290 lines) is a generic array-schema validator.

## 6. Performance notes

- No Python loop over joints in BH's forward; FK loops are BR's. Python loops
  exist only at load/bake: mass topology (mass.py:82-153), per-part inertia
  (mass.py:253), kintree parents (shared.py:190), studio.
- Dense (V,J) weight matrix used for LBS; the sparse (V,K) padded weights are
  only used by studio segmentation (studio/state.py:43). Dense einsum is fine
  for J<=56 but for MHR (more skin joints; count not verified) a K-sparse
  gather may be much less work; worth benchmarking.
- Pose-corrective matmul dominates bytes moved; bf16/fp16 posedirs or sparse
  posedirs (sparsity not measured) are candidates. MHR MLP output layer is the largest
  memory resident; `use_correctives=False` drops it (architecture.md:250).
- dtype/device: `with_shape` rejects mismatched dtype/device instead of casting
  (body.py:134-138), consistent with BR contracts.
- Missing from the benchmark: backward pass (fitting is the main use), CPU,
  peak memory, bake cost (with_shape incl. mass), per-frame expression,
  torch.compile/CUDA-graph lanes, long trajectories (T=1k-10k frames), and a
  stored machine-readable result file (numbers only live in a markdown table).
  Suggested suite: {bake, vertices fwd, vertices fwd+bwd, FK, RNEA, mass} x
  {SMPL, SMPL-X, MHR} x B in {1, 64, 1024, 8192} x {cpu, cuda} x {fp32, bf16},
  report median, p90, peak memory, and ratio to upstream smplx / MHR.

## 7. What should survive

- Bake/stream split with shape living in BR `ModelValues` (body.py:115-146,
  shared.py:151): differentiable per-subject link lengths on a fixed topology.
- Pure tensor op module `core/pipeline.py` with documented shapes; dense
  matrix-blend LBS in one einsum.
- Body as a real BR `Model` (free-flyer + spherical): IK, RNEA, BR optim work
  on humans with no adapter (examples/03_ik_reach.py:51-93).
- Native-vs-BR joint order gather (`native_pose_joint_indices`, body.py:181-188)
  and classic<->q converters with parity tools (tools/parity/*) and a vertex
  gate (contracts.md:185-187: max err 6e-7 m SMPL family, 1.4e-6 m MHR).
- Declarative npz schema validation (io/schema.py) and asset conversion offline.
- `SkinnedMeshMode` as the template for how BH plugs into the shared viewer.

## Prior notes (skim only)

`BetterHuman/references/design/betterrobot_readiness.md` (379 lines) predates
the current code (it discusses `make_smpl_like_model` presets and asks whether
the Structure/Values split is load-bearing, which the code now answers yes).
`BEST_PRACTICES.md` (643 lines) not read. Treat both as stale.

## Open questions

- Does BR FK torch loop per joint or per depth level? (decides fused human FK+LBS design.)
- Should expression / per-frame blend groups be modelled as a pose-like input?
- Is studio's editor feature set (segment/frames editors, persistence) required
  in the unified viewer, or can it stay a BH app on top of BR viewer layers?
- Which GPU produced the contracts.md numbers? Not stated in the lines read.
