# P1 — human theme: adversarial architecture review of BetterHuman (BH)

Scope: BH `src/better_human` (worktree `/home/rikhat.akizhanov/better/dev/BetterHuman`, head `bad0de3`) + `docs/`.
Budget note: I read the core (`core/body.py`, `core/spec.py`, `core/rig.py`) in depth and the other modules by public
signature (`grep` of every `def`/`class`); the studio (`viz/studio`) was not reviewed (viewer theme). Web sources were
not consulted; reference claims cite the `references/design/*.md` digests, which cite upstream file:line.

## Current state (facts, with paths)

- **Entry point.** `better_human.load(family, path, **options) -> Body` (`src/better_human/__init__.py`). The family
  registry is `_LOADERS = {name: getattr(families, name) for name in families.__all__}`: six loaders `smpl, smplh,
  smplx, mano, flame, mhr` (`families/__init__.py`). Assets for `star` and `dmpl` sit in `models/` with no loader.
- **Three-level state.** `FamilySpec` (data: vertices, faces, joints, skin, `identity`/`streams` bases, `corrective`,
  `rig`, `parts`, `landmarks`, `unit/up_axis/forward_axis`, `meta`; `core/spec.py:71-113`) -> `Body` (BR `Model`
  topology, segmentation, mass topology, contact frames; `core/body.py:23-68`) -> `Identity` from `Body.bake(**groups)`
  (baked `v_shaped`, rest joints, `Model.with_values(joint_placements, body_inertias, frame_placements)`;
  `core/body.py:120-172`) -> `HumanState` from `Identity.forward(q, stream)` (memoised outputs; `core/body.py:201-330`).
- **Shape space is linear.** `Body.bake` sums `blend_shapes(coefficients, spec.identity[name])` for every identity
  group before the rig sees it (`core/body.py:~125`); `FamilySpec.identity: Mapping[str, Tensor]` (linear bases only).
- **Rig is the one open protocol.** `Rig` (`core/rig.py:41-64`: `build_model`, `bake`, `placements_from_rest`,
  `stream_joint_basis`, `native_joint_ids`, `corrective_features(spec, model, q)`, `to`) with two implementations:
  `RegressedRig` (SMPL family, `core/rig.py:75`) and `AuthoredRig` (MHR, `families/mhr_rig.py:24`).
- **Correctives are a closed enum.** `Corrective.kind: Literal["none", "linear", "mlp"]` (`core/spec.py:18-30`);
  the feature is `rig.corrective_features(spec, model, q)` — pose only (`core/body.py:285`).
- **`meta` is load-bearing.** Core reads `meta["closed_segmentation"]`, `meta["inverse_bind_poses"]`,
  `meta["bind_joint_positions"]` (`core/body.py:59,269,277`); families read `root_parameter_transform`
  (`families/mhr.py:147`), `hand_components`, `hand_mean`, `native_joint_ids`, `side`, `hand_joints`
  (`families/classic.py:52-121,286`), `translation_basis`, `scale_basis`, `joint_offsets`, `joint_pre_rotations`
  (`families/mhr_rig.py:45-66`). None of these keys is declared on `FamilySpec`.
- **Contact frames** are vertex-anchored BR operational frames re-placed on every bake (`core/contact_frames.py`,
  `Body._attach_contact_frames`). They require a `RegressedRig`; verified:
  `bh.load("mhr", contact_frames={"heel": (0, None)})` -> `ValueError: contact frames need a regressed rig (SMPL family)`.
- **Mass** is per-part mesh integration through BR `spatial.Inertia.from_mesh` (`core/mass.py:4-19`, `232`), density
  scalar or per part (`Body(density=...)`).
- **Reduction** = `drop_joints(spec, names)` + `core/reduction.py` (`slice_parameter_basis`, `reduce_joint_data`); one
  known failing test `tests/core/test_reduction.py::test_corrective_slice_is_exact_when_dropped_joint_is_at_rest`.
- **Classic conversion** per family (`families/classic.py`: `smpl/smplh/smplx/mano/flame_{from,to}_classic`;
  `families/mhr.py:120,154`).
- **Terms** (`terms.py`): `human_state` (fixed or variable identity), `joints`, `points`, `landmarks`, `reprojection`
  with `project(points, camera)`, `ground`, `contact`, `pose_prior`, `shape_prior`, `contact_wrenches`, `root_wrench`
  (the only time-aware term: finite differences on `(B, T, nq)` via `model.difference`, `terms.py:213-223`).
- **IO**: `io/load.py` (`resolve_model_path`, `BETTERHUMAN_MODELS_DIR`), `io/schema.py` (`ModelSchema`, `validate`,
  `format_version` field at `:104`), `io/convert_chumpy.py`.
- **Missing modules** (grep for retarget/measure/transfer/sequence in `src`: nothing): retargeting, measurements,
  family-to-family transfer, motion/sequence type, fitting recipes, STAR/SUPR/Anny/DMPL families.
- **BR reuse is clean.** BH imports BR `lie.so3`/`se3`, `spatial.Inertia`, `forward_kinematics_state`,
  `Model.with_values/difference`, `inverse_dynamics`; `core/classic.py:10-14` wraps `so3.exp` rather than re-deriving.
  No FK, solver or Lie math is duplicated (matches the non-goal in `docs/reference/roadmap.md`).
- **Speed (CPU, SMPL, measured)**: `bh.load("smpl")` 0.48 s; `bake(betas (64,10))` 0.20 s (includes mass);
  `forward + vertices()` for B=64 0.021 s -> `(64, 6890, 3)`.

## How references evolved (with sources)

- **SMPL -> SMPL-H/X -> STAR -> SUPR.** The LBS core never changed; each successor added parameter groups (hand PCA,
  expression) or changed the corrective. STAR made the pose corrective *shape-dependent*: `pose_feat =
  cat(pose_quat[:, 4:], beta[:, 1])` (`references/design/STAR.md:105-109,166`, upstream `star.py:113`). Lesson: the
  corrective input is (pose, identity), not pose alone.
- **Anny** replaced beta-PCA with an interpretable *phenotype* space mapped nonlinearly (1-D piecewise-linear anchor
  interpolation, `references/parametric_body/anny/.../interpolation.py`, `references/design/anny.md:54`) and does
  partial models by flags (`bones_to_remove` + reparent/reweight) and safetensors `ModelData` (`anny.md:303-318`).
- **momentum** (Meta) converged on one sparse `ParameterTransform` from compact model parameters to joint DOFs, and on
  `simplify`/`subsetParameterTransform` returning a smaller `Character` with skin, limits and locators remapped
  (`references/design/momentum.md:476-492`, upstream `character.h:137-192`).
- **SOMA-X** grew an identity-backend ABC (`BaseIdentityModel.get_rest_shape()` + `NATIVE_UNIT/UP/FORWARD` metadata
  and a normalization boundary) so several body models plug in behind one pose/skin pipeline
  (`references/design/SOMA-X.md:529-540`, upstream `identity_model.py:146-274`).
- **GMR** started with positional JSON match tables per (human, robot) pair and MuJoCo/mink IK with warm start from
  the previous frame; the digest recommends a typed per-link task (`w_pos`, `w_rot`, offsets) solved as batched NLLS
  (`references/design/GMR.md:381-400`).
- **soma-retargeter** is the same pipeline on Newton: FK source -> weighted position/rotation residuals per joint pair
  -> LM on robot `q` -> warm start; plus smooth joint-limit and smoothness terms
  (`references/design/soma-retargeter.md:382-398`).
- **mink** maps 1:1 onto BR residuals (frame, posture, limits; `references/design/mink.md:448-469`) — retargeting
  needs no new solver, only terms and a recipe.
- **GarmentMeasurements** uses declarative `.mes` recipes (plane, selection ring, half-space cuts, convex hull), but
  hard `min`/nearest-vertex selections block reuse as an optimizer residual; digest: design the measurement engine
  differentiable from day one, and ship a self-describing weight format (`GarmentMeasurements.md:208-222`).

## Findings (ranked, F1..Fn: severity, evidence, consequence)

- **F1 — high — Correctives cannot see identity.** `Rig.corrective_features(spec, model, q)` (`core/rig.py:61`,
  called at `core/body.py:285`) gets only `q`; `Corrective.kind` is a closed `Literal` (`core/spec.py`). STAR and SUPR
  concatenate a shape coefficient into the feature (STAR.md:105-109). Consequence: adding STAR/SUPR is a protocol
  change touching both rigs, `Body`, `HumanState` and `Corrective`, not a new brick. `models/star` is already present.
- **F2 — high — Shape space is hard-wired to linear blend shapes.** `Body.bake` sums linear bases before calling the
  rig (`core/body.py:~125`). Anny's phenotype space (nonlinear) and SOMA-X-style identity backends cannot be expressed
  as `FamilySpec.identity` data. MHR shows the workaround (nonlinear proportions inside `AuthoredRig.bake`), but that
  only reaches the skeleton, not `v_shaped`. Consequence: Anny needs a core change; the roadmap vision lists Anny.
- **F3 — high (maintainability) — `FamilySpec.meta` is an undeclared contract.** Core code reads three keys and the
  families read ~10 more (list in Current state). A new family author has no typed list of what core needs; a missing
  key fails deep inside `HumanState.skin_affines`, not at construction; `spec.to(dtype, device)` must know which meta
  tensors to move (not verified for every key). Consequence: the "family as data" promise holds only for families
  written by people who read `body.py`.
- **F4 — medium — Closed family registry.** `_LOADERS` is built from `families.__all__`; a user's own scan model or a
  lab-internal family cannot go through `bh.load` without editing BH. The `Rig` protocol is open, the front door is not.
- **F5 — medium — Contact frames coupled to `RegressedRig`.** Verified `ValueError` for MHR. Vertex-anchored frames
  need only `v_shaped`, rest joints and parent joints, which every `RigBake` already returns. Consequence: MHR, the
  family with the best skeleton, has no foot/hand contact frames, so `contact_wrenches`/`root_wrench` dynamics work
  only on SMPL-family bodies.
- **F6 — medium — No fitting brick (BVR's main need).** `terms.py` has the residual bricks, but no staged recipe
  (SMPLify-style: root -> body -> shape -> hands/face), no camera type (`project(points, camera)` takes a raw tensor,
  `terms.py:82`), no robust-loss defaults, no per-frame warm start. Every consumer re-assembles BR `run`/`Phase`.
- **F7 — medium — No motion/sequence type.** `HumanState` is one batch; the `(B, T, nq)` time axis appears only
  implicitly inside `root_wrench` (`terms.py:213-223`). BR has `terms.robot.finite_difference` (`terms/robot.py:114`).
  Clips (AMASS, BVR video), smoothing priors, retargeting and dynamics over time all need one agreed time axis and fps.
- **F8 — medium — Vision modules absent: retargeting, family transfer, measurements.** `docs/reference/roadmap.md`
  promises "conversion between body models and retargeting to humanoid robots are modules over the same core";
  nothing exists in `src`. References show retargeting = frame residuals + LM + warm start (GMR, soma-retargeter, mink),
  i.e. composable from existing BR/BH bricks once F7 exists.
- **F9 — low — `HumanState` accessor style is mixed.** Properties: `joints`, `joint_poses`, `joint_transforms`,
  `skin_affines`, `corrective_feature`; methods with no arguments: `vertices()`, `frame_poses()`. Verified:
  `state.vertices.shape` raises `AttributeError: 'function' object has no attribute 'shape'`. Cheap to fix only
  before the API is public.
- **F10 — low — Asset handling is minimal.** `format_version` exists (`io/schema.py:104`), but no asset manifest
  (expected files, checksums, license/gating note per family) and no conversion provenance; `resolve_model_path` uses
  one env var. Fine now; becomes friction with 10+ families.
- **F11 — low — BH-generic code that could live in BR later.** `vertex_normals` and `frame_local_placements`
  (vertex-anchored frames; useful for robot skins/tactile pads), per-part mesh closing (`core/mass.py:
  build_part_topology`), and `project` (pinhole camera; BVR and robot cameras both need it). LBS/skinning stays in BH.

## Roadmap (table: brick | why | prerequisite | when)

| Brick | Why (user) | Prerequisite in architecture | When |
|---|---|---|---|
| Corrective features take identity | STAR/SUPR users; shape-dependent soft tissue | `corrective_features(spec, model, q, identity_params)` | now |
| Typed `meta` contract (declared keys for core) | family authors | list keys in `FamilySpec` docstring; later typed fields (`BindPose`, `closed_segmentation`) | now (doc) / next (types) |
| Time-axis convention `(B, T, nq)` + fps in BH contracts | clip users, BVR | write it into `docs/contracts/bh-addendum.md`; `root_wrench` already uses it | now |
| `register_family(name, loader)` | labs with own models | open `_LOADERS` | next |
| STAR family (+ DMPL as a stream group) | assets already in `models/` | F1 fixed | next |
| `Motion` type (q `(B,T,nq)`, stream, fps, identity) + smoothness terms | AMASS/BVR clips, retargeting | time-axis convention; reuse BR `finite_difference` | next |
| Fitting recipes (`fit_keypoints`, `fit_video`) as `Phase` lists + `Camera` | BVR, mocap users | camera type (BR `spatial` or BH), F7 | next |
| Rig-agnostic contact frames | MHR dynamics, contact terms | `RigBake` exposes rest joints in skin frame (it does) | next |
| Retarget module (typed match table: link, frame, `w_pos`, `w_rot`, offsets; warm start) | humanoid-robot users (GMR/soma) | `Motion`; BR frame residuals; per-robot table files | later |
| Family transfer (SMPL<->SMPL-X<->MHR via vertex correspondences + fit) | dataset users | fitting recipes; correspondence asset format | later |
| Identity-space protocol (`IdentitySpace.rest_shape(params) -> v_shaped`) | Anny, SOMA-X | linear case stays data; protocol only when a second kind lands | later |
| Anny family (phenotype space, CC0 weights) | users without SMPL licence | identity protocol | later |
| Measurements (declarative recipes, differentiable sections) | apparel, anthropometry (GarmentMeasurements) | landmarks as static spec data (exists) | later |
| Asset manifest (files, checksums, licence gate, provenance) | many-family installs | `io/schema.py` versioning (exists) | later |
| Move vertex-anchored frames / mesh closing / camera to BR | robot skins, BR cameras | a second in-tree BR caller | maybe |
| Clothing / garment layers | apparel | measurements + mesh offsets | maybe |

## Prep changes recommended now (each <= 1 day, non-breaking)

1. **Widen `Rig.corrective_features`** to also receive the baked identity coefficients (keyword-only, default
   `None`); pass them from `HumanState.corrective_feature`. Both in-tree rigs ignore it. Removes the only protocol
   change STAR needs (F1).
2. **Declare the `meta` keys core reads** (`closed_segmentation`, `inverse_bind_poses`, `bind_joint_positions`) in the
   `FamilySpec` docstring and in `docs/contracts/bh-addendum.md`, with shapes; family-private keys stay private (F3).
3. **Write the time-axis rule** (`(B, T, nq)`, time second-to-last of the leading axes, `dt` explicit) into
   `bh-addendum.md`, citing `root_wrench` as the first user (F7).
4. **Decide `HumanState` accessor rule before release** (see open question 1); if "no-arg output = property", make
   `vertices` and `frame_poses` properties now while the only callers are in-tree (F9).
5. **Record the reserved namespaces** (`better_human.retarget`, `better_human.fit`, `better_human.measure`) in
   `docs/reference/roadmap.md` so contributors do not scatter them (F6, F8).

## Open questions for the owner

1. `HumanState.vertices()` / `frame_poses()` are methods while `joints` etc. are properties. Break to properties
   before merge (recommended), or document "methods compute geometry, properties are cheap"?
2. Which family comes next: STAR (assets on disk, needs F1 only) or Anny (licence-free, needs an identity protocol)?
3. Fitting pipelines: in BH (`better_human.fit`, recommended, BVR imports it) or in BVR?
4. Retargeting: a BH module or a separate package depending on BH + robot URDFs?
5. Should BH allow third-party families through `register_family`, or keep families in-tree during alpha?
