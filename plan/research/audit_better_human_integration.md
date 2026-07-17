# Audit: better_human integration — what BetterRobot needs to host parametric body models

Auditor: read-only architecture audit, 2026-07-16.
Question: better_human will be rewritten **on top of** BetterRobot (BR) so kinematics/dynamics
come from BR natively while parametric-body specifics (betas, blendshapes, LBS, meshes) stay
in better_human. What must BR provide for that to be elegant?

---

## 1. Scope & method

### 1.1 What was read

- **better_human** (`/data3/rikhat.akizhanov/better/better_human`): all of
  `src/better_human/{core,smpl,urdf,utils}` (963 LOC core + 840 LOC model variants),
  `smpl/config/*.json`, `scripts/convert.py`, demos, `pyproject.toml`, README, models/ layout.
- **BR** (`/data3/rikhat.akizhanov/better/BetterRobot`): `data_model/model.py`, `data.py`,
  `frame.py`, `joint_models/{spherical,free_flyer,base}.py`, `kinematics/{forward,jacobian}.py`,
  `dynamics/rnea.py` (+ grep of aba/crba/centroidal), `spatial/inertia.py`,
  `io/{build_model.py,parsers/programmatic.py,builders/{kinematic_tree,smpl_like}.py}`,
  `residuals/{limits,regularization,pose}.py`, `docs/reference/roadmap.md`,
  `references/design/BEST_PRACTICES.md`, `references/design/pyroki.md`.
- **Consumers**: `BetterVideoReconstruction/tools/smplx_robot/{model,inertia}.py`,
  `tools/human_optim/{optimizer,losses,motion}.py` (headers + key functions);
  `BetterHumanForce/tools/robot_motion/{dynamics,smpl_inertia,motion}.py`.

### 1.2 Experiments run (raw results)

All via `uv run python` inside BetterRobot, CPU, float32, `make_smpl_like_model()`
(25 joints incl. universe, nq=99, nv=75):

| # | Experiment | Result |
|---|-----------|--------|
| EXP1 | `dataclasses.replace(model, joint_placements=leaf.requires_grad_())` → FK → `.backward()` | **grad flows**: `placements.grad` non-None, 125 nonzero entries |
| EXP2 | `joint_placements` shaped `(4, njoints, 7)` (per-batch shape) → `forward_kinematics_raw` | **crash**: `RuntimeError: The size of tensor a (25) must match the size of tensor b (4) at non-singleton dimension 0` |
| EXP3 | `body_inertias` as autograd leaf → `rnea` → `.backward()` | **grad flows**: 72/250 nonzero entries |
| EXP4 | `body_inertias` shaped `(4, nbodies, 10)` → `rnea` | **crash**: same dim-0 mismatch |
| EXP5 | `model.integrate` / `model.difference` roundtrip on free-flyer + 23 spherical joints | max err **1.07e-8** — manifold ops correct |
| EXP6 | Input SMPL kintree order vs built `model.joint_names` | **DFS reorder confirmed**: input `(pelvis, left_hip, right_hip, spine1, …)` → model `(root, left_hip, left_knee, left_ankle, left_foot, right_hip, …)` |

### 1.3 better_human as it stands (background)

A small pypose-based (`pyproject.toml` deps: `numpy, torch, pypose, viser, trimesh`)
`torch.nn.Module` library:

- `core/humanoid.py` (56 lines): near-empty ABC — one abstract method, the rest commented out.
- `smpl/base.py` (`SMPLBase`): registers `v_template (V,3)`, `shapedirs (V,3,nb)`,
  `posedirs`, `J_regressor (J,V)`, `lbs_weights (V,J)`, kintree (base.py:52-74).
  **Its q layout already mirrors BR**: `[0:7]` free-flyer, then 4-wide quaternion blocks per
  spherical joint (`joint_q_idx`/`joint_v_idx`, base.py:109-113). Forward =
  `forward_shape(betas)` → shaped verts + regressed joints (base.py:196-211), pose
  blendshapes (base.py:213-224), hand-rolled 4×4-matrix FK loop (base.py:157-194), LBS
  (base.py:226-271).
- `smpl/kinematics.py`: hand-rolled world Jacobians via adjoint packing + a `(J, nv)`
  `J_mapping` mask (kinematics.py:6-24), Jacobian time-derivative, LOCAL/LWA conversions
  (LWA derivative `NotImplementedError`, kinematics.py:80).
- `smpl/dynamics.py`: hand-rolled pypose RNEA with dict-based per-joint state
  (dynamics.py:8-131) — duplicates BR's `rnea`.
- `smpl/mass.py`: per-body mass/COM/inertia by building a `trimesh.Trimesh` per body per
  batch element in a Python double loop, with `.detach().cpu().numpy()` (mass.py:16-46) —
  non-differentiable, CPU-bound.
- `smpl/sequence.py`: `(B, T)` sequence wrapper; finite-difference velocities/accelerations
  split per joint type (sequence.py:55-117) — duplicates BR `model.difference` /
  `AccelerationResidual`; assembles 6×6 spatial inertias (sequence.py:135-147).
- `smpl/visualize.py`: viser skinned-mesh visualization via `add_mesh_skinned` with
  `lbs_weights` + per-bone world transforms (visualize.py:104-114).
- `smpl/model/*.py`: SMPL(24 joints), SMPL-H, SMPL-X(55), SMPLXMini, **SMPLXMid(52)** — a
  variant purpose-built to "map directly onto a BetterRobot kinematic model"
  (smplx_mid.py:11-28), MANO, STAR; `flame.py` is empty. `from_classic`/`to_classic`
  convert the standard smplx (betas, body_pose axis-angle, transl, global_orient)
  convention to the q layout (smpl.py:48-92).
- `smpl/config/*.json`: topology, joint names, **landmark frames** (`frame_names` = nose,
  eyes, ears, fingertips, toes, heels with `frame_vertex_ids` — vertex-anchored keypoints),
  vertex segmentation, per-body density tables (Dempster/Chandler/Clauser/Custom).
- `urdf/` subpackage: **empty** — a 0-line `__init__.py`, nothing else; no consumer imports
  `better_human.urdf`.
- `scripts/convert.py`: chumpy `.pkl` → `.npz` conversion (data prep, stays).

**How consumers use it today** (both already bridge to BR by hand):

- BVR `tools/smplx_robot/model.py:59-84`: builds a BR `Model` from SMPLXMid via
  `build_kinematic_tree_model(root_kind="free_flyer", child_kind="spherical")` at **fixed
  betas** (offsets computed under `torch.no_grad()`, model.py:48); then needs 40 lines of
  q-remap shims (`remap_q_bh_to_model`/`remap_q_model_to_bh`, model.py:87-128) with a
  per-joint Python loop per call because BR reorders joints DFS.
- BVR `tools/smplx_robot/inertia.py:88-117`: per-body convex-hull mass/COM/inertia via
  trimesh in a Python loop, renormalized to 70 kg; feeds `build_kinematic_tree_model`
  (inertia.py:168-180).
- BHF `tools/robot_motion/smpl_inertia.py:136-149`: same pattern with face segmentation +
  density tables; `dynamics.py:23-82` rebuilds the inertial BR model from an `.npz`;
  `dynamics.py:133-181` computes v/a via `model.difference` central differences.
- BVR `tools/human_optim/optimizer.py:1-26`: the kinematic optimization loop runs
  **better_human forward per iteration** (vertices + joints) with hand-rolled Adam/L-BFGS;
  losses use posed **vertices** for landmark reprojection
  (`vertices_world[:, frame_vertex_ids[...]]`, losses.py:63) and chamfer.

---

## 2. Confirmed problems

### CP-1. `Model` cannot carry per-sample (batched) joint placements — HIGH

- Evidence: `Model.joint_placements: (njoints, 7)` (model.py:63); FK indexes dim 0:
  `T_placement = model.joint_placements[j]` (kinematics/forward.py:112). EXP2 crash with
  `(B, njoints, 7)`.
- Why it matters: SMPL joint locations are a **differentiable function of betas**
  (`J_regressor @ (v_template + shapedirs·betas)`, better_human base.py:206-209). A batch of
  people = a batch of skeletons with different bone lengths. Today a BR model can represent
  exactly one shape; BVR bakes betas in under `no_grad` (smplx_robot/model.py:48), so
  **shape cannot be optimized through BR FK** and multi-person batches need one Model each.
- Note the near-miss: `se3.compose` broadcasts (`(…,7)×(…,7)`,
  lie/_torch_native_backend.py:238-247) and EXP1 shows autograd already flows through
  `joint_placements` — the blocker is only the `[j]` (dim-0) indexing convention, in FK and
  nowhere else (Jacobians consume only `data.joint_pose_world`, jacobian.py:30-79).

### CP-2. Same for `body_inertias` in every dynamics recursion — HIGH

- Evidence: `Inertia(model.body_inertias[i]…)` at rnea.py:167, crba.py:60, aba.py:121,
  centroidal.py:41 and 152. EXP4 crash with `(B, nbodies, 10)`.
- Why it matters: per-body inertia derives from body shape (betas) — three separate
  hand-rolled implementations exist (better_human `mass.py`, BVR `inertia.py`, BHF
  `smpl_inertia.py`). Per-sample dynamics (RNEA torques per person) is a core BHF/BVR use
  case. The `Inertia` value type itself is already batch-shaped `(...,10)`
  (spatial/inertia.py:18-21) — again only the indexing convention blocks it. EXP3 shows
  autograd to inertias already works unbatched.

### CP-3. `build_model` force-reorders joints DFS; no order-preserving option — MEDIUM-HIGH

- Evidence: `_ir_topo_sort` is an unconditional DFS (io/build_model.py:160-197). EXP6:
  SMPL kintree order (which IS already topological — parents before children) comes out
  permuted. Consequence in the wild: BVR `remap_q_bh_to_model` / `remap_q_model_to_bh`
  (smplx_robot/model.py:87-128) — per-joint Python loops executed on every conversion, plus
  the same q-order impedance documented in BVR `human_optim/motion.py:12` and BHF
  `motion.py`.
- Why it matters: the rewritten better_human will define q in SMPL order (that's what every
  external dataset/checkpoint uses). If BR permutes it, every boundary crossing pays a remap
  and every off-by-one is a silent pose corruption risk.

### CP-4. No public API for "template Model + per-sample parameters" — HIGH (design gap)

- Evidence: the only ways to set placements/inertias are the builder pipeline
  (`ModelBuilder` → IR → `build_model`), which is Python-heavy, partially
  autograd-breaking (`float(m)` casts of masses, builders/kinematic_tree.py:36-44;
  `masses.tolist()` in BVR inertia.py:175), and rebuilds topology every call — or
  raw `dataclasses.replace` on a frozen class (what EXP1/EXP3 did), which is undocumented,
  unvalidated, and not part of any contract. `Data._model_id = id(model)`
  (model.py:133) is stored but never checked anywhere (grep: no consumer), so replace
  happens to work by accident.
- Why it matters: this is *the* integration primitive better_human needs:
  build the 52-joint topology **once**, then per optimization step rebind
  betas-derived placements/inertias without re-running the builder. Today that path is
  neither designed nor tested.

### CP-5. Fixed `(7,)` frame placements can't express shape-dependent markers — MEDIUM

- Evidence: `Frame.joint_placement: torch.Tensor  # (7,)` frozen struct
  (data_model/frame.py:21-28); `update_frame_placements` and `get_frame_jacobian` read
  `frame.joint_placement` per frame (forward.py:215, jacobian.py:200). better_human's
  landmarks (nose/ears/fingertips/heels — `frame_names` + `frame_vertex_ids` in
  smpl/config/smpl.json) are positions on the *shaped* mesh: their body-local offset depends
  on betas. BVR's reprojection losses consume them via posed vertices (losses.py:63).
- Why it matters: marker-based costs (keypoint reprojection, marker IK, mocap retargeting —
  cf. `references/design/BEST_PRACTICES.md` §7.5 "marker IK as a residual") are the
  bread-and-butter residuals for humans. With per-sample frame placements, BR's existing
  `PositionResidual` + analytic frame Jacobian machinery would serve them directly; without,
  better_human must bypass BR's residual/Jacobian layer for every marker term.

### CP-6. Joint limits are meaningless for spherical joints; no rotation-prior residual — MEDIUM

- Evidence: build packs spherical position limits as `[-1, 1]` per quaternion component
  (io/build_model.py:390-397) — inert, since unit-quaternion components never leave
  [-1, 1]. `JointPositionLimit` explicitly zeroes Jacobian rows for `nq != nv` joints
  (residuals/limits.py:19-55). So for an all-spherical human, the entire limit residual is
  a no-op.
- Why it matters: anatomical joints need swing/twist cone limits or pose priors. BVR
  hand-rolls `pose_prior_loss` / `root_orient_prior_loss` (human_optim/losses.py imports)
  outside BR. Note `RestResidual` *does* work for spherical joints (tangent-space
  `model.difference`, regularization.py:21-65) — so a mean-pose prior is already
  expressible; per-joint weighting and swing-twist limits are not.

### CP-7. AUTO Jacobian fallback = central FD, O(2·nv) full FK sweeps — MEDIUM (for humans)

- Evidence: `residual_jacobian` FD fallback loops `for i in range(model.nv)` with two full
  FK calls each (kinematics/jacobian.py:266-284); it also builds `v0` unbatched `(nv,)` and
  returns `(dim, nv)` ignoring batch.
- Why it matters: SMPLXMid has nv=159 → **318 FK passes of a 52-joint model per residual
  per iteration**, per batch element. Fine for a 7-DOF Panda, a scalability trap for
  humans. Custom better_human residuals (reprojection, priors) won't have analytic
  Jacobians on day one, so the default path matters.

### CP-8. Hot-path per-joint `.to(device,dtype)` and sync-y validation — LOW-MEDIUM (noted; perf audit's domain)

- Evidence: `model.joint_placements[j].to(device=…, dtype=…)` inside the FK loop
  (forward.py:112), `model.body_inertias[i].to(…)` inside RNEA/CRBA/ABA loops
  (rnea.py:167 etc.); `_validate_q` does `bool(((norm-1).abs() > tol).any())` — a
  GPU→CPU sync — and runs **twice** per public FK call (forward.py:64; called at
  forward.py:97 inside `forward_kinematics_raw` and again at forward.py:181 in
  `forward_kinematics`). At human scale (52 joints × per-iteration FK inside an optimizer)
  these per-joint fixups multiply.

### CP-9. pypose dependency mismatch — LOW (planned rewrite absorbs it)

- Evidence: better_human is pypose throughout (`base.py:2`, `utils/lie.py`); BR removed
  pypose (BetterRobot/CLAUDE.md "PyPose is no longer a dependency").
- The rewrite must port `SE3_Adj`, `LieDifference`, `SO3_2_SE3`, `se3_adj_dual`
  (utils/lie.py:6-66) to `better_robot.lie.se3/so3` — all have 1:1 equivalents
  (`se3.adjoint`, `so3.log/compose/inverse`), so this is mechanical.

---

## 3. Suspected problems needing verification

- **S-1. Broadcasting contract for `(B, T)` problems.** If placements are `(B, njoints, 7)`
  and q is `(B, T, nq)`, naive `[..., j, :]` indexing gives `(B, 7)` vs `(B, T, 7)` —
  broadcast aligns from the right and would mispair B with T. The fix needs an explicit
  rule (model batch dims must be right-aligned-broadcastable against q batch dims, i.e.
  caller passes `(B, 1, njoints, 7)`). Untested; must be part of the contract tests if R1
  is adopted.
- **S-2. LM/GN scalability at human dimensions.** nv=159 with T=100 knots → 15,900
  variables; dense normal equations are ~2 GB at float32. Whether `SparseCholesky` /
  block-sparse paths in `optim/` actually exploit trajectory structure is out of scope here
  (optim audit) but is a hard prerequisite for BR replacing BVR's Adam/L-BFGS loop with LM.
- **S-3. torch.compile behavior at 52+ joints.** The FK loop unrolls per joint; roadmap
  says `@torch.compile(fullgraph=True)` is "not yet applied" (roadmap.md, Performance).
  Compile time and graph size for a 52-joint unroll are unmeasured.
- **S-4. `Data._model_id`** is set (model.py:133) but never validated (grep across src/ and
  tests/ finds no consumer). Either enforce it (then `with_params` must handle it) or drop
  it. Currently dead weight that would *appear* to forbid the replace-based pattern while
  actually not checking anything.
- **S-5. `builders` autograd.** `_origin_from_translation` writes `o[:3] = xyz`
  (kinematic_tree.py:29-33) — probably differentiable w.r.t. `xyz` via index_put, but the
  mass path (`float(m)`) is definitely not. Nobody should rely on builder differentiability
  either way; verify or explicitly document it as non-differentiable.

---

## 4. What is actually good and should be kept

- **`JointSpherical` and `JointFreeFlyer` are real and correct.** Quaternion manifold
  `integrate`/`difference` (joint_models/spherical.py:52-62) verified to 1e-8 (EXP5).
  BR *does* have ball joints — the SMPL joint model maps 1:1 (free-flyer root + N spherical),
  and better_human's q layout was already designed to match (base.py:109-113).
- **The programmatic builder exists and is battle-tested by two real projects.**
  `build_kinematic_tree_model(joint_names, parents, translations, root_kind, child_kind,
  mass/com/inertia_per_body)` (io/builders/kinematic_tree.py:105-230) is exactly the right
  primitive; BVR (smplx_robot/model.py:74) and BHF (robot_motion/dynamics.py:70) both build
  24- and 52-joint humans through it today. `make_smpl_like_body` (builders/smpl_like.py)
  correctly keeps SMPL specifics out of the core.
- **FK and RNEA are autograd-clean through model tensors** (EXP1/EXP3). This is the single
  most important enabler: differentiable-shape support is an *indexing + API* change, not an
  algorithm rewrite. FK already uses list-accumulate + stack precisely to keep autograd
  clean (forward.py:79-135).
- **Jacobians are placement-agnostic.** `_compute_joint_jacobians_raw` and
  `get_frame_jacobian` consume only `data.joint_pose_world` + motion subspaces
  (jacobian.py:30-79, 129-227) — once FK accepts batched placements, all analytic Jacobians
  (and therefore all pose/position residuals) are correct with zero changes.
- **`Inertia` is batch-shaped by construction** (`(...,10)` packed, spatial/inertia.py) with
  the parallel-axis shift inside `_to_6x6` — the convention consumers already target
  (BHF smpl_inertia.py:19-21 docstring).
- **`model.difference`-based residuals already serve humans**: BHF derives v/a for RNEA via
  `model.difference` central differences (robot_motion/dynamics.py:133-181), replacing
  better_human's entire `sequence.py`; `RestResidual` is a working mean-pose prior for
  spherical joints.
- **`model.meta` escape hatch** (model.py:88) — the right place for better_human to stash a
  back-reference to its parametric data without BR knowing about it (pattern already used
  for `asset_resolver` / `ir`).
- **In better_human, keep (do not move into BR):** LBS + pose/shape blendshapes, PCA hand
  spaces, the model zoo + config JSONs (segmentations, densities, landmark tables),
  chumpy→npz conversion, `from_classic`/`to_classic` convention adapters. All of
  better_human's *duplicated* robotics code — FK (base.py:157-194), Jacobians
  (kinematics.py), RNEA (dynamics.py), sequence FD (sequence.py) — should be deleted in the
  rewrite; BR's equivalents are strictly more capable (better_human's own LWA Jacobian
  derivative is `NotImplementedError`, kinematics.py:80).

---

## 5. Recommendations (prioritized)

### R1 — Batched, rebindable kinematic/inertial parameters on `Model` (P0, effort M, risk: medium)

The one structural change everything else hangs off. Two parts:

**(a)** Change parameter indexing in the recursions from dim-0 to trailing-dim:
`model.joint_placements[..., j, :]` (forward.py:112) and
`model.body_inertias[..., i, :]` (rnea.py:167, crba.py:60, aba.py:121,
centroidal.py:41,152). With the existing broadcasting `se3.compose` this makes
`(B…, njoints, 7)` placements work through FK, Jacobians, and all dynamics unchanged.

**(b)** Add a validated public rebind API instead of raw `dataclasses.replace`:

```python
class Model:
    def with_params(
        self,
        *,
        joint_placements: Tensor | None = None,   # (..., njoints, 7)
        body_inertias:   Tensor | None = None,    # (..., nbodies, 10)
        frame_placements: Tensor | None = None,   # (..., nframes, 7)  [with R3]
    ) -> "Model":
        """Same topology, new parameter tensors. Shapes validated; quats
        renormalized; autograd flows to the inputs. O(1), no rebuild."""
```

better_human's rewrite then becomes:

```python
class SMPLXMid:
    def __init__(self, npz):
        self.template = build_kinematic_tree_model(...)     # topology once
        ...buffers: v_template, shapedirs, J_regressor, lbs_weights...

    def bind(self, betas):                                   # (B, nb)
        joints = self.rest_joints(betas)                     # (B, J, 3), differentiable
        placements = placements_from_rest_joints(joints, self.template)  # (B, njoints, 7)
        inertias = self.inertia(betas)                       # (B, nbodies, 10), see R4
        return self.template.with_params(joint_placements=placements,
                                         body_inertias=inertias)

model = smplx.bind(betas)                # betas is an autograd leaf
data  = br.forward_kinematics(model, q)  # d(loss)/d(betas) flows (EXP1/EXP3 prove core path)
```

Alternative considered: pass `joint_placements=` as a runtime kwarg to
`forward_kinematics`/`rnea`. Rejected: every downstream consumer (residuals, CostStack,
solve_ik, viewer) takes `model` implicitly, so a kwarg would have to be threaded through
the entire stack; `with_params` needs zero signature changes.
Risks: the (B, T) broadcasting contract (S-1) must be written down and contract-tested;
`Data._model_id` (S-4) must be resolved (recommend: validate against topology identity, not
`id()`).

### R2 — Order-preserving model build + q-layout adapters (P0, effort S, risk: low)

- `build_model(ir, preserve_order=True)` (or make `_ir_topo_sort` a *stable* Kahn sort that
  keeps input order whenever it is already topological — SMPL kintree order always is).
  Kills `remap_q_bh_to_model` and friends (BVR smplx_robot/model.py:87-128).
- Add a vectorized permutation helper for the cases where remapping is still needed:

```python
perm_q, perm_v = model.q_permutation(other_joint_order: Sequence[str])
q_br = q_ext[..., perm_q]     # one gather, no per-joint Python loop
```

Risk: tests that pin DFS order; keep DFS as default, opt-in flag.

### R3 — Frame placements as a rebindable tensor table (P1, effort M, risk: low)

Move per-frame `(7,)` placements from `Frame` structs into
`model.frame_placements: (..., nframes, 7)` (Frame keeps name/parent/type metadata);
`update_frame_placements` (forward.py:213-216) and `get_frame_jacobian` (jacobian.py:200)
read the table. Combined with R1(b) this gives shape-dependent **markers/sites**: better_human
regresses each landmark's body-local offset from betas once per bind, and every existing
`PositionResidual`/`PoseResidual` + analytic Jacobian works on markers unmodified — i.e.
marker IK and keypoint reprojection targets ride the standard residual stack
(BEST_PRACTICES §7.5). Vertex-exact landmarks (pose-dependent skinned position) stay in
better_human; the rigid body-local approximation is what a rigid-body library can and should
offer.

### R4 — Differentiable mesh inertia in `spatial/` (P1, effort M, risk: medium-numerics)

Three hand-rolled trimesh loops exist (better_human mass.py:27-46 — B×J Python loop with
`.detach().cpu().numpy()`; BVR inertia.py:88-117; BHF smpl_inertia.py:136-147). The
underlying math (signed tetrahedron volume integrals over faces) is closed-form and
batches trivially in torch:

```python
Inertia.from_mesh(vertices: (..., V, 3), faces: (F, 3), density: float | Tensor)
    -> Inertia   # (..., 10); autograd flows to vertices → betas
inertia_from_vertex_parts(vertices, faces_per_body | vertex_to_body, densities,
                          joints) -> Tensor  # (..., nbodies, 10), COMs joint-relative
```

This makes inertia a differentiable function of betas (feeding R1's `body_inertias`),
removes the trimesh dependency from the hot path, and unifies the density-table and
convex-hull-renormalization variants as caller policy. Risk: numerical parity — test
against trimesh on the same meshes (the two consumer implementations are the fixtures).

### R5 — Ball-joint priors and limits (P1, effort S–M, risk: low)

- `JointRotationPrior(model, q_mean, per_joint_weight)` — tangent-space deviation like
  `RestResidual` but with a `(njoints,)` or `(nv,)` weight vector (humans need strong spine,
  weak shoulder priors).
- `SwingTwistLimitResidual(model, twist_axis, swing_max, twist_range)` — the anatomically
  meaningful limit for spherical joints, since quaternion box limits are inert
  (build_model.py:390-397, limits.py:19-55).
- Document the pattern "learned pose prior (GMM/VPoser) = custom Residual over q slices" —
  the Residual protocol already admits it; better_human ships the priors.

### R6 — Default Jacobian strategy guidance + batched FD (P2, effort S, risk: low)

For nv≳50 models the AUTO→FD fallback (jacobian.py:266-284) is 2·nv FK sweeps and silently
unbatched. Either batch the FD probe (`(nv, nv)` perturbation matrix through one batched FK)
or make AUTO prefer autodiff above an nv threshold; at minimum document the trap.

### R7 — Viewer `SkinnedMeshMode` (P2, effort S, risk: low)

viser natively supports `add_mesh_skinned` (better_human visualize.py:104-114 already drives
it with lbs_weights + per-bone world transforms). A BR render mode taking
`(vertices_rest, faces, skin_weights)` and updating bones from `data.joint_pose_world` gives
better_human visualization for free and BR a differentiator. No mesh math enters BR core.

### R8 — Retarget API design should be marker/frame-based (P2, design note)

`tasks/retarget.py` is a stub (roadmap.md, Tasks). better_human's references include GMR
(references/retarget/GMR). When designed, `retarget(source_model, target_model,
frame_correspondences)` over R3-style markers subsumes human→robot and human→human
retargeting; don't design it URDF-robot-only.

### R9 — Hot-loop hygiene for large nv (P2, effort S; coordinate with perf audit)

Hoist per-joint `.to(device,dtype)` out of the FK/RNEA loops (do one `model.to()` check at
entry), and drop the duplicate `_validate_q` (forward.py:97 vs 181) or make the
sync-inducing quaternion check optional (`validate="once"|"never"`).

### Explicit non-goals (keep out of BR)

LBS/skinning math, blendshapes, joint regressors, PCA hand/face spaces, SMPL file formats,
density tables, chumpy conversion, `from_classic` adapters — all stay in better_human. BR
provides: topology + parameters (R1/R3), spherical-joint manifold math (exists), FK/Jacobians
/dynamics (exist), residual/solver stack, inertia-from-mesh (R4), viewer hooks (R7).

---

## 6. Open questions

1. **Broadcast contract shape** (S-1): should `with_params` require model batch dims to be
   *left-aligned* with q batch dims (auto-unsqueeze internally, e.g. `(B, njoints, 7)` +
   `(B, T, nq)` → placements viewed as `(B, 1, njoints, 7)`), or push alignment onto the
   caller? Auto-alignment is friendlier but adds a rule; explicit is safer. Needs a decision
   before R1 lands.
2. **One Model per shape vs one shared template**: is `with_params` returning a new frozen
   `Model` acceptable to `torch.compile` plans (guards on tensor identity), or should
   parameters eventually live in `Data`/a separate `ModelParams` pytree (pyroki-style split
   of static structure vs. leaf arrays, references/design/pyroki.md §2)? R1(b) is compatible
   with both; the compile story should pick the final home.
3. **Where do q-convention adapters live**: `from_classic`/`to_classic` and axis-angle↔quat
   conversions — better_human-owned (recommended), or does BR want a generic
   `q_permutation`/`q_convert` utility layer (R2 covers only permutation)?
4. **Pose blendshapes need per-joint local rotation matrices** — better_human can compute
   them from q slices directly (`so3.to_matrix` on each 4-wide block); does BR want to expose
   `data.joint_rotation_local` as a convenience, or is that needless surface?
5. **Spherical joint limits in IR**: should `IRJoint` grow optional swing/twist limit fields
   so MJCF (which can express them) round-trips, or are limits purely residual-level (R5)?
6. **`Data._model_id`** (S-4): enforce or delete?
7. **Scale**: BVR fits a global body `scale` about the pelvis in addition to betas
   (inertia.py:153-156, mass ∝ s³ / inertia ∝ s⁵ folded by pre-scaling the rest mesh). Is
   uniform scale a better_human concern (fold into placements/inertias before `bind`) or
   worth a first-class BR hook? Recommend: better_human concern; R1 makes it trivial.
