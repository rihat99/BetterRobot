# BetterHuman target architecture — design study (2026-10-06)

Inputs: `subagents/scan_better_human.md` (code audit, cited as "scan §n"), reference
summaries in `/home/rikhat.akizhanov/better/BetterHuman/references/design/*.md` (grepped,
not read in full; treated as prior notes), 3 web searches. Evidence tags:
[code] = scan cites path:line; [ref] = reference summary line; [web] = searched today;
[est] = my estimate, not measured. Numbers without a tag are from scan §2 (contracts.md,
GPU unnamed).

## 1. What the best libraries do that BetterHuman (BH) does not

Limited to motion reconstruction, dynamics, retargeting.

| Capability | Who does it | BH today |
|---|---|---|
| Identity vs per-frame blend groups (expression streamed, not baked) | smplx (expression is a forward input), momentum (identity PCA + face-expression blend shapes, `blend_shape.h` [ref momentum.md:63]) | expression is a shape group: per-frame face = full re-bake incl. inertias [code smplx.py:136,141] |
| Model params -> joint params as a sparse linear map | momentum `parameter_transform.h` + pseudo-inverse [ref momentum.md:59,69]; SOMA procedural twist transform [ref SOMA-X.md:79] | MHR only, via BR mimic/reduced coords reached through a private path [code mhr/deformation.py] |
| Sparse / top-K skinning | MHR sparse weights [ref MHR.md:11]; SOMA warp sparse LBS kernel [ref SOMA-X.md:88]; anny warp LBS + DQS [ref anny.md:50-51] | dense (V,J) einsum [code pipeline.py:176-191] |
| Sparse, local pose correctives | STAR, SUPR (sparse correctives + sparse regressor with bias) [ref STAR.md:6-9, SUPR.md:5-6,113] | dense posedirs, 61 MB read per SMPL-X call (scan §2) |
| FK+LBS cache in one runtime object | SOMA `BatchedSkinning` [ref SOMA-X.md:86] | FK in BR, skinning in BH, quat->matrix round trip per call [code pipeline.py:44,176] |
| Canonical topology bridge across families | SOMA (one topology for SMPL-X, MHR, anny...) [ref SOMA-X.md:5,16; web arXiv 2603.16858] | none |
| Markers / mocap fitting (C3D, virtual markers, marker tracking) | momentum (marker tracking, C3D IO) [ref momentum.md:6,41] | vertex landmarks only (scan §5.8) |
| Robust losses in the fitting stack | momentum Barron generalized loss [ref momentum.md:55] | none in BH (BR optim has none per context) |
| Structured GN solver over sparse error functions | momentum (column-subset QR) [ref momentum.md:54] | none; BVR builds its own |
| Pose prior, joint limits | SMPLify/VPoser (GMM / VAE), momentum limit terms [unverified], mink limit tasks [ref mink.md] | none (scan §5.8) |
| Biomechanical skeleton (hinge knees, scapula, 46 DoF) | SKEL / BSM, SMPL shape space, OpenSim-like joints [web skel.is.tue.mpg.de, ACM TOG 10.1145/3618381] | none; SMPL ball joints only |
| Human->robot retargeting | GMR: two-stage frame-task IK on mink, height scale tables [ref GMR.md:14,152-160]; soma-retargeter: batched LM over clips (batch axis = clips) on Newton [ref soma-retargeter.md:69-77]; OmniRetarget: interaction mesh, Laplacian deformation + hard kinematic constraints [web arXiv 2509.26633] | none |
| Ground contact / anti-skate / penetration terms | OmniRetarget (contact preservation), GMR foot/ground handling [ref GMR.md:14] | contact frames exist, no terms [code contact_frames.py] |
| Rigid-body dynamics, inertias from mesh, ID with contacts | none of the above (anny: "no CRBA/RNEA/ABA" [ref anny.md:25]); OpenSim/AddBiomechanics do | inertias + BR RNEA work; no contact-force split. This is BH's differentiator. |

Takeaway: BH's unique asset is "a real dynamics model with per-subject
differentiable inertias" (scan §7). Everything else in the table is catch-up, and most
of it (markers, priors, limits, contact, retarget) is optim terms, not body code.

## 2. Target package layout and core types

### 2.1 Tree (target sizes; total core ~3.5k lines, apps separate)

```
better_human/
  __init__.py              ~30   load(), Body, Identity, BodyOutput, families registry
  body.py                 ~250   Body (static, per family+options) and its forward()
  identity.py             ~150   Identity (per-subject tensors) + bake()
  family.py               ~150   FamilySpec (pure data) + CorrectiveKind/SkinKind enums
  registry.py              ~40   @register / load(name, path, **options)
  ops/                            pure tensor functions, documented shapes, no classes
    blend.py               ~60   dense + sparse blend shapes
    regress.py             ~50   dense / sparse(+bias, SUPR) joint regression
    correctives.py        ~180   linear (R-I), sparse-linear (STAR/SUPR), MLP (MHR)
    skinning.py           ~200   dense LBS, top-K gather LBS, subset LBS, DQS (later)
    params.py              ~80   sparse param->joint transform (momentum/MHR style)
  families/                       data loaders + small policies, one file each
    smpl.py               ~200   SMPL, SMPL-H, MANO (same policy, different tables)
    smplx.py              ~150   SMPL-X (expression = stream group, hand PCA)
    star.py / supr.py   ~80 each sparse correctives / sparse regressor
    mhr.py                ~300   MHR (param transform, MLP correctives)
    skel.py               ~250   SKEL/BSM (biomech joints; later)
    tables/*.json                joint names, landmark sets, part maps (data, not code)
  markers.py              ~200   MarkerSet: vertex, barycentric, joint-offset markers; named sets
  contact.py              ~150   ContactSet: sole/palm patches as frames + normals
  mass.py                 ~180   vectorized segment inertias (one batched pass, no per-part loop)
  terms/                  ~500   human residuals on BR optim (see 2.6)
  dynamics.py             ~200   inverse dynamics with contact-force split (see 2.7)
  retarget/               ~450   human->robot retargeting (see 2.5)
  io/                     ~400   npz load + schema, classic codecs, AMASS/C3D readers
  viz/                    ~400   layers on BR viewer: SkinnedMesh, Markers, Contacts, Segments
apps/studio/             ~1.2k   editors only (segment, contact-frame, landmark), on BR viewer
tools/convert, tools/parity      offline, unchanged
benchmarks/              ~500   suite of 2.3 below
```

### 2.2 Core types

```python
@dataclass(frozen=True)
class FamilySpec:                     # pure data, loaded from npz + tables
    name: str
    tree: TreeSpec                    # parents, joint kinds, names (BR public builder input)
    template: Tensor                  # (V, 3)
    faces: Tensor                     # (F, 3)
    identity_basis: dict[str, Tensor] # {"betas": (V,3,S)} -> baked once
    stream_basis: dict[str, Tensor]   # {"expression": (V,3,E)} -> per frame
    regressor: Regressor              # dense (J,V) or sparse (+bias)
    skin: SkinWeights                 # top-K: idx (V,K) int, w (V,K)
    corrective: Corrective            # kind + arrays (dense posedirs / sparse / MLP)
    param_transform: SparseMap | None # model params -> joint params (MHR, SKEL)
    landmarks: MarkerSet; contacts: ContactSet; parts: PartMap

class Body:                           # one per (family, options); holds BR Model structure
    spec: FamilySpec; model: br.Model
    def bake(self, **identity: Tensor) -> Identity: ...
    def forward(self, ident: Identity, q: Tensor, *, stream: dict[str, Tensor] = {},
                outputs: Outputs = Outputs.VERTICES) -> BodyOutput: ...
    def classic_to_q(self, classic) -> Tensor;  def q_to_classic(self, q) -> Classic

@dataclass(frozen=True)
class Identity:                       # leading batch dims allowed, differentiable
    values: br.ModelValues            # placements, inertias, frame placements
    v_shaped: Tensor                  # (..., V, 3)
    rest_joints: Tensor               # (..., J, 3)
    joint_stream_jac: dict[str, Tensor]  # (J,3,E): d(rest_joints)/d(stream coeffs)
    inertias: Tensor | None           # lazy; computed on first dynamics use

@dataclass
class BodyOutput:
    data: br.Data; joints: Tensor     # (..., J, 3) posed
    vertices: Tensor | None           # (..., V, 3) only if requested
    markers: Tensor | None            # (..., M, 3) skinned on the marker subset only
```

Policies become enums + arrays (`CorrectiveKind.{NONE, LINEAR, SPARSE, MLP}`), dispatched
in `ops/`, replacing stored callables + `object.__new__` cloning [code body.py:105-108,199-206]
and the `getattr(body, "contact_frames")` duck typing [code shared.py:145]. Adding SMPL-H or
MANO = a table file + a 20-line loader. STAR/SUPR = loader + `SPARSE` kind (already planned
in ops). SKEL needs `param_transform` and non-spherical joint kinds — the BR builder must
support revolute/universal joints with custom axes (it does for MHR).

### 2.3 Per-frame stream groups (expression, MHR scale)

Upstream SMPL-X regresses joints from `v_shaped` including expression [ref smplx.md:135-142],
so expression moves jaw/eye joints slightly. Exact streaming without a re-bake:
precompute `joint_stream_jac = J_reg @ exprdirs` (J,3,E; tiny) at bake; per frame,
`rest_joints + jac @ psi` -> placement delta -> per-frame `ModelValues` override of
placements only (inertias untouched). This needs BR FK to accept placements with a time
batch dim matching `q` (2.4). Alternative (cheaper, inexact): freeze joints at neutral
expression; mark as an option, default exact.

### 2.4 Exact public BetterRobot API BH needs (and nothing else)

| Need | Proposed public BR name | Replaces private use |
|---|---|---|
| Build tree from arrays | `br.build_tree(parents, kinds, axes, placements, names, frames=None) -> Model` | `io.builders.kinematic_tree`, `io.ir.IRFrame` [scan §3] |
| Joint kinds | `br.JointKind` enum (FREE_FLYER, SPHERICAL, REVOLUTE, PRISMATIC, FIXED, MIMIC) | `model_structure.JOINT_KIND_CODES` |
| Reduced coordinates | `Model.expand_q(q)` / `Model.reduce_v(v)` (mimic / param transform) | `data_model.reduced_coordinates.expand_configuration` |
| Values replacement | `Model.with_values(joint_placements=, body_inertias=, frame_placements=)` broadcasting leading dims, incl. per-frame placements | exists [code shared.py:151]; extend to per-call batch |
| FK with rotations as matrices | `br.forward_kinematics(model, q, rotation="matrix")` or `Data.joint_rotation_world` | quat->matrix round trip [code pipeline.py:44,176] |
| Dynamics | `br.rnea`, `br.crba`, `br.frame_jacobian`, `br.centroidal_momentum` | — |
| Mesh inertia | `br.spatial.Inertia.from_mesh_segments(verts, faces, face_segment, density)` batched | per-part loop [code mass.py:253] |
| Lie | `br.lie.so3`, `br.lie.se3` | already public |
| Optim | `Variable`, `Residual`, `Problem`, LM/GN, `TorchOptimizer`, robust loss, phase runner | BVR's `tools/optim.py` |
| Tasks | frame-pose, joint-limit, velocity-limit, self-collision residuals | — |
| Viewer | `Layer` protocol, `RenderContext`, timeline/playback, gizmo/picking hooks | studio's own server [code studio/app.py:60] |

Rule: BH imports only `better_robot` top level and documented subpackages; a contract test
in BH greps imports against this list.

### 2.5 Retargeting: lives in `better_human.retarget`

BR must not import BH; retargeting needs both; BH already depends on BR. So it lives in BH,
built only from BR public `tasks` residuals + `optim`. A third package is warranted only if
robot->robot retargeting or non-SMPL sources (BVH/C3D skeletons) become first-class; keep
the core source-agnostic (input = named world frames over time, not a Body) so it can move.
Design, merging GMR and soma-retargeter:
- `MatchTable`: human frame -> robot frame, position/rotation weights, local offsets, per
  phase (GMR's two hand-tuned tables become explicit phases) [ref GMR.md:152-160].
- Scale: per-segment from body heights (GMR human_scale) or fitted (shape-aware: the
  human `Identity` gives exact segment lengths).
- Phase 1 per-frame LM (positions), phase 2 positions+rotations, phase 3 whole-clip
  smoothing with joint/velocity limits, foot-contact no-skate, z>=0. Batch axis = clips x
  frames (soma-retargeter pattern) on GPU.
- Optional OmniRetarget-style interaction-mesh Laplacian term for objects/terrain [web].

### 2.6 Human terms (`terms/`, each a BR `Residual`)

`Keypoints3D(markers, target, conf)`, `Reprojection2D(markers, camera, kp2d, conf)`
(robust loss), `MarkerResidual` (C3D/virtual markers), `PosePriorGMM` / `PosePriorVAE`
(VPoser weights optional asset), `JointLimits` (per-family table; SKEL natural limits),
`ShapePrior` (L2 on betas), `GroundContact` (height + tangential velocity of contact
frames, gated by a contact probability), `Penetration` (contact patches vs ground plane),
`Smoothness` (acceleration in tangent space via `Model.difference`), `FootSkate`.
Markers/keypoints use subset skinning (2.8 c): never full-mesh LBS in a keypoint fit.

### 2.7 Inverse dynamics with contact (`dynamics.py`)

`M(q)a + h(q,v) = S^T tau + sum_c J_c(q)^T f_c`. The 6 floating-base rows give
`J_base^T f = (M a + h)_base`; solve f per frame as a small batched QP/LS with friction cone
(linearized pyramid) and a prior that splits load by contact-frame height/velocity
(GaitDynamics-style), then `tau` from the actuated rows. Report residual base wrench as a
plausibility metric. Differentiable via unrolled projected-GN or implicit diff of the KKT
(depends on BR optim's differentiable-layer design).

### 2.8 Viewer split

BH ships layers only: `SkinnedMesh` (exists [code viz/skinned_mesh.py]), `Markers`
(targets vs model, residual lines), `ContactForces` (arrows; may be a BR layer since robots
need it too), `Segments` (part colors). Studio keeps only editors that write BH assets:
segment editor, contact-frame editor, landmark picker, persistence. Posing gizmo and
picking move to BR viewer as generic tools. Target studio ~1.2k lines (from ~3.16k).

## 3. Performance plan

### 3.1 Where time goes (SMPL-X, from scan §2 + arithmetic)

- FK: launch-bound, flat in batch (SMPL 2.5 ms @B=1 vs 1.6 ms @B=256); warp FK 0.36-0.63 ms.
  This is the whole 1.1-2.5x gap to upstream smplx.
- Pose correctives: 486 x 31,425 fp32 = 61 MB read per call; bandwidth-bound.
- Dense LBS: 10,475 x 55 x 12 MACs = 6.9 MFLOP per body; at B=256 it reads weights once,
  writes 256 x 10,475 x 3 x 4 B = 32 MB of vertices. Cheap per body but output-bound.
- Backward: autograd saves (B,V,3) blends + (B,J,12) transforms; LBS backward needs a
  (B,V,J)-sized intermediate under naive einsum if not careful [est].

### 3.2 Techniques, expected gain

| # | Technique | Expected effect | Basis |
|---|---|---|---|
| a | Level-parallel FK (one op per tree depth, SMPL depth ~9 vs 24 joints) + CUDA graph / torch.compile | FK 1.6-2.5 ms -> 0.2-0.5 ms; matches warp | [est], warp numbers above |
| b | FK emits rotation matrices; drop quat round trip and rest-bind rebuild | ~0.1-0.3 ms at B=1 | [est], [code pipeline.py:44,139-142] |
| c | Subset skinning: skin only marker/landmark vertices (~100 of 10,475) | keypoint fits: LBS+correctives ~50-100x less work | arithmetic: 100/10,475 rows of posedirs and weights |
| d | Top-K gather LBS (K=4-8 influences/vertex) | ~7-14x fewer LBS FLOPs vs J=55; win mainly for MHR (J>100) | arithmetic; MHR/SOMA ship sparse weights [ref] |
| e | Sparse/local posedirs (zero rows per joint feature, STAR-style) | 3-10x fewer corrective bytes [est]; exact for STAR/SUPR | [ref STAR.md:9] |
| f | bf16 storage of posedirs/MLP weights, fp32 accumulate | 2x fewer bytes; error ~ 0.4% of corrective magnitude (~tens of um) [est]; must pass the 6e-7 m gate in fp32 mode only | [est] |
| g | Fused FK+corrective+LBS kernel with handwritten backward (custom autograd.Function) | inference 2-4x over eager [est]; SOMA reports >7,000 meshes/s @B=128 on A100 with warp | [web arXiv 2603.16858] |
| h | Chunked/checkpointed time axis for T=10k clips | memory O(chunk) instead of O(T) | — |

Targets [est]: SMPL-X vertices fwd @B=256 <= 2 ms (upstream 4.32); @B=1 <= 0.7 ms
(upstream 1.71); keypoint-only fwd+bwd 10x faster than upstream full mesh + gather.
Order: a, b, c first (largest, no numerics risk); d, e, f after the benchmark suite exists.
SOMA's 7,000 meshes/s (~18 ms per 128) is slower than upstream smplx on our unnamed GPU,
so "warp" alone is not the win; FK launch count and bytes moved are.

### 3.3 Memory at large batch x long trajectories

One SMPL-X vertex tensor at T=10,000 fp32 = 1.26 GB; with autograd (blend, corrective
output, posed) ~3-4x that [est]. Rules: default outputs = joints + markers; vertices on
request; `forward_chunked(T_chunk=512)` with activation checkpointing for whole-mesh
losses (contact/penetration); identity tensors broadcast (never expanded to T).

### 3.4 Benchmark suite (`benchmarks/`, JSONL output)

Rows: {bake (with/without mass), FK, vertices fwd, vertices fwd+bwd, markers-only fwd+bwd,
expression stream, RNEA, contact-ID, retarget step} x {SMPL, SMPL-X, MANO, MHR, SKEL}
x B in {1, 64, 1024, 8192} x T in {1, 1k, 10k} for trajectory rows x {cpu, cuda} x
{fp32, bf16}. Baselines: upstream smplx (all SMPL rows), MHR TorchScript / pymomentum
(MHR rows), SOMA warp (if already installed; no installs), anny (optional). Metrics:
median, p90, peak memory (`torch.cuda.max_memory_allocated`), max vertex error vs baseline.
Output: one JSON line per row
`{bench, family, device, gpu_name, dtype, B, T, pass, median_ms, p90_ms, peak_mb,
max_err_m, baseline, ratio, git_sha_bh, git_sha_br, torch}`; a regression check compares
against a committed baseline file with a tolerance (e.g. 15% slower = fail). Shares BR's
benchmark harness (warmup, CUDA sync, timing) rather than its own 812-line script.

## 4. User code under the target API

```python
import better_human as bh, better_robot as br, torch

# 4.1 load + vertices for a batch
body = bh.load("smplx", "assets/smplx_neutral.npz", hands=True, face=True)
ident = body.bake(betas=betas)                                  # betas (B, 10)
out = body.forward(ident, q, stream={"expression": psi},        # q (B, nq), psi (B, 10)
                   outputs=bh.Outputs.VERTICES | bh.Outputs.JOINTS)
out.vertices, out.joints                                        # (B, V, 3), (B, J, 3)

# 4.2 fit to 3D keypoints over a clip (generic optim: terms + phases)
kp = bh.markers.named(body, "openpose25")                       # MarkerSet
betas = br.optim.Variable(torch.zeros(1, 10))
q = br.optim.Variable(body.neutral_q(T), manifold=body.model)   # (T, nq)
def residuals(stage):
    ident = body.bake(betas=betas)                              # differentiable bake
    out = body.forward(ident, q, outputs=bh.Outputs.MARKERS, markers=kp)
    terms = [bh.terms.Keypoints3D(out.markers, target_TK3, conf_TK, loss=br.optim.Huber(0.02)),
             bh.terms.PosePriorGMM(body, q, weight=1e-3),
             bh.terms.JointLimits(body, q, weight=10.0),
             bh.terms.ShapePrior(betas, weight=1e-2)]
    if stage == "refine":
        terms += [bh.terms.Smoothness(body.model, q, weight=1.0),
                  bh.terms.GroundContact(out.data, body.contacts, p_contact_T, weight=5.0)]
    return terms
br.optim.run_phases(residuals, [
    br.optim.Phase("global", vars=[q.root], optimizer=br.optim.LM(), iters=20),
    br.optim.Phase("body", vars=[q, betas], optimizer=torch.optim.Adam, lr=1e-2, iters=300),
    br.optim.Phase("refine", vars=[q, betas], optimizer=br.optim.LM(), iters=30)])

# 4.3 inverse dynamics with contact forces
ident = body.bake(betas=betas, mass=bh.Mass(total_kg=72.0))     # inertias from segments
v, a = br.finite_diff_velocity(body.model, q, dt), ...          # or from a spline fit
res = bh.dynamics.inverse_dynamics(body, ident, q, v, a,
                                   contacts=body.contacts.select("feet"), mu=0.8)
res.tau, res.forces, res.residual_wrench                        # (T,nv), (T,C,3), (T,6)

# 4.4 retarget a human clip to a humanoid robot
robot = br.load("unitree_g1.urdf")
rt = bh.retarget.Retargeter(robot, bh.retarget.MatchTable.load("smplx_to_g1.json"),
                            scale=bh.retarget.scale_from_identity(body, ident, robot))
q_robot = rt.run(bh.retarget.frames_from_body(body, ident, q),  # (T, n_frames, 7)
                 phases=("pos", "pos_rot", "smooth"), batch_clips=True)   # (T, robot.nq)
```

Every name above is proposed, not existing. 4.2 assumes BR optim gets `run_phases`,
`Phase`, robust losses, and Variables that accept torch optimizers — that is the BR optim
study's call; BH only supplies residual objects.

## 5. Migration order and risks

Order (each step lands with parity gates green: 6e-7 m SMPL family, 1.4e-6 m MHR):
1. BR: publish the API of 2.4 (build_tree, JointKind, expand_q, FK matrix output).
   BH switches imports; add the import-contract test. Risk: low; unblocks everything.
2. BH core rewrite: FamilySpec/Body/Identity/BodyOutput; policies -> enums; port SMPL,
   SMPL-X, MHR; expression as stream group with exact joint Jacobian. Risk: MHR classic
   carries shape (scan §1) — `classic_to_q` must return (identity kwargs, q) for all families.
3. Benchmark suite (3.4) before any speed work; record baselines incl. CPU and backward.
4. Speed: level-parallel FK in BR (benefits robots too), subset skinning, matrix FK.
   Then top-K LBS, sparse posedirs, bf16 behind flags. Risk: compile/graph lanes and
   dynamic shapes; keep eager as reference.
5. Vectorized mass (single batched pass) + lazy inertias. Risk: closed-part caps topology
   currently pure-Python lists [code mass.py:119-153]; precompute caps offline into the npz.
6. Add SMPL-H, MANO, STAR, SUPR (data + loaders); SKEL after (needs non-spherical joints,
   param transform; licence check).
7. Terms + dynamics.py (port BVR's human_optim terms after the BR optim redesign lands).
8. Retargeting (GMR-parity test on one clip: per-frame frame-position error vs GMR output).
9. Viewer: BH layers on unified BR viewer; trim studio to editors.
Risks: BR optim redesign timing gates 7-8; per-frame placement override may conflict with
BR's assumption that ModelValues are per-subject; licences (STAR/SUPR/SKEL non-commercial,
assets cannot ship); exact-parity vs bf16/sparse speedups must stay opt-in.

## 6. Open questions for the owner

1. Expression/MHR per-frame scale: exact (per-frame placements via BR values override) or
   approximate (joints frozen at neutral expression)? Recommended: exact, default on.
2. Retargeting home: `better_human.retarget` (recommended) or a third package if
   robot->robot or BVH/C3D sources matter soon?
3. SKEL/BSM priority vs STAR/SUPR: SKEL matters for biomechanics/ID (hinge knees, 46 DoF),
   STAR/SUPR mainly for speed/quality of correctives. Recommended: SKEL first if ID is a
   near-term deliverable.
4. Is a canonical-topology bridge (SOMA-style) in scope, or are families kept separate?
5. Contact-ID solver: LS with friction-pyramid QP per frame (fast, non-smooth) or smooth
   penalty (differentiable, approximate)? Depends on whether ID sits inside fitting loss.
6. Default outputs: should `forward` return vertices by default (smplx habit) or joints +
   markers only (memory/speed)? Recommended: explicit `outputs`, no default vertices in fits.
7. Which GPU produced the stored benchmark numbers (scan open question); needed for baselines.
8. Which studio editors are still used (segment, contact-frame, landmark picker)?

Sources (web): https://skel.is.tue.mpg.de/ ; https://arxiv.org/abs/2509.26633v1 (OmniRetarget);
https://arxiv.org/pdf/2603.16858 (SOMA, A100 throughput); https://arxiv.org/abs/2509.06607v1 (BSM).
