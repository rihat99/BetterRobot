# M4 — Consumer Feature Packs & Full Migration: Agent Execution Instructions

> **Implementation log (2026-07-17):** T4.1–T4.4's in-repo surfaces are
> implemented and verified on `dev`; fixed-size padded point sets plus boolean
> masks are the accepted convention. T4.5 and T4.6 are deliberately deferred:
> collision needs owner/external evidence, and BHF/BVR access or modification is
> prohibited. CI remains manual-only. See `m4_results.md` for exact deviations.

> Read `plan/for_agents/README.md` first. It carries the standing rules
> (deletion ordering, honesty rules, kernel requirements, committed
> benchmark definitions, test commands). Standing rules **1** (deletions
> land with/after their replacement — grep BOTH consumer repos before
> deleting anything), **4** (performance acceptance uses committed
> benchmark definitions, never bare ratios), **5** (no new abstraction
> without a second concrete caller in-tree), and **8** (evidence-gated
> decisions stay with the owner) govern this milestone directly.

> **Update (owner decision 2026-07-17 — branch strategy):** the redesign
> lives on a dedicated branch; the consumers switch branches at this
> milestone. Legacy surfaces may already be gone from the redesign branch
> (deleted in M1/M2 without shims), so T4.6's gating changes from "delete
> shims only after both repos migrate" to "execute the migration table as
> the consumers switch branches". The symbol-by-symbol table itself
> remains mandatory, and standing rule 1's revised text in the README is
> the authority over any "keep shims" wording below.

## Mission

Close the gap between what BetterRobot (BR) *is* and what its two real
consumers actually need, then finish the migration so the hand-rolled
duplicates can be deleted. Two projects —
`/data3/rikhat.akizhanov/better/BetterVideoReconstruction` (BVR, monocular
SMPL-X + contact-force reconstruction) and
`/data3/rikhat.akizhanov/better/BetterHumanForce` (BHF, human-force
tooling) — independently re-implemented vision residuals, an
inverse-contact-force solve, Lie/geometry utilities, viewer glue, and a
whole optimization engine because BR could not express their problems.
M2 (variable blocks, providers, batched solvers, phases) and M3
(parametric/batched ModelValues, order-preserving build) removed the
structural blockers. **M4 lands the concrete feature packs on top of that
substrate and executes the co-developed migration**: every feature ships
together with the consumer-side change that deletes the hand-rolled
version. The success metric is *deleted consumer lines*, per the roadmap's
tracked inventory. Roadmap done-when, per item, verbatim: **"the
corresponding consumer code is deleted from that repo and replaced by a BR
call."**

## Prerequisites

M4 sits at the top of the optimization stack; it consumes almost
everything below it. Before starting, confirm each of these landed (they
are the substrate the residuals and the migration are written against):

- **M2a** (`m2a_variable_blocks_and_slice.md`) — the block world is
  public: `VarSpec`/`Values`/`Problem`, residuals declare `reads`, the
  **provider DAG** with evaluation-local context, `ResidualItem` carrying
  a **per-item robust kernel**, the tangent-space autograd helper, and the
  **custom-residual author guide**. Every M4 residual is authored *through*
  that guide (T2a.9) — the vision residuals are the guide's second and
  third real users after the vertical slice. Check:
  `grep -rn "class VarSpec\|class Problem\|class ResidualItem" src/better_robot/optim/`
  hits, and the M2a slice test passes.
- **M2b** (`m2b_batched_second_order_solvers.md`) — batched LM/GN on
  `init_state/update/run`, per-block `scale`, feasible-retraction bounds.
  The contact-force task (T4.2) and the parity reproduction (T4.6) run on
  these solvers.
- **M2c** (`m2c_first_order_phases_tasks.md`) — matrix-free Adam, the
  **phase engine** (BVR's `run_phases` semantics upstreamed), `solve_ik`
  re-based as a preset, and the `costs/` shim marked for retirement with a
  migration note. T4.6 reproduces BVR stage 1 *as a phase configuration*,
  not a subclass.
- **M3** (`m3_parametric_model_breadth.md`) — batched/parametric
  ModelValues across FK/Jacobians/RNEA; **order-preserving build +
  public `q_permutation`** (this is what deletes the ~180-line q-remap
  shims — M4 only removes the *call sites*, M3 owns the mechanism);
  vectorized `integrate`/`difference`; `Inertia.from_mesh` (deletes the
  three hand-rolled inertia loops); **marker/site frames as rows in the
  frame table** (so SMPL landmark verts ride the residual stack — the
  projection/chamfer residuals depend on this). Check:
  `grep -rn "def q_permutation\|def from_mesh\|preserve_joint_order" src/better_robot/`
  hits.
- Consumer repos readable and **read-only during residual development**
  (BVR, BHF above). They are the requirements spec and the parity oracle.
  The migration commits (T4.6) *do* modify them — but only in lockstep
  with the BR feature that replaces the deleted code, and each pair of
  commits (BR feature + consumer deletion) must leave both repos importing
  cleanly.

If M2/M3 have not landed, **stop** — every task here dead-ends on a
missing block-world or a per-object `Frame`.

## Sizing & parallelism

Roadmap label: **M each**, co-developed with the consumers.

```
Independent, parallelizable once M2/M3 are in (hand to parallel agents):
  T4.1  Vision residual pack
  T4.2  Inverse contact-force task
  T4.3  Small utilities
  T4.4  Viewer playback API + stub decisions
  T4.5  Collision: port SelfCollisionResidual OR cut the package

Ordered, depends on ALL of T4.1–T4.5:
  T4.6  Full consumer parity + symbol-by-symbol migration table + shim deletion
```

Each of T4.1–T4.5 is co-developed with its consumer-side deletion (that
pairing is what "done" means). T4.6 is the integration gate: it reproduces
whole pipelines and retires the remaining shims. Do not start T4.6's
shim-deletion commits until T4.1–T4.5's replacements exist *and* both
repos are migrated onto them (standing rule 1).

## Tasks

### T4.1 — Vision residual pack  [M]

**Goal / done-when:** BR ships `ProjectionResidual` (pinhole reprojection
with per-point confidence weights), a masked chamfer residual, and the
point-SDF trio (penetration / attraction / clearance) sharing **one**
provider nearest-neighbour pass — all authored through the M2a
custom-residual guide, all wired as `ResidualItem`s with **per-item robust
kernels** (Geman-McClure added). The consumer-side done-when: BVR's
`tools/human_optim/losses.py` (564 lines) reprojection + chamfer + scene-SDF
terms are deleted and replaced by BR residual constructions in T4.6.

**Current state (verified 2026-07-17):**

- BR has **no** camera/projection, point-cloud, or SDF residual, and no
  per-point weight vector on any residual. Robust kernels exist
  (`optim/kernels/{l2,huber,cauchy,tukey}.py`, protocol
  `optim/kernels/base.py`: `weight(squared_norm)` + `rho`) but enter
  **only** solver-globally through IRLS (`optim/kernels/*` consumed at the
  solver, not per residual). Geman-McClure does **not** exist.
- The reference implementations to port are all in
  **BVR `tools/human_optim/losses.py`** (cite lines):
  - `geman_mcclure(r2, scale2)` — `losses.py:26`. The kernel to add.
  - `sapiens_reprojection_gm(...)` — `losses.py:39-71`: pinhole reprojection
    of body joints + landmark verts to Sapiens Goliath-308 keypoints,
    per-point detection **score** as weight, gated at `score_thresh`,
    uniform GM scale. This is the `ProjectionResidual` spec.
  - `init_reprojection_gm(...)` — `losses.py:74-106`: identity-correspondence
    2D anchor (each joint to its own initial projection), `frame_mask` over
    valid frames. Same residual, different targets/weights.
  - `pad_frames(...)` — `losses.py:109-132`: ragged per-frame clouds →
    dense `(T, M, C)` + `(T, M)` validity mask. The padded-batch convention
    the chamfer/SDF residuals need (open question, see Pitfalls).
  - `chamfer_loss(...)` — `losses.py:135-187`: bidirectional masked chamfer
    between visible vertices and padded depth points, optional per-vertex
    `vertex_weights`, chunked `cdist`.
  - `scene_signed_distance(...)` — `losses.py:190-244`: **one detached
    nearest-neighbour pass** ("One detached nearest-neighbour pass, shared
    by the three scene losses"), feeding
    `penetration_penalty` (`losses.py:253`) / `attraction_penalty`
    (`losses.py:366`) / `clearance_penalty` (`losses.py:324`) — the three
    heads differ *only* in the penalty on the shared signed distance `s`.
    This is exactly the M2a provider-DAG "one NN pass → ≥3 residuals"
    pattern (T2a.4), promoted from test-local slice code to library
    residuals.
  - Camera math in **BVR `tools/human_optim/camera.py`**:
    `matrix_to_se3` (`camera.py:26-31`), `transform_points`
    (`camera.py:45-53`), `project(points_cam, K, eps)` (`camera.py:55`).
- BHF's per-point-Cauchy pattern (`tools/object_align/sdf_fit.py:222-239`,
  `_data_loss` via `query_udf`) is the **second** concrete caller for the
  point-SDF + per-item-kernel design (satisfies standing rule 5).

**Implementation plan:**

1. **Add the Geman-McClure kernel.** New file
   `src/better_robot/optim/kernels/geman_mcclure.py` (follow `file-naming`
   + `python-standards`), mirroring the `Cauchy` shape (`__init__(*, c)`,
   `rho(squared_norm)`, `weight(squared_norm)`). Port the closed form from
   `BVR losses.py:26`. Register it wherever the kernel registry/exports
   live beside `cauchy.py`.
2. **Per-item robust kernels.** Confirm M2a's `ResidualItem` records a
   `kernel` (T2a.3 step 5) and M2b applies it per item in IRLS. If the
   built-in residuals do not yet accept a `kernel=` at the item level,
   this is where it lands — the kernel moves from solver-global to
   per-residual-item so a consumer can put GM on reprojection and plain L2
   on priors *simultaneously* (03 §6 change 3). The kernel returns the
   **raw** residual weighting; do not pre-multiply inside the residual (the
   ABI keeps raw rows + solver-side weighting — 03 §2.2).
3. **`ProjectionResidual`** in `src/better_robot/residuals/` (name via
   `file-naming`; suggest `projection.py`). Sketch **(direction, not final
   signature)**:

   ```python
   class ProjectionResidual(Residual):
       reads = ("q",)                       # frame origins / marker verts via RobotStateProvider
       def __init__(self, model, point_ids, K, extrinsics, target_px, *,
                    weights=None, valid_mask=None): ...
       # point_ids: frame ids OR marker/site frame-table rows (M3) to project
       # residual rows: (pred_px - target_px), dim = 2 * n_points (per point x,y)
       # weights: (..., n_points) per-point confidence, gated by the caller
       # kernel (e.g. GemanMcClure) lives on the ResidualItem, not here
   ```

   - `pred_px = project(transform_points(p_world, extrinsics), K)` reusing
     the camera math ported into a small utility (see T4.3's 4×4↔7-vec /
     projection note — decide with the owner whether `project`/`transform_points`
     live in a new `camera`-thin module or as residual-private helpers;
     **keep them geometric and camera-model-thin** — do NOT build a camera
     abstraction, audit §7 G5 risk).
   - Analytic Jacobian block (optional but high value — analytic was
     verified ~39× faster than FD): `d(pixels)/d(p_cam) · R_cam · J_frame`.
     If you ship only the autodiff block first, register it via
     `jacobian_blocks` using the M2a AD strategy (jacrev/jacfwd per dim) —
     never silent FD (M0 deleted the lying enum). A raising analytic block
     is an error, never a silent fallback (author guide, T2a.9 §3).
   - Marker points (SMPL landmark verts) are **frame-table rows** (M3), so
     they ride the frame Jacobian stack for free — this is the M3
     dependency. Do not add a bespoke vertex path.
4. **Masked chamfer residual** (suggest `residuals/chamfer.py`). Residual =
   per-vertex nearest-neighbour distance vector to the (padded) target
   cloud, with the `(T, M)` validity mask and optional `vertex_weights`.
   The NN/`cdist` argmin is **detached** (gradients flow through the
   distance to the *matched* point, not through the argmin) — declare the
   detached-output convention per the author guide (T2a.9 §4). Chunk the
   `cdist` as `losses.py:135-187` does. Decide and record the ragged
   convention (Pitfalls).
5. **Point-SDF trio sharing ONE provider.** A `SceneSDFProvider` (M2a
   `Provider`: `inputs`, `outputs`, `__call__(ctx)`) runs the detached
   nearest-neighbour / signed-distance pass **once** per evaluation and
   exposes `(signed_distance, dmin, confidence, has_point)` (mirroring
   `scene_signed_distance`, `losses.py:190-244`). Three residuals —
   `ScenePenetrationResidual`, `SceneAttractionResidual`,
   `SceneClearanceResidual` — each declare `reads=("scene_sdf",)` and apply
   their penalty head (`losses.py:253/324/366`). The provider runs at most
   once even with all three active (assert this with a counting test — it
   is the whole point of the DAG). BHF's `SpatialHashNN`
   (`tools/geometry/nn_search.py:48`) and BVR's `cdist` pass are the two
   NN backends; port/adopt one batched NN utility (audit §7 G5), keep it
   geometric.

**What to test** (`tests/residuals/test_projection.py`,
`test_chamfer.py`, `test_scene_sdf.py`; float32, invoke `write-tests`):

- Projection: residual value parity vs a hand-computed pinhole projection
  on a synthetic `(K, extrinsics, points)`; per-point weight zeroing below
  threshold; analytic-vs-autodiff Jacobian parity (atol ~1e-3); gradcheck
  through `q` at a non-degenerate pose. A point behind the camera
  (depth ≤ eps) is clamped, not NaN.
- Chamfer: value parity vs the padded `losses.py` form on toy clouds;
  validity mask excludes padding columns from the min; a frame with no
  valid point drops out; gradient flows to matched vertices only.
- Scene-SDF trio: **the provider runs exactly once** per `residual()` /
  `gradient()` call with all three residuals active
  (monkeypatch-count the NN pass); each head's sign/magnitude matches
  `penetration/attraction/clearance_penalty`; the three never
  double-count the shared distance.
- Geman-McClure kernel: `rho`/`weight` match `losses.py:26` on a range of
  squared norms incl. 0; bounded in `[0, 1)`.
- Per-item kernels: one `Problem` with GM on a projection item and L2 on a
  prior item produces the same weighting as applying each kernel
  independently (no global-IRLS bleed).

**Pitfalls / do-not-forget:**
- **Ragged per-frame data is an owner-open question** (audit §8.4): should
  BR standardize the padded `(tensor, validity_mask)` convention of
  `pad_frames` as *the* data-carrying-residual contract, or stay out of it?
  Prototype with the padded convention (it is what both the chamfer and SDF
  terms need), **write up the cost, and stop for owner review** before
  freezing it as a public convention (standing rule 8).
- Keep the vision pack **camera-model-thin**: no camera class, no
  intrinsics abstraction beyond `(K, extrinsics)` tensors (audit §8.3 —
  scope-creep-into-a-vision-library is the named risk). Deliberate
  non-goals that BR must NOT absorb (audit §7): visibility/front-facing
  masks, Sapiens/Goliath keypoint mappings, contact-label heuristics,
  silhouette rendering, RANSAC.
- The warp lane for these residuals is **M6** (`m6_...md` T6.5 does pose
  first; projection/point-cloud kernels follow the same
  gradients-in-forward ABI). M4 ships the **torch** residuals only — they
  are the oracle M6 tests against. Return raw `(B..., dim)` rows +
  `(B..., dim, nv)` Jacobian blocks; weighting stays solver-side.
- Update `residuals/CLAUDE.md` and the residual concept doc in the same
  change (docs drift is a verified systemic failure — assessment §2.6).

### T4.2 — Inverse contact-force task  [M]

**Goal / done-when:** a single BR task solves the inverse-contact-force
problem both repos hand-roll. Done-when: BVR `stages.py:757-901`
(`_world_to_local`/`_build_fext`/`_solve_person`) and BHF
`optimize_dynamics.py:102-320` are replaced by one BR call in T4.6.

**Current state (verified — the two are byte-similar copies):**

- **BVR** `tools/human_optim/stages.py`: `_world_to_local` (`:757`),
  `_build_fext` (`:764`) scatters masked world forces into
  `(T, njoints, 6)`, `_solve_person` (`:776`) central-diffs `v,a`, runs
  `br.rnea(model, data, q, v, a, fext=fext)` (`:833`, `:857`), minimizes
  `base_wrench = tau[..., :6]` + `force_magnitude` (`:834`) with
  `torch.optim.LBFGS` (`:844`).
- **BHF** `scripts/motion/optimize_dynamics.py`: `_build_fext` (`:102`) and
  `_precompute_world_to_local` (`:132`) — near-identical; `br.rnea(...,
  fext=...)` (`:244`, `:281`); objective `base_wrench` (`:250`) +
  force-magnitude + `force_smooth` + `torque_smooth`.
- **`rnea(fext=...)` autograd already works** and is used in anger by both
  (audit §6.3: autograd through `fext` into LBFGS). This is a hard
  requirement of the task API — verify it stays intact.

**Implementation plan:**

1. `src/better_robot/tasks/` — a `solve_contact_forces(...)` task (name via
   `file-naming`). Sketch **(direction, not final signature)**:

   ```python
   def solve_contact_forces(
       model, q_traj, contacts, *, gravity, weights, dt,
       optimizer_cfg=...,
   ) -> ContactForceResult:
       # contacts: per-frame active mask (T, C), contact frame/joint ids (C,),
       #           and the world→local rotation source (frozen q ⇒ static)
       # forces f_world (T, C, 3) are a VARIABLE BLOCK (M2a); the base-wrench
       #   ‖tau[:6]‖² is a residual/objective term; force-magnitude /
       #   force-smooth / torque-smooth are additional terms with weights
       # internally: central-diff v,a (M3 vectorized difference); scatter fext;
       #   br.rnea(fext=...); solve on the M2b batched solver / M2c phases
   ```

2. Express forces as an M2a **variable block** and the base-wrench as a
   residual (or scalar `ObjectiveTerm`, per the M2a T2a.5 decision) — this
   is the payoff of doing the task *after* the block world exists (audit §7
   G8: "API design should wait for G2/G3 so forces are just another var
   block"). The `fext` scatter becomes a provider that maps the force block
   → `(T, njoints, 6)` and runs RNEA once per evaluation.
3. Keep the world→local force rotation precompute (`so3.to_matrix` of the
   frozen `oMi[:, contact_ids, 3:7]`, detached) as-is — it is correct and
   q is frozen in this fit.
4. The task must accept per-term weights (`base_wrench`, `force_magnitude`,
   `force_smooth`, `torque_smooth`) matching both consumers' knobs.

**What to test** (`tests/tasks/test_contact_forces.py`):
- On a synthetic short clip (T ≤ 10, a free-flyer + a few joints), the
  recovered forces drive `‖tau[..., :6]‖` toward zero (base wrench
  balanced) within tolerance; the loss decreases monotonically on a fixed
  seed.
- Autograd flows through `fext` into the force block (gradcheck of the
  base-wrench term w.r.t. `f_world`, fp32 loose tolerance) — the explicit
  regression guard on the `rnea(fext=...)` differentiability the consumers
  rely on.
- Force-smooth / torque-smooth terms penalize the intended differences
  (unit test each term's contribution against a hand-computed value).
- Batched over persons/clips (leading batch axis) matches sequential
  per-element solves within stated tolerances (standing rule 3).

**Pitfalls:**
- BR spatial convention is `[lin, ang]`; `fext` rows are `[force(3),
  torque(3)]` with torque zeroed by both consumers — preserve that packing.
- Gravity: BVR sets it per-clip via `dataclasses.replace(model, gravity=g6)`
  (`stages.py:795`, audit §4.9). M3/M4 should let gravity be a task
  argument or a builder kwarg — do not force `dataclasses.replace` on
  users. If M3 did not add a gravity kwarg, thread it through the task and
  file a one-line note for the builder-ergonomics follow-up.
- The dynamics warp kernel for RNEA is **M6** (`m6_...md` T6.7, dynamics
  LAST) — this task runs the torch-lane RNEA; it must not assume a kernel.

### T4.3 — Small utilities with real demand ONLY  [S each]

**Goal / done-when:** exactly the four utilities the consumers demand
*today* land in BR, each with its citing call site; **no speculative
additions** (standing rule 5 — every new symbol needs a second concrete
caller, which here is the consumer call site plus the parity test).
Done-when: each utility's consumer call site is deleted and replaced by the
BR call in T4.6.

**The four, each with its verified demand:**

1. **euler ↔ quaternion** (`so3.from_euler` / `so3.to_euler`). Demand:
   **BHF `tools/robot_motion/motion.py:8,61`** imports `pypose` solely for
   `pp.euler2SO3(euler)` — pypose is a runtime dependency of BHF *for this
   one call* (audit §3.12). Landing this deletes BHF's last pypose import.
   Add to `lie/so3.py`; state the Euler convention (order + intrinsic/
   extrinsic) explicitly in the docstring and test it against pypose's
   output on the exact `[-π/2, 0, 0]` case BHF uses.
2. **4×4 ↔ 7-vector interop** (`se3.from_matrix` / `se3.to_matrix`).
   Demand: hand-rolled **twice** — BVR `camera.py:26-31` (`matrix_to_se3`)
   and BHF `transforms.py:90-120` (`so3_exp`/`se3_exp` building 4×4 from
   BR's 7-vector via `so3.to_matrix`). `so3.from_matrix`/`to_matrix` exist;
   add the SE3 homogeneous-matrix pair. Scalar-last quaternion, `[tx,ty,tz]`
   translation — the frozen convention.
3. **Weighted batched Umeyama** (`umeyama` / similarity-transform fit).
   Demand: **BHF `tools/geometry/transforms.py:125-275`** — `umeyama_7dof`
   (`:125`), `umeyama_7dof_weighted` (`:182`), `umeyama_6dof_weighted`
   (`:240`), "ready to upstream" (audit §7 G9). Port the SVD-of-cross-
   covariance form with the `det(R)=+1` reflection fix and per-point
   weights, batched over leading dims. Decide placement (a `lie/`-adjacent
   geometry helper or `spatial/`); keep it plain-tensors-in/out.
4. **Trajectory SLERP smoothing** (`smooth_trajectory` / `kernel_smooth`).
   Demand: **BHF `tools/robot_motion/smoothing.py`** — `kernel_smooth`
   (`:154`), `_iterative_mean` (`:108`) via `so3.slerp`/`se3.sclerp`,
   Gaussian/box kernels (`:49,:60`). Upstream it and **de-loop the batch
   dim** (`smoothing.py:193` has a per-batch Python loop). Operate on the
   `Trajectory` container. `so3.slerp`/`se3.sclerp` already exist and are
   the primitives.

**What to test** (`tests/lie/`, `tests/tasks/` as appropriate):
- euler↔quat: roundtrip `to_euler(from_euler(e)) ≈ e` away from gimbal
  lock; parity with pypose on BHF's fixture; gradcheck (needs M0 θ=0 fix if
  it routes through exp — it should route through direct quaternion
  construction, no singularity).
- 4×4↔7-vec: roundtrip both directions; batch shapes `()`, `(T,)`,
  `(B, T)`; orthonormal `R`, unit quaternion out.
- Umeyama: recovers a known `(s, R, t)` from transformed points exactly
  (noise-free); reflection case (`det < 0`) handled; weighted version
  down-weights an outlier; batched over leading dims.
- SLERP smoothing: a constant-rotation trajectory is unchanged; a noisy
  trajectory's smoothed variance drops; identical value/gradient to the
  BHF `_iterative_mean` reference on a fixed window; batched form matches
  the looped form.

**Pitfalls:**
- **Do not add anything else.** The audit lists tempting nearby utilities
  (mesh→inertia — that is M3's `Inertia.from_mesh`; pinhole projection —
  that rides T4.1; contact detection — a deliberate non-goal, audit §7).
  Adding a fifth "while I'm here" utility violates standing rule 5.
- euler conventions are a minefield — pin the exact order and frame in the
  docstring and assert it; a silently-different convention passes roundtrip
  tests and breaks the consumer.
- Quaternion double-cover: `slerp` and any mean must respect the M1
  engineering-contract hemisphere-alignment policy (03 §9) — forward-filled
  identical frames produce bit-identical adjacent quaternions in both
  consumers (audit §5.2); the smoother must not NaN there.

### T4.4 — Viewer: public playback API; implement-or-delete the stubs  [M]

**Goal / done-when:** a **public** per-frame playback/update API exists so
BHF `tools/robot_motion/playback.py` stops reaching into `viewer._scene`
and `viewer._backend`; and every viewer stub is **either implemented
(because a consumer uses it today) or deleted (because none does)** — no
shipped stubs (03 §6, roadmap M4). The two surfaces BHF imports **must keep
working**: `viewer.Visualizer` and `viewer.overlays.ForceVectorsOverlay`
(verified live imports, `playback.py:94-95`).

**Decision procedure (apply per surface — standing rule 8 where genuinely
ambiguous, otherwise mechanical):**

> **Implement** the surface iff a consumer imports/calls it today (grep
> both repos). **Delete** it otherwise — including its `__init__` export,
> its `docs/` claim, and its test. A stub that neither consumer uses is a
> lie (assessment §1.6 rule: "if a symbol is not on the roadmap page it is
> implemented and tested"); it must not ship.

**Current state (verified 2026-07-17):**

- **Live consumer use — KEEP + make public:** BHF `playback.py` calls
  `Visualizer(model)` (`:101`), `viewer.add_trajectory` (`:102`),
  `viewer.show` (`:263`), constructs `ForceVectorsOverlay` (`:168`), and
  **reaches into `viewer._scene` / `viewer._backend`** (`:103-105`) to grab
  joint-sphere handles and recolor/scale them per frame by torque
  (audit §4.8). `ForceVectorsOverlay` is **functional** (real
  `attach/update/update_frame/set_visible/detach` — not a stub) and
  `Visualizer.add_trajectory` is real (`visualizer.py:140`).
- **Stubs with NO consumer — DELETE:**
  - `VideoRecorder` (`viewer/recorder.py:35` — every method raises) and
    `Visualizer.record` (`visualizer.py:211-212`, `_FUTURE_MSG_RECORD`);
    `render_trajectory` (`recorder.py:79`).
  - `ComOverlay` (`overlays/com.py:17` — all raise).
  - `PathTraceOverlay` (`overlays/path_trace.py:25` — all raise).
  - `ResidualPlotOverlay` (`overlays/residual_plot.py:17` — all raise).
  - `TrajectoryPlayer`'s transport stubs (`trajectory_player.py:95-121`:
    `seek/step/pause/set_speed/set_loop/set_ghost/set_trace/set_batch_index`
    all raise; `show_frame`/`play`/`seek_frame` work). No consumer imports
    `TrajectoryPlayer` at all (grep: BHF drives its own threaded loop in
    `playback.py`; `scripts/precompute/view_motion.py` uses `Trajectory`).
  - `OffscreenBackend` (`renderers/offscreen_backend.py` — all raise),
    `CollisionMode` (`render_modes/collision.py` — all raise), `Camera`
    path stubs (`camera.py:52+`), `interaction.py` stubs — grep both repos;
    if unused, delete.

**Implementation plan:**

1. **Public playback/update API.** Add a public method surface on
   `Visualizer` (or a small public `Playback` object) covering exactly what
   `playback.py:103-190` does through the privates: push frame `k` (already
   `TrajectoryPlayer.show_frame` / `Scene.update_from_q`), and a
   **public handle to per-body render primitives** so a caller can set
   per-body color and scale per frame without touching `_scene`/`_backend`.
   Design it from the `playback.py` usage, not speculatively — it needs:
   (a) per-frame `q` push, (b) per-joint-sphere color set, (c) per-joint
   scale set, (d) an overlay-data stream for `ForceVectorsOverlay`. Nothing
   more (audit §7 G10).
2. **Delete the dead stubs** listed above: remove the classes, their
   `overlays/__init__.py` / `viewer/__init__.py` exports, their `docs/`
   sections, and the `_FUTURE_MSG_*` constants that only they use. Retire
   any test asserting the stub raises. Update `docs/concepts/viewer.md §10`
   and the viewer `__init__` docstring (both currently advertise the
   deleted placeholders) in the same change (standing rule 2).
3. **Keep `ForceVectorsOverlay` and `add_trajectory` byte-compatible** —
   BHF constructs them by keyword; do not change their signatures without a
   paired BHF edit.
4. Migrate BHF `playback.py` onto the new public API in T4.6 (the `_scene`/
   `_backend` reach-in deletion is its co-developed consumer change).

**What to test** (`tests/viewer/`):
- The public playback API drives a headless `Scene` (viser lazily imported;
  the module imports without viser installed — existing invariant) through
  frames and sets per-body color/scale; a smoke test mirrors the
  `playback.py` recolor loop against the **public** surface only (no
  `_scene`/`_backend`).
- `import better_robot.viewer as v; v.Visualizer; from
  better_robot.viewer.overlays import ForceVectorsOverlay` still succeed
  (the frozen consumer surface).
- A grep-style contract test (or extend the existing one) asserting no
  `NotImplementedError`-only class remains exported from `viewer` — the
  "no shipped stubs" guarantee.

**Pitfalls:**
- `Trajectory` (`tasks.trajectory.Trajectory`) is a **live BHF data
  container** (`playback.py:26`, `smoothing.py`, `view_motion.py`) — it is
  not a viewer stub; do not touch it here (it is a migration-table row, not
  a deletion).
- Deleting `VideoRecorder` removes `render_trajectory` and the `recorder`
  module — check nothing in `examples/` or `docs/` executable snippets
  imports them (CI runs snippets per M1 item 10).

### T4.5 — Collision: port SelfCollisionResidual OR cut the package  [M]

**Goal / done-when:** either capsule self-collision ships as a working
`SelfCollisionResidual` **with a torch reference**, or `collision/` is cut
from the tree. **No third option** (roadmap M4; 03 §6). The torch reference
is required regardless of M6's warp-led plan — it is the oracle M6 (`m6_...md`
T6.6) tests its kernel against.

**Current state (verified — assessment §1.6, "every query path raises"):**

- **Constructs fine (portable schema):** geometry dataclasses
  `Sphere/Capsule/Box/HalfSpace/Plane` (`collision/geometry.py:19-53`); the
  pair registry `register_pair` (`pairs.py:19`).
- **Every query/optimization path raises `NotImplementedError`:**
  `point_to_segment` (`closest_pts.py:14`), `segment_to_segment`
  (`closest_pts.py:26`), `colldist_from_sdf` (`geometry.py:66`),
  `RobotCollision.from_model/world_capsules/self_distances/world_distances`
  (`robot_collision.py:48,55,62,73`), `distance` (`pairs.py:40`),
  `SelfCollisionResidual` (`residuals/collision.py:23`, raises at `:50,:57`),
  `WorldCollisionResidual` (`residuals/collision.py:72`, raises).
- **Consumer demand:** neither BVR nor BHF imports `collision/` today
  (audit §2.6). BUT the humanoid self-collision need is real for the
  human-motion work, and M6's boundary table lists capsule self-collision
  as a warp target that *feeds the M4 residual*. So this is an
  **evidence-gated decision, not an automatic cut.**

**Decision — produce the evidence and STOP for owner review (standing rule
8):** Determine whether a *current* consumer pipeline needs humanoid
self-collision (check BHF motion optimization and BVR human_optim for any
self-intersection avoidance; grep for "collision", "self", "penetration"
*among body parts* as opposed to scene penetration which is T4.1). Present
the owner with: (a) "port it" — if self-collision is on a near-term
consumer path (the human work plausibly wants it); (b) "cut it" — if no
consumer needs it within this milestone horizon. Do not silently default.

**Implementation plan (the PORT branch, if chosen):**

1. Implement `segment_to_segment` closest-distance (`closest_pts.py:26`) —
   the standard clamped-parametric segment-segment routine, batched, with a
   defined tie-break for parallel segments (M6's kernel must match this
   tie-break exactly — `m6_...md` T6.6 Pitfalls). Clamp the `sqrt` at 0 per
   an explicit epsilon policy.
2. `RobotCollision.from_model` + `world_capsules` (`robot_collision.py`):
   attach capsule geometry to bodies, transform endpoints by FK poses.
3. `SelfCollisionResidual` (`residuals/collision.py:23`): over the model's
   collision pairs, residual rows = margin-violation of pairwise capsule
   distances; analytic gradient w.r.t. endpoints chain-ruled to `q` via the
   frame Jacobians. Author it through the M2a residual guide (T2a.9);
   `reads=("q",)`, provider gives world capsule endpoints.
4. Update `docs/concepts/collision_and_geometry.md` and the roadmap stub
   inventory in the same change (standing rule 2).

**Implementation plan (the CUT branch, if chosen):**

1. Delete `src/better_robot/collision/` and `residuals/collision.py`
   (`SelfCollisionResidual`/`WorldCollisionResidual`), their exports, the
   `solve_ik` `collision_margin`/`collision_weight` config options that are
   never read (assessment §1.6), the collision `docs/` pages, and the
   `CollisionMode` viewer stub (T4.4 already flagged it).
2. Update the layer-DAG note (`collision ∥ kinematics`) and `CLAUDE.md`.
3. Notify the M6 executor: `m6_...md` T6.6 says "if collision was cut, this
   task is void and the boundary-table row is updated to say so" — leave a
   one-line pointer in the PR description.

**What to test (PORT branch)** (`tests/collision/`,
`tests/residuals/test_self_collision.py`):
- `segment_to_segment` parity vs a reference (e.g. a brute-force sampled
  minimum) on random segments incl. the parallel and touching cases;
  gradient direction correct at near-contact and deep penetration.
- `SelfCollisionResidual` on a humanoid model: zero residual when clear,
  positive violation when two capsules interpenetrate; gradcheck w.r.t. `q`;
  an IK-with-self-collision toy solve reduces penetration.
- `tests/collision/` no longer has any `NotImplementedError`-asserting test.

**What to test (CUT branch):** a contract test asserting `collision` is not
importable / not exported; `solve_ik` rejects or ignores the removed config
keys with a clear message; no `docs/` snippet references collision.

**Pitfalls:**
- Do not ship a half-port: if you implement `segment_to_segment` but leave
  `RobotCollision`/`SelfCollisionResidual` raising, that is the forbidden
  third option (a shipped stub). Port the *whole* self-collision path or
  cut the package.
- Parallel-segment closest points are non-unique — fix the tie-break now
  and document it; M6's kernel parity depends on it.

### T4.6 — Full consumer parity + symbol-by-symbol migration + shim deletion  [L] — the gate

**Goal / done-when (all three, in order):**
1. **Committed benchmark definitions written FIRST** (standing rule 4):
   hardware, dtype, shapes, warmup, iteration budget, statistic, success
   tolerance — checked into the repo *before* any parity measurement. These
   define "parity quality and runtime" for the two pipelines below.
2. **Whole-pipeline parity on BR primitives:** reproduce **BVR
   `human_optim` stage 1** (phased: projection + chamfer + SDF, plus scalar
   terms) and **BHF motion optimization** on BR primitives (blocks +
   providers + residuals + phases + solvers), at parity quality and runtime
   per those committed definitions.
3. **Execute the symbol-by-symbol migration table** and **delete the
   remaining shims** (`costs/`, the `Data` aliases, the legacy
   `GaussNewton.minimize` path, the standalone kernels) — **only after both
   repos are migrated** (standing rule 1).

**Build the migration table by grepping both repos for every
`better_robot` import → its M2/M3/M4 replacement.** The pre-verified
inventory below (grep evidence 2026-07-17) seeds it; expand/verify with
file paths before executing. Each row: *consumer symbol @ site → BR
replacement → landing milestone → delete-after*.

M2c's inherited, deliberately unverified seed ledger is
`plan/migration/bhf_legacy_surface.md`; re-grep the consumer before relying on
any row as deletion evidence.

**Migration table (verified import inventory — seed, expand in the PR):**

| Consumer symbol @ site | Replacement | Lands | Notes |
|---|---|---|---|
| BHF `costs.stack.CostStack` @ `scripts/motion/optimize_motion.py:210` | M2a `Problem` + `ResidualItem`s (phase engine M2c) | M2c/M4 | `costs/` shim retired **after** BHF motion migrates (T4.6) |
| BHF `residuals.{Acceleration,ContactConsistency,ReferenceTrajectory}Residual` + `residuals.base.ResidualState` @ `optimize_motion.py:211-216` | same residuals re-based on `reads`/blocks (M2c) | M2c | keep working through migration |
| BHF `optim.optimizers.gauss_newton.GaussNewton` + `.minimize` @ `tools/geometry/icp.py:58,330-332` | M2b batched GN on `init_state/update/run` (ICP workarounds → config, per M2c done-when) | M2b/M2c | **legacy `GaussNewton.minimize` path deleted after** ICP migrates |
| BHF `optim.kernels.{cauchy.Cauchy,huber.Huber}` @ `tools/object_align/sdf_fit.py:40-41` | per-item kernels (T4.1) — same `rho`/`weight` classes, now first-class residual-item option | M4 | standalone kernel import deleted after sdf_fit migrates |
| BHF `io.builders.{JOINT_NAMES,build_kinematic_tree_model,make_smpl_like_model}` + `smpl_like.PARENTS` @ `optimize_motion.py:119`, `precompute/*`, `robot_motion/{dynamics,motion}.py` | **kept** (audit §6.2: right-shaped API); M3 adds order-preserving build + `q_permutation` | M3 | **not a deletion** — extend; deletes the ~180-line remap shims (M3) |
| BHF `tasks.trajectory.Trajectory` @ `smoothing.py`, `motion.py`, `playback.py`, `view_motion.py` | **kept** container; SLERP smoothing upstreamed (T4.3) | M4 | Trajectory stays; `smoothing.py` deleted after T4.3 |
| BHF `viewer.Visualizer` + `viewer.overlays.ForceVectorsOverlay` @ `playback.py:94-95` | **kept**; new public playback API replaces `_scene`/`_backend` reach-in (T4.4) | M4 | signatures frozen; `playback.py` internals-access deleted |
| BHF `forward_kinematics`, `data_model.data.Data` @ `playback.py:92-93`, `motion.py` | kept public FK; `Data` construction via `model.create_data` | — | keep |
| BHF **deprecated `Data.oMi`** @ `motion.py:201`, `playback.py:115,180,190` | `data.joint_pose_world` (the canonical name) | M1/M4 | **`Data` alias shims deleted after** BHF migrates off `oMi` |
| BHF `lie.so3`/`se3`/`tangents.hat_so3` @ many sites | **kept** (adoption success — audit §6.1); numerics fixed in M0 | M0 | keep; euler/4×4 helpers added (T4.3) delete pypose + hand-rolled 4×4 |
| BVR `lie.se3`/`so3` @ `camera.py:18`, `motion.py:36`, `stages.py:759` | **kept**; `matrix_to_se3`/`project` → BR `se3.from_matrix`/projection helper (T4.1/T4.3) | M4 | camera.py hand-rolled interop deleted |
| BVR `data_model.model.Model` @ `smplx_robot/model.py:20`, `inertia.py:43` | **kept**; parametric/batched values (M3) | M3 | `smplx_robot/` remap shim deletable (M3) |
| BVR `io.builders.build_kinematic_tree_model` @ `model.py:71`, `inertia.py:142` | **kept**; order-preserving build (M3) | M3 | keep; extend |
| BVR `kinematics.forward_kinematics`, `import better_robot as br` (`br.rnea`, `br.forward_kinematics`) @ `stages.py:779-780` | **kept** public API | — | keep |
| BVR `tools/optim.py` (188-line engine), `tools/human_optim/losses.py` (564 lines) | M2a blocks + M2c phase engine + T4.1 vision residuals + T4.2 contact-force task | M2/M4 | the big deletions — land with T4.6 parity |

**Implementation plan:**

1. **Write the benchmark definitions first** (`tests/bench/definitions.md`
   or the M1-established location). Two pipeline cases: BVR-human_optim-
   stage-1-parity and BHF-motion-opt-parity. Each: model scale (SMPL-X-mid
   52-joint / SMPL-24), T, dtype fp32, fixed iteration budget, warmup,
   median+IQR timing with `torch.cuda.synchronize()` discipline (or CPU
   wall on the dev box — CUDA is broken here, standing rule 7), and a
   **quality tolerance** (final loss / RMSE within X of the consumer's own
   result on the same synthetic input). Get owner sign-off on the
   definitions before measuring (standing rule 8).
2. **Reproduce the two pipelines** as BR-primitive constructions (in
   `tests/` or `examples/`, synthetic data, no consumer imports):
   - BVR stage 1: a `Problem` with the T4.1 projection + chamfer + scene-SDF
     residuals and scalar smoothness/prior terms, driven by the M2c phase
     engine (root → full → refine → lbfgs phases, from BVR `stages.py`).
   - BHF motion: the M2c-rebased `Acceleration`/`ContactConsistency`/
     `ReferenceTrajectory` residuals on the block `Problem`, tangent
     parameterization via `model.integrate` (autograd now works — M0 θ=0
     fix), matching `optimize_motion.py:257-354`'s objective.
   Assert loss trajectory and final quality meet the committed tolerance,
   and runtime meets the committed budget. A regression is a **finding to
   report**, not silently accept (standing rule 4 / README working style).
3. **Co-developed migration commits.** For each replaced surface, land the
   BR feature (T4.1–T4.5) and the consumer-side deletion **as a paired
   change** that leaves both repos importing cleanly. Track deleted lines
   per the success-metric inventory: ~24 hand-rolled `torch.optim` call
   sites, the 188-line engine (`BVR tools/optim.py`), the 564-line loss
   library (`BVR losses.py`), the two fext loops, the four utilities'
   duplicates. (The two ~200-line q-remap shims and three inertia loops are
   **M3's** deletions — reference them, do not double-count.)
4. **Delete the remaining shims LAST**, only after both repos are off them:
   the `costs/` shim (M2c marked it; delete when BHF `optimize_motion.py`
   is migrated), the deprecated `Data` aliases (delete when BHF is off
   `oMi`), the legacy `GaussNewton.minimize` path (delete when BHF `icp.py`
   is migrated to `init_state/update/run`), and the standalone-kernel import
   surface if T4.1 supersedes it. Re-grep BOTH repos immediately before each
   deletion (standing rule 1) — the grep is the gate, not your memory.

**What to test:**
- The two parity reproductions are permanent tests/benchmarks that pass on
  the committed definitions (loss + runtime).
- After each migration commit: `uv run pytest tests/ -v` green in BR; both
  consumer repos still import cleanly (`python -c "import <module>"` per
  migrated file) — automate this as a checklist the PR runs.
- After the shim deletions: a grep proving zero remaining references to the
  deleted symbols in BOTH repos; the BR contract tests
  (`tests/contract/`) updated to reflect the removed surfaces without
  weakening the layer DAG.

**Pitfalls:**
- **Ordering is the whole game.** Deleting `costs/` or `Data.oMi` before
  BHF migrates breaks a live consumer (assessment §2.5 migration caveat).
  The grep-both-repos gate is mandatory before every deletion.
- Parity "quality" is per the committed tolerance, not exact equality —
  the block-world assembly and the batched solvers legitimately differ from
  the consumers' hand-rolled loops near thresholds (standing rule 3). State
  the tolerance; do not chase bit-parity.
- Do not tune the BR reproduction on the benchmark's own targets and report
  the tuned number as default (M6 T6.12 Pitfalls, same discipline) —
  report the default configuration.
- `costs/` shim retirement is *nominally* M2c (`m2c_...md`), but its actual
  deletion is gated on the BHF migration that lands here — coordinate so it
  is deleted exactly once, in the repo that migrates last.

## Milestone acceptance checklist

- [ ] Prereqs confirmed: M2a block world public (`VarSpec`/`Problem`/
      provider DAG/author guide), M2b solvers, M2c phase engine + `solve_ik`
      preset, M3 order-preserving build + `q_permutation` + `Inertia.from_mesh`
      + marker frames.
- [ ] **Vision pack:** `ProjectionResidual` (per-point weights, gated),
      masked chamfer (padded/validity), point-SDF trio sharing ONE provider
      pass (counting test proves single pass), Geman-McClure kernel added,
      per-item robust kernels working (GM + L2 simultaneously). Ragged-data
      convention decision made and owner-reviewed.
- [ ] **Contact-force task:** `solve_contact_forces` — forces as a var
      block, base-wrench + magnitude + smooth terms, `rnea(fext=...)`
      autograd verified, batched-parity tested.
- [ ] **Utilities (exactly four):** euler↔quat, 4×4↔7-vec, weighted batched
      Umeyama, trajectory SLERP smoothing — each with its citing consumer
      site and parity test; no speculative fifth.
- [ ] **Viewer:** public per-frame playback/update API (per-body color/
      scale, overlay stream) so BHF stops using `_scene`/`_backend`;
      `Visualizer` + `ForceVectorsOverlay` unchanged and working;
      `VideoRecorder`/`ComOverlay`/`PathTraceOverlay`/`ResidualPlotOverlay`/
      `TrajectoryPlayer` transport stubs / unused backends **deleted** with
      docs + exports; no `NotImplementedError`-only class exported.
- [ ] **Collision:** owner-reviewed decision recorded — either
      `SelfCollisionResidual` ships whole (segment-segment + RobotCollision
      + residual, torch reference for M6) OR `collision/` is cut whole
      (package + residuals + `solve_ik` options + docs). No half-port.
- [ ] **Parity benchmark definitions committed BEFORE measurement**;
      BVR-stage-1 and BHF-motion reproductions pass at committed quality +
      runtime tolerances.
- [ ] **Migration table executed**: every `better_robot` import in both
      repos mapped and migrated (or explicitly "kept"); deleted-consumer-
      line count tracked against the success-metric inventory.
- [ ] **Shims deleted LAST, grep-gated**: `costs/`, `Data` aliases, legacy
      `GaussNewton.minimize`, standalone-kernel import surface — each
      deleted only after both repos migrated; re-grep proves zero
      references.
- [ ] Full BR suite green (`uv run pytest tests/ -v`); pinocchio-parity
      suite untouched and green; both consumer repos import cleanly after
      every migration commit.
- [ ] Docs / per-package `CLAUDE.md` truth pass for every surface added or
      deleted this milestone (standing rule 2).

## Out of scope

- **Warp kernels for any of these residuals / collision / dynamics** →
  `m6_warp_fast_path_and_cuda_graphs.md` (T6.5 pose residual, T6.6
  collision consuming M4's torch reference, T6.7 dynamics). M4 ships the
  torch oracles only.
- **The block world, providers, batched solvers, phase engine, `solve_ik`
  rebase, `LeastSquaresProblem` death** → M2a/M2b/M2c. M4 *uses* them.
- **Parametric/batched ModelValues, mimic coordinate map, order-preserving
  build + `q_permutation`, `Inertia.from_mesh`, marker frame table** → M3.
  M4's deletions of the remap shims and inertia loops are *M3's* mechanism;
  M4 only removes call sites where its own features touch them.
- **A camera abstraction / vision library** — anti-goal (audit §8.3). Keep
  projection to `(K, extrinsics)` tensors + thin helpers.
- **Retargeting (`solve_retarget`)**, jerk/Yoshikawa/nullspace residuals —
  separate roadmap items; delete-or-implement per their own milestones (the
  Jerk/Yoshikawa/Nullspace stubs are M3/roadmap, not M4). Do not
  half-implement them here.
- **Contact detection, visibility masks, keypoint mappings, silhouette
  rendering, RANSAC** — deliberate consumer-owned non-goals (audit §7).

## References

- `plan/04_roadmap.md` — M4 section (the item list + per-item "the
  corresponding consumer code is deleted and replaced by a BR call") and
  the success-metric inventory (the deleted-line ledger).
- `plan/03_architecture.md` §6 (residual library: blocks, per-item robust
  kernels, the new residual families in demand order, collision port-or-cut
  — "No third option"); §9 (migration as a deliverable: symbol-by-symbol
  table, adapters, commit ordering; per-element failure semantics); §2.2
  (residual ABI: raw `r`, weighting torch-side); §5 (marker frames as table
  rows).
- `plan/01_assessment.md` §1.6 (collision: geometry/pairs construct, every
  query path raises); §2.5 (migration caveat: `costs.stack.CostStack` and
  `Data.oMi` are live BHF imports — deletions land with/after replacements).
- `plan/research/audit_consumer_gaps.md` — the PRIMARY evidence source:
  §3.2 (the 564-line loss library, term-by-term BR gap), §3.7 (the
  duplicated fext contact-force loops), §4.7 (no projection/point-cloud/
  per-term-kernel support), §4.8 (viewer `_scene`/`_backend` reach-in), §6
  (what to keep), §7 G5/G8/G9/G10 (the residual pack / contact task /
  utilities / viewer recommendations), §8 (owner-open questions: ragged
  data, where the camera lives), and the deliberate non-goals list.
- `plan/for_agents/m2a_variable_blocks_and_slice.md` — the provider DAG
  (T2a.4), the `ResidualItem`+per-item-kernel contract (T2a.3 step 5), the
  custom-residual author guide (T2a.9) every M4 residual is written
  through, the scalar-term decision (T2a.5) the vision scalar terms depend
  on, the tangent-space autograd helper (T2a.2).
- `plan/for_agents/m2b_...md` / `m2c_...md` — the solvers and phase engine
  T4.2/T4.6 run on; the `costs/` shim retirement note.
- `plan/for_agents/m3_parametric_model_breadth.md` — order-preserving build
  + `q_permutation` (deletes remap shims), `Inertia.from_mesh`, marker
  frames as table rows (projection/chamfer depend on it).
- `plan/for_agents/m6_warp_fast_path_and_cuda_graphs.md` — T6.5 (pose
  residual kernel ABI the vision residuals follow), T6.6 (collision kernel
  **requires M4's torch reference**; if collision is cut, that row is
  voided).
- Consumer evidence (read-only for development; modified only in T4.6's
  paired migration commits):
  - BVR `tools/human_optim/losses.py` (`:26` GM kernel, `:39-71`/`:74-106`
    reprojection, `:109-132` pad_frames, `:135-187` chamfer, `:190-244`
    shared NN pass, `:253/:324/:366` SDF heads), `camera.py:26-55`
    (projection + 4×4 interop), `stages.py:757-901` (fext contact-force),
    `tools/optim.py` (188-line engine → M2c phases).
  - BHF `scripts/motion/optimize_dynamics.py:102-320` (fext contact-force),
    `scripts/motion/optimize_motion.py:209-354` (CostStack motion opt →
    parity target), `tools/object_align/sdf_fit.py:40-41,222-239`
    (per-point Cauchy + UDF data term), `tools/geometry/transforms.py:90-275`
    (4×4 interop + Umeyama), `tools/robot_motion/smoothing.py:49-193` (SLERP
    smoothing), `tools/robot_motion/motion.py:8,61` (pypose euler2SO3),
    `tools/robot_motion/playback.py:92-190` (viewer `_scene`/`_backend`
    reach-in + `oMi`).
