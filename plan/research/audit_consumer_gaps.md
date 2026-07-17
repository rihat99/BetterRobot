# Consumer Gap Analysis: BetterVideoReconstruction + BetterHumanForce

Audit of how the two real consumer projects use — and fail to use — BetterRobot (BR).
Written 2026-07-16 by a read-only auditor agent. Every claim cites `file:line`; micro-checks
were run inside BR's venv (`uv run python`) and raw numbers are reported.

Path abbreviations used throughout:

- **BR** = `/data3/rikhat.akizhanov/better/BetterRobot`
- **BVR** = `/data3/rikhat.akizhanov/better/BetterVideoReconstruction`
- **BHF** = `/data3/rikhat.akizhanov/better/BetterHumanForce`

---

## 1. Scope & method

**What was read.** BVR: `tools/optim.py`, all of `tools/human_optim/` (losses, optimizer,
stages, qmath, motion, camera, prefit, contacts, depth_fit), `tools/smplx_robot/` (model,
inertia, dynamics, vertex_weights), `CLAUDE.md`, `pyproject.toml`. BHF: `tools/robot_motion/`
(motion, smoothing, dynamics, contact, playback, smpl_inertia), `tools/geometry/` (icp,
_icp_problem, transforms, nn_search), `tools/object_align/` (sdf_fit, polish, refine, pose),
`tools/smplx_alignment/` (losses, refine heads), `scripts/motion/{optimize_motion,optimize_dynamics}.py`,
`scripts/precompute/build_robot_motion.py`, `README.md`, `pyproject.toml`, tests. BR side:
`lie/_torch_native_backend.py`, `data_model/model.py` + `joint_models/{spherical,free_flyer}.py`,
`optim/{problem,state}.py`, `optim/optimizers/{gauss_newton,levenberg_marquardt,adam,lbfgs}.py`,
`optim/kernels/cauchy.py`, `costs/stack.py`, `residuals/base.py`, `io/builders/`, roadmap.

**How.** Exhaustive grep for `better_robot` imports in both consumer repos (all hits listed in
§2); grep for `torch.optim.` to census hand-rolled loops (§3); two runnable micro-checks inside
BR (autograd of `so3.exp`/`se3.exp` at zero tangent; `Model.difference` vs a vectorized twin —
§4.1, §4.4). Doc/comment claims by consumers were **verified against current BR code**, and two
were found stale (§5).

**Consumers in one paragraph each.**

- **BVR** reconstructs per-person SMPL-X trajectories + contact forces from monocular video.
  Its optimization core is a hand-rolled engine (`tools/optim.py`: `Problem`/`Phase`/`run_phases`)
  driving a 4-stage fit (root → kinematics_1 → kinematics_2 → dynamics) over `q (T, 211)` per
  person, with ~15 loss terms (reprojection, chamfer, scene-SDF penetration/attraction/clearance,
  priors, tangent-space smoothness, velocity dampers). BR is used only for: Lie primitives,
  `rnea` in the force-only dynamics stage, and `build_kinematic_tree_model` to make the SMPL-X
  body a BR model for that stage.
- **BHF** is the earlier human-force project. It is the *deepest* BR consumer
  (`scripts/motion/optimize_motion.py` uses `CostStack` + three BR residuals + `model.integrate`
  + `stack.gradient`; `tools/geometry/icp.py` uses BR's `GaussNewton`; `object_align/sdf_fit.py`
  uses BR's `Cauchy`/`Huber` kernels) — yet even there every actual iteration loop is
  `torch.optim.LBFGS`/`Adam` driven by the consumer, and each BR-solver adoption required
  workarounds documented in code comments (§4.5).

**Grand total of hand-rolled optimizer instantiations** (grep `torch.optim.`, excluding
`legacy/` and lr_schedulers): **24 call sites across 10 files** in the two projects
(BVR: optim.py×2, stages.py×1, mhr_to_smplx/converter.py×4; BHF: smplx_alignment/refine.py×4,
mhr_to_smplx×4, mhr_to_smpl×3, object_align/polish.py×2, optimize_motion.py×2, sdf_fit.py×1,
optimize_dynamics.py×1).

---

## 2. Inventory — every BR API the consumers import/call

### 2.1 `lie` (se3 / so3 / tangents) — the most-used layer by far

| Consumer file | APIs |
|---|---|
| BVR `tools/human_optim/motion.py:36,50-53` | `se3.log/compose/inverse`, `so3.log/normalize/compose/inverse` (batched tangent differences) |
| BVR `tools/human_optim/camera.py:18,29,41` | `so3.from_matrix`, `se3.compose` (camera→world folding) |
| BVR `tools/human_optim/stages.py:759` | `so3.to_matrix` (world→local force rotation) |
| BHF `tools/geometry/transforms.py:17-19,87-118` | `se3.exp`, `so3.exp/to_matrix`, `tangents.hat_so3` — wrapped into matrix-form adapters |
| BHF `tools/geometry/_icp_problem.py:29,36,46` | `so3.from_matrix/to_matrix` (Sim3 state packing) |
| BHF `tools/object_align/{sdf_fit.py:39, polish.py:39, refine.py:29}` | `so3.exp/compose/normalize/to_matrix/from_matrix` (delta-rotation pose params) |
| BHF `tools/robot_motion/smoothing.py:29` | `so3.slerp`, `se3.sclerp` (geodesic kernel smoothing) |
| BHF `tools/robot_motion/motion.py:12,64` | `se3.compose` (axis-convention fix) |
| BHF `scripts/motion/optimize_dynamics.py:134` | `so3.to_matrix` |
| BHF tests `tests/test_object_align/*.py`, `scripts/static/synthetic_bench.py:31` | `so3.*` for fixtures |

### 2.2 `io.builders` — the second pillar

| Consumer file | APIs |
|---|---|
| BVR `tools/smplx_robot/model.py:71`, `inertia.py:142` | `build_kinematic_tree_model` (52-joint SMPL-X-mid: free-flyer root + 51 spherical, per-body mass/com/inertia) |
| BHF `tools/robot_motion/dynamics.py:43,70` | `build_kinematic_tree_model`, `JOINT_NAMES` |
| BHF `scripts/motion/optimize_motion.py:119`, `scripts/precompute/{build_robot_motion.py:125,268, view_motion.py:52}` | `build_kinematic_tree_model`, `make_smpl_like_model`, `JOINT_NAMES` |
| BHF `tools/robot_motion/motion.py:11,88,...` | `smpl_like.PARENTS`, `JOINT_NAMES` (q/contact remap tables) |

### 2.3 Kinematics / dynamics / data_model

| Consumer file | APIs |
|---|---|
| BVR `tools/human_optim/stages.py:779-780,801,824,833,849,857` | `br.forward_kinematics`, `br.rnea(model, data, q, v, a, fext=…)`, `model.create_data`, `data.joint_pose_world`, `model.idx_vs/nvs/body_name_to_id`, `dataclasses.replace(model, gravity=g6)` (line 795) |
| BVR `tools/smplx_robot/dynamics.py:34` | `model.difference` (central-difference v, a) |
| BHF `scripts/motion/optimize_motion.py:288,291` | `model.integrate` (tangent parameterisation), `br.forward_kinematics(compute_frames=True)`, `model.frame_id` (line 143) |
| BHF `scripts/motion/optimize_dynamics.py:205,232,244` | `br.forward_kinematics`, `model.create_data`, `br.rnea` |
| BHF `tools/robot_motion/{motion.py:192-209, playback.py:92-95}` | `forward_kinematics`, `data.oMi`, `Data` |

### 2.4 `optim` — used only in fragments

| Consumer file | APIs |
|---|---|
| BHF `tools/geometry/icp.py:58,330-332` | `GaussNewton(tol, eps).minimize(problem, max_iter)` on a hand-built duck-typed problem |
| BHF `tools/object_align/sdf_fit.py:40-41,193-198,234` | `Cauchy(c)`, `Huber(delta)` — used as plain robust-loss functions (`kernel.rho(d²)`) inside a torch Adam loop, **not** inside a BR solver |
| BHF `scripts/motion/optimize_motion.py:210-216,257-272,345` | `CostStack`, `AccelerationResidual`, `ContactConsistencyResidual`, `ReferenceTrajectoryResidual`, `ResidualState`, `stack.residual/gradient` — but the loop itself is `torch.optim.LBFGS` with hand-wired `.grad` |

### 2.5 `tasks.Trajectory` and `viewer`

| Consumer file | APIs |
|---|---|
| BHF `tools/robot_motion/{smoothing.py:32,169,245, motion.py:17,193,254}`, `scripts/precompute/view_motion.py:53` | `tasks.trajectory.Trajectory` (as an npz-adjacent data container) |
| BHF `tools/robot_motion/playback.py:94-95,101-106` | `Visualizer`, `viewer.add_trajectory`, `ForceVectorsOverlay` — **plus private `viewer._scene` / `viewer._backend`** (line 103-105) |

### 2.6 What is *never* used

`solve_ik`, `solve_trajopt`, `LevenbergMarquardt` (as a consumer-facing solver), `Adam`/`LBFGS`
(BR's versions), `MultiStageOptimizer`, `LeastSquaresProblem` (the real class — BHF duck-types
it instead), any residual other than the three in §2.4, `aba`/`crba`/centroidal, `collision/`,
`SE3`/`Pose` typed wrappers, the `backends` layer, all Jacobian APIs
(`get_frame_jacobian` — one BVR docstring references a deleted `tools.human_optim.velocity`
module that used FK Jacobians, `losses.py:431`; the code today uses position differences instead).

---

## 3. Hand-rolled — what the consumers built themselves that overlaps BR's mission

### 3.1 A complete optimization engine (BVR `tools/optim.py`, 188 lines)

`Problem` (named leaf tensors + one shared forward returning a lazy `State` + named scalar loss
terms), `Phase` (per-phase weight column, iteration budget, Adam-vs-LBFGS, per-leaf gradient
masks, `on_start` hooks), `run_phases` (fresh optimizer per phase, MultiStep lr decay,
sync-free logging — "tensors are converted to floats only at log time, so no per-iteration GPU
sync sneaks in", `optim.py:80-82`). This engine drives the whole BVR kinematic pipeline
(`human_optim/optimizer.py:398-593`) and the scale pre-fit (`human_optim/prefit.py`).

**Why BR couldn't serve it:**
1. BR's variable model is a single flat tensor (`LeastSquaresProblem.x0`, `optim/problem.py:28`;
   `ResidualState.variables: torch.Tensor`, `residuals/base.py:43`) — BVR needs multiple named
   leaves and per-leaf gradient masks.
2. BR is strictly least-squares (residual *vectors*); BVR's terms are arbitrary scalar losses
   (Geman-McClure means, masked chamfer averages, hinge penalties) that have no natural residual
   vector form of static `dim` (the `Residual` Protocol requires a static `dim` attribute,
   `residuals/base.py:56`).
3. BR's first-order optimizers materialize the **full dense Jacobian each iteration**
   (`optim/optimizers/adam.py:79`, `lbfgs.py:81,148`) — at BVR's problem size (`q (T,211)`,
   T = hundreds; residual dim in the tens of thousands) this is unusable (§4.2).
4. No lazy shared-intermediate mechanism: BVR's scene terms share one detached NN pass
   (`losses.py:190-244` via the lazy `State`, `optimizer.py:412-424`); BR's CostStack calls each
   residual independently (`costs/stack.py:104-116`).

### 3.2 The full loss library (BVR `human_optim/losses.py`, 564 lines + `prefit.py`)

Per term, with the BR gap it exposes:

| Term | Location | BR gap |
|---|---|---|
| Geman-McClure keypoint reprojection (Sapiens + init-anchor) | `losses.py:39-106` | No camera model, no projection residual, no GM kernel, no per-point score weighting in BR |
| Bidirectional masked chamfer vs padded depth clouds | `losses.py:135-187` | No point-cloud residuals; no ragged/padded per-frame data support |
| Point-cloud SDF + 3 penalty heads (penetration / attraction / clearance) sharing one NN pass | `losses.py:190-407` | No scene representation, no NN search, no SDF residual (BR `residuals/collision.py` is a stub per roadmap) |
| Contact-velocity damper, radial (viewing-ray) velocity damper | `losses.py:410-452,492-520` | `ContactConsistencyResidual` exists but is frame-position-based and can't express label gating/confidence weighting/detached rays |
| Pose/orientation quaternion priors | `losses.py:455-489` | BR `RestResidual` is q-space; no quaternion-dot prior |
| Tangent-space accel/jerk/snap smoothness | `losses.py:523-564` | BR has Velocity/Acceleration residuals but Jerk is a stub (roadmap); and see §3.4 for why they don't use BR's differences |
| One-sided robust chamfer, converter-velocity tie | `prefit.py:70-120` | same as above |

### 3.3 Manifold handling *around* BR's singularity (BVR `human_optim/qmath.py`, 30 lines)

The module docstring is explicit: *"The optimizer treats `q (T, nq)` as a free leaf tensor
(**no exp retraction — that has a zero-tangent autograd singularity in BetterRobot's Lie
backend**). The quaternion blocks therefore drift off the unit sphere between steps and must be
re-normalized inside the forward pass"* (`qmath.py:3-8`). Verified true against current BR —
see §4.1.

### 3.4 A vectorized re-implementation of `Model.difference` (BVR `human_optim/motion.py`, 54 lines)

`tangent_difference` computes exactly `Model.difference` (root `se3.log(q0⁻¹∘q1)`, per-joint
`so3.log(q0⁻¹∘q1)`) but batched over frames *and* joints. The docstring: *"identical value and
gradient, but batched over frames and joints in a handful of ops instead of a per-joint Python
loop — several-fold faster (≈4–10× on CPU here), and needing no BetterRobot model build/remap
just to difference"* (`motion.py:19-23`). Verified: BR's `Model.difference`/`integrate` loop
over joints in Python with per-joint `cat` (`data_model/model.py:160-190`); micro-benchmark §4.4.
Consequence: BVR's `make_smplx_robot_model` (`smplx_robot/model.py:59-84`), built *"purely so
the kinematic stage can measure acceleration / jerk smoothness with BetterRobot's exact Lie
`model.difference`"* (`model.py:5-7`), is now **dead code** — its only remaining reference is a
docstring in `inertia.py:133`.

### 3.5 q-layout remapping at every boundary (both projects, ~180 duplicated lines)

BR reorders bodies depth-first at build time, so both consumers wrote name-keyed permutation
pairs: BVR `smplx_robot/model.py:87-128` (`remap_q_bh_to_model` / inverse, 52 joints), BHF
`robot_motion/motion.py:71-181` (`remap_smpl_q_to_model` / inverse + two contact-column
remappers, 24 joints). BHF's docstring notes the failure mode: *"Feeding SMPL-ordered q straight
into `forward_kinematics` therefore produces scrambled FK"* (`motion.py:82-84`). BR offers no
"build in caller's joint order" option and no layout-mapping utility.

### 3.6 Robot-from-SMPL construction + inertia (both projects)

- BVR `smplx_robot/inertia.py:53-180`: per-part mass/COM/inertia from convex hulls of
  LBS-argmax vertex segments, renormalized to a target total mass, scale folded in (mass ∝ s³
  implicitly), then `build_kinematic_tree_model(mass_per_body=…, com_per_body=…, inertia_per_body=…)`.
- BHF `robot_motion/smpl_inertia.py:76-149`: same idea for SMPL-24 with shipped face
  segmentation + anthropometric density tables (Dempster/Chandler/Clauser).

BR's builder accepts the tensors (good — §6) but provides no mesh→inertia utilities; both
projects hand-roll trimesh pipelines. This belongs mostly in `better_human`'s future scope, but
BR could own the generic "watertight mesh → (mass, com, inertia)" helper.

### 3.7 Contact-force estimation loops (both projects, near-duplicates)

BVR `human_optim/stages.py:757-901` (`_world_to_local`, `_build_fext`, `_solve_person`) and BHF
`scripts/motion/optimize_dynamics.py:102-300`: identical pattern — freeze `q(t)`, central-diff
`v,a` via `model.difference` (BVR `smplx_robot/dynamics.py:12-41`, BHF
`robot_motion/dynamics.py:133-181` — byte-similar copies), scatter masked world forces into a
`(T, njoints, 6)` `fext`, run `br.rnea`, minimize base-wrench + regularizers with
`torch.optim.LBFGS`. BR has `rnea(fext=…)` but no "base-wrench residual", no contact-force
variable concept, no task-level solver for this extremely common inverse-dynamics fit.

### 3.8 ICP / Sim3 fitting (BHF `geometry/icp.py` 362 + `_icp_problem.py` 169 lines)

RANSAC + outer re-association loop + inner BR `GaussNewton` on hand-built
`Sim3PointToPlane/PointProblem` classes that duck-type `LeastSquaresProblem` **including the
private `_nv` attribute** (`_icp_problem.py:99,142`). Three workarounds are documented in the
code — trust-region clipping, relative damping, external convergence tests — see §4.5. Plus a
custom `SpatialHashNN` (`geometry/nn_search.py`, 283 lines) and batched/weighted Umeyama
(`geometry/transforms.py:125-275`).

### 3.9 Robust-pose fitters (BHF `object_align/sdf_fit.py` 633 + `polish.py` 741 lines)

Hand-rolled Adam loops with: per-parameter lr groups (`sdf_fit.py:398-412`), robust-kernel
annealing schedules (`sdf_fit.py:308-355`), best-state snapshot/restore with NaN guards
(`sdf_fit.py:498-567`, `polish.py:480-560`), post-step parameter clamping
(`sdf_fit.py:541-553`), delta-rotation parameterisation `q = so3.compose(so3.exp(δω), q_base)`
(`sdf_fit.py:425` — note δω initialized to `1e-6`, not 0, `sdf_fit.py:358-360`, consistent with
the exp-at-zero NaN in §4.1). BR kernels are used as plain functions; everything else is manual.

### 3.10 SLERP kernel smoothing (BHF `robot_motion/smoothing.py`, 270 lines)

Weighted geodesic means via iterative `so3.slerp`/`se3.sclerp` over gathered windows —
a natural "trajectory filter" utility BR lacks (BR primitives made it possible; note the
per-batch Python loop at `smoothing.py:193`).

### 3.11 A second, BR-free refinement stack (BHF `tools/smplx_alignment/`, ~4.1k lines)

Four-stage Adam/L-BFGS refiner over **named SMPL-X parameter blocks**
(`transl/global_orient/body_pose/betas/scale` as separate leaves, per-stage free-var sets,
`refine.py:1-33`) with seven losses (projection, image-aligned scene penetration, chamfer,
floor half-space, contact pull, regularization — `losses.py:1-29`) plus a quasi-static physics
term (`physics.py`). Uses **sklearn KDTree** for NN (`losses.py:36`). Zero BR imports — it
predates BR adoption but is still the active static-image path; it independently converged on
the same multi-block/staged/robust pattern as BVR's engine.

### 3.12 Miscellaneous small overlaps

- Pinhole projection + extrinsic transforms: BVR `camera.py:45-67`, BHF
  `smplx_alignment/losses.py:43-47`.
- 4×4 ↔ 7-vector conversion helpers: BVR `camera.py:26-30`, BHF `transforms.py:102-118`.
- Euler→quaternion via **pypose** (`pp.euler2SO3`) because BR has no euler conversion:
  BHF `robot_motion/motion.py:8,61` (pypose still a runtime dep of BHF for this one call;
  BVR also lists pypose in `pyproject.toml:13` though no non-legacy import remains).
- Contact detection/readout: BVR `human_optim/contacts.py` (SDF → guarded per-joint min →
  drift-gated hysteresis, 497 lines), `depth_fit.py`; BHF `robot_motion/contact.py`
  (LBS-argmax vertex→joint aggregation). BR has nothing in this space (collision residuals are
  roadmap stubs).

---

## 4. Confirmed problems (evidence · why it matters · severity)

### 4.1 `so3.exp` / `se3.exp` autograd is NaN at zero tangent — kills tangent-space optimization

**Evidence.** Micro-check run in BR's venv:

```
so3.exp grad at 0:                tensor([nan, nan, nan])
se3.exp retraction grad at 0:     tensor([1., 1., 1., nan, nan, nan])
log-diff grad at identical rotations (via exp leaf): tensor([nan, nan, nan])
```

Root cause: `so3_exp` computes `theta = theta2.sqrt()` and `qw = cos(theta/2)`
(`lie/_torch_native_backend.py:147-159`); `d(sqrt)/d(theta2)` is `inf` at 0, and the
`0 × inf` in the chain yields NaN — the Taylor stitch covers `qxyz` but **not `qw`**.
`lie/CLAUDE.md` claims "fp64 gradcheck covers … so3_{exp,log}" and that ops "stay smooth and
differentiable across the singularity" — the claim is false at exactly `θ = 0`, which is the
**most common point in practice**: every delta-from-init parameterisation starts there.
Propagates into `SphericalJoint.integrate` (`joint_models/spherical.py:52-56`) and
`FreeFlyer.integrate` (`free_flyer.py:47-51`), hence `Model.integrate` (`model.py:160-174`).

**Consumer impact (this is not theoretical).**
- BVR abandoned exp retraction entirely and re-normalizes quaternions by division inside the
  forward (`qmath.py:3-8` names BR's singularity as the reason).
- BHF `optimize_motion.py` parameterises as `model.integrate(q_init, delta_v)` with
  `delta_v = 0` at start (`optimize_motion.py:285-288`) — autograd through that is NaN at
  iteration 0, so they bypass autograd and hand-wire `stack.gradient` into
  `torch.optim.LBFGS` (`optimize_motion.py:319-347`).
- BHF `sdf_fit.py` initialises its rotation delta to `1e-6` instead of 0
  (`sdf_fit.py:358-360`) — a tell-tale epsilon hack.

**Severity: Critical.** The single biggest blocker to "BR owns the consumers' loops": the
canonical manifold-optimization pattern (optimize a tangent delta with autograd) NaNs at its
starting point.

### 4.2 BR's first-order optimizers build dense Jacobians every step — unusable at consumer scale

**Evidence.** `Adam.minimize`: `J = problem.jacobian(state.x)  # (dim, nv)` then `J.mT @ r`
every iteration (`optim/optimizers/adam.py:79-80`); `LBFGS.minimize` likewise
(`lbfgs.py:81-82,148`). With the default `JacobianStrategy.AUTO` → central finite differences,
that is ~`2·nv` full residual evaluations per gradient. For BVR's per-person problem
(`nv ≈ T·159` tangent, T in the hundreds; residual dim ≫ 10⁴) both memory and time are
prohibitive. The Adam docstring even says the `problem.gradient(x)` route "is preferred"
(`adam.py:5-7`) yet the code doesn't use it.

**Consumer impact.** Nobody uses BR's Adam/LBFGS. BVR wrote its own engine; BHF drives
`torch.optim.LBFGS` manually even when the cost is 100% BR residuals (§2.4).

**Severity: High.** BR's matrix-free path (`problem.gradient` + `apply_jac_transpose`,
`problem.py:72-84`) exists and works (BHF uses it) — the optimizers just don't sit on top of it,
and there is no optimizer at all that uses **torch autograd** for the gradient (blocked by 4.1
anyway).

### 4.3 Single flat variable — no multi-block parameters

**Evidence.** `LeastSquaresProblem` has one `x0` (`optim/problem.py:28`); `ResidualState`
carries one `variables` tensor (`residuals/base.py:28-44`); `CostStack.add` takes only a scalar
`weight` per item (`costs/stack.py:47-61`).

**Consumer impact.** Every consumer fit is multi-block: BVR `q` with per-phase DOF masks
(`optimizer.py:94-108`), BVR dynamics `f_world (T,C,3)` (`stages.py:823`), BHF sdf_fit
`(δω, t, log_s)` with per-block lr (`sdf_fit.py:398-412`), BHF smplx_alignment
`(transl, orient, pose, betas, scale)` with per-stage free sets (`refine.py:4-15`). None
expressible in BR. For contrast, pyroki/jaxls' `Var`/`VarValues` pattern solves exactly this
(`BR/references/design/pyroki.md:139-241`).

**Severity: High.** This is the structural reason BVR's engine exists.

### 4.4 `Model.integrate`/`difference` per-joint Python loop — measurably slow, drove a re-implementation

**Evidence.** `model.py:160-190`: Python loop over `njoints`, one small Lie op + slice per
joint, `torch.cat` at the end. Micro-benchmark (CPU, BR venv, T=200, SMPL-24 via
`make_smpl_like_model`):

```
model.difference: 2.874 ms    vectorized twin: 1.211 ms    ratio 2.4×
```

The gap grows with joint count (52-joint SMPL-X ⇒ BVR's claimed ≈4–10×, `motion.py:23`,
plausible but not re-measured here) and will be far worse on GPU (launch-bound: ~52 tiny
kernels vs ~6 batched ones per call).

**Consumer impact.** BVR wrote `tangent_difference` (§3.4) and stopped building a BR model for
the kinematic stage at all, leaving `make_smplx_robot_model` dead.

**Severity: High** for many-joint models (the entire better_human use case). Fix is easy:
group same-kind joints and batch (all sphericals in one `so3.log`).

### 4.5 GN/LM solver ergonomics forced documented workarounds

**Evidence** (all in consumer comments + BR code):

- **No step bounding**: *"BetterRobot's GN/LM solvers don't bound step magnitude; without this
  the first inner step … can propose >1 m translations"* — hand trust-region clip inside the
  problem's own `step()` (`_icp_problem.py:51-74`). BR GN indeed applies the full
  Cholesky step unconditionally (`gauss_newton.py:54-63`).
- **Absolute damping is unsafe**: *"Absolute eps (BetterRobot's default 1e-8) is unsafe here:
  our point-to-plane JtJ diagonal scales with the inlier count (~5000 …), so a fixed eps either
  over- or under-damps depending on M"* (`icp.py:209-217`); consumer recomputes
  `eps = 1e-3·max(diag(JtJ))` per outer iteration (`icp.py:326-330`). BR GN: `eps: float = 1e-8`
  added as `eps·I` (`gauss_newton.py:23,53`).
- **Scale-dependent stopping**: convergence checked externally on `|Δrmse|` because GN's
  `‖JᵀR‖ < tol` *"scales with data magnitude and is hard to tune across problems"*
  (`icp.py:295-297`; BR `gauss_newton.py:69`).
- **Private-attribute contract**: the duck-typed problems must provide `_nv`
  (`_icp_problem.py:99,142`) because solvers read `problem._nv` (`gauss_newton.py:45`).
- **Per-iteration host syncs**: `float(Jtr.norm())`, `float(0.5·(r@r).sum())` every iteration
  (`gauss_newton.py:65-69`, `levenberg_marquardt.py:88,113`), `torch.eye` allocated per
  iteration (`gauss_newton.py:53`, `levenberg_marquardt.py:97`) — the exact pattern BVR's
  engine explicitly avoids (`tools/optim.py:80-82`).

**Severity: Medium-High.** The one consumer who *did* adopt a BR solver had to wrap it in three
layers of corrective machinery.

### 4.6 Solvers are unbatched

**Evidence.** GN/LM assume 1-D residual/param (`(dim, nv)`, `r @ r`, scalar `float()` costs —
`gauss_newton.py:49-66`, `levenberg_marquardt.py:90-113`). No `(B,…)` problem batching.

**Consumer impact.** BVR's workload is P persons × (and in ICP-like settings, many objects ×)
independent small/medium fits; batching is BR's headline promise ("batched by default") yet the
optimization layer can't batch. BVR instead loops persons (`stages.py:947`).

**Severity: Medium** (High if BR wants GPU throughput as a selling point).

### 4.7 No camera/projection, point-cloud, or per-term robust-kernel support

**Evidence.** Residual library covers pose/position/orientation/limits/rest/velocity/accel/
contact-consistency/reference (BR CLAUDE.md); nothing takes intrinsics, a point cloud, an SDF,
or a per-datum weight vector. Kernels exist but only enter globally through solver-level IRLS
(`levenberg_marquardt.py:27-44`); `CostStack.add` has no `kernel=` (`costs/stack.py:47-61`).
Meanwhile: BVR's data terms are all robustified per-point (GM, `losses.py:26-36`) with
per-point score weights (`losses.py:67`); BHF applies Cauchy per-point manually
(`sdf_fit.py:234`).

**Severity: High** for the "replace their losses" goal — this is most of what the consumers
actually optimize.

### 4.8 Viewer public API insufficient for playback tooling

**Evidence.** BHF `playback.py:103-105` grabs `viewer._scene` and `viewer._backend` (with a
runtime check that they exist) to recolor per-joint spheres per frame and scale them by torque;
`TrajectoryPlayer.seek/.step/.pause/...` are stubs (roadmap). BVR's viewer
(`tools/viewer/app.py:369`) uses raw viser and ignores BR's viewer entirely.

**Severity: Low-Medium.**

### 4.9 Model post-construction tweaks require `dataclasses.replace`

**Evidence.** To set gravity from GeoCalib, BVR does
`model = dataclasses.replace(model, gravity=g6)` (`stages.py:795`) — no builder parameter or
API for gravity.

**Severity: Low** (but a tell about builder ergonomics).

### 4.10 Stale/contradictory documentation confirmed

- BHF `optimize_motion.py:281-284` justifies the autograd bypass with *"pypose's
  `SE3.Log().backward()` has a factor-of-2 error … (see `BetterRobot/src/better_robot/kinematics/CLAUDE.md`)"*
  — that CLAUDE.md no longer mentions pypose (grep: 0 hits); PyPose was removed in P10-D. The
  *stated* reason is stale, but the bypass remains necessary today because of §4.1 and §4.2.
- `lie/CLAUDE.md` "smooth and differentiable across the singularity" vs measured NaN (§4.1).
- `adam.py:5-7` says the gradient route is "preferred" while the code uses the dense-Jacobian
  route (§4.2).
- BVR `losses.py:431` and `motion.py:30` reference `tools.human_optim.velocity` — module does
  not exist (deleted; it was the only FK-Jacobian consumer).

**Severity: Low** individually; collectively they show doc-claims must not be trusted.

---

## 5. Suspected problems needing verification

1. **52-joint `Model.difference` speedup factor.** Measured 2.4× at 24 joints/CPU/T=200;
   BVR claims ≈4–10× at 52 joints (`motion.py:23`). Direction confirmed, magnitude at 52
   joints and on GPU unverified. Expect GPU to be worse (launch-bound).
2. **`so3.log` gradient near identity.** The exp-side NaN is confirmed; my log-side probe
   through division-normalized identical quaternions returned finite grads (likely because the
   incoming gradient is exactly 0 there), but forward-filled/held frames produce *bit-identical*
   adjacent quaternions in both consumers (BHF `motion.py:219-242`), and the
   `sqrt(0)`-inside-`torch.where` pattern (`_torch_native_backend.py:174-183`) is the classic
   NaN-leak shape. Needs a targeted gradcheck at and around identity with nonzero upstream
   gradients before declaring `log` safe.
3. **Whether `tools/smplx_alignment` (BHF, BR-free) is still an active pipeline** or slated to
   fold into the newer object_align/static flow — determines how much weight its (identical)
   multi-block gap should carry. Its non-legacy location and README references suggest active.
4. **FD Jacobian accuracy/cost on consumer-style data terms** (chamfer/SDF with detached NN):
   BR's AUTO strategy is central FD (`BR/CLAUDE.md`); with re-associated correspondences FD
   across the association boundary can be wrong; untested here because consumers never route
   these terms through BR.
5. **BVR's `pypose` dependency is likely removable** (`pyproject.toml:13`; no non-legacy
   imports found) — worth confirming with a full run.

---

## 6. What is actually good and should be kept

1. **The `lie` functional facade is the adoption success story.** Both projects import
   `se3`/`so3` everywhere (§2.1) — including inside their own optimizers — and BHF's
   `transforms.py:5-10` explicitly consolidated *onto* BR ("same algebra, same Taylor
   stitching … consolidated in one place across the two repos"). Plain-tensors-in/out, scalar-last
   quaternion, `[lin, ang]` tangent — conventions are consistently adopted with no complaints.
   Keep the functional style; fix the numerics (§4.1).
2. **`build_kinematic_tree_model` is exactly the right shape of API.** Both projects build
   SMPL bodies programmatically with name/parent/translation/`root_kind`/`child_kind` +
   per-body inertia kwargs (§2.2) — this is the generic hook the better_human rewrite needs.
   Keep; extend (joint-order option, gravity kwarg).
3. **`rnea(model, data, q, v, a, fext=…)` with a scatterable `fext`** is used in anger by both
   dynamics stages and works, including autograd through `fext` into LBFGS
   (`stages.py:830-850`, `optimize_dynamics.py:276-300`). The Model/Data split
   (`create_data(batch_shape=(T,))`) batches FK/RNEA over whole clips cleanly.
4. **The matrix-free `stack.gradient` / `apply_jac_transpose` path** is validated by a real
   consumer at trajectory scale (`optimize_motion.py:283-285` praises it as "O(T·nv) sparse
   J^T r"). It should become the backbone of BR's first-order optimizers instead of the dense
   route.
5. **Robust kernels as standalone objects** (`Cauchy.rho/weight` on squared norms) proved
   independently reusable outside any BR solver (`sdf_fit.py:193-198,234`). Keep the tiny
   class-with-`rho`/`weight` contract; expose it as a first-class per-term option.
6. **`Trajectory` as a plain container + `so3.slerp`/`se3.sclerp`** enabled BHF's smoothing
   module cheaply. `ContactConsistencyResidual`'s `contact_weights` design (per-frame float
   weights) matched real annotation data with zero friction (`optimize_motion.py:254-272`).
7. **`solve_ik`/`solve_trajopt` were never the ask.** Nothing in either consumer needs the IK
   task layer; what they need is the layer *below* it (problem/solver/residual toolkit). That
   validates BR's layered design intent — the toolkit layers just need to be usable standalone.

---

## 7. Gap list & recommendations (prioritized; concrete change · effort S/M/L · risk)

**G1 — Fix `exp` (and audit `log`) autograd at θ=0.** Taylor-stitch `qw`
(e.g. `qw = cos` via a `theta2`-parameterized branch or compute `half²` products without
`sqrt`), add gradcheck *at exactly zero* and at forward-filled identical configurations for
`so3/se3 exp/log/integrate/difference`. Effort **S**. Risk: low (numerics-only, testable).
Unlocks: autograd tangent-delta optimization, `model.integrate`-based parameterisations —
the pattern both consumers want.

**G2 — Multi-block variables ("Var" layer).** Let a problem own named parameter blocks with
per-block: manifold (euclidean/SO3/SE3/log-scale), bounds, freeze masks, lr/trust scaling —
`problem.add_var("q", init, manifold=…)`, residuals declare which vars they read (pyroki/jaxls
pattern, `references/design/pyroki.md`). Effort **L**. Risk: medium — this is the redesign
center of gravity; do it before 1.0, migration cost grows with every new residual.

**G3 — First-order optimizers on autograd/matrix-free gradients + phase scheduling.**
Replace dense-J Adam/LBFGS with: (a) autograd path over `0.5‖r‖²` (needs G1), (b) the existing
`problem.gradient` path; add a `Phase`/schedule abstraction (per-phase weight columns, active
var sets, iters, optimizer choice, `on_start` hook) equivalent to BVR `tools/optim.py` — that
188-line file is effectively the requirements spec, including sync-free logging. Effort **M**
(given G2). Risk: low-medium. Deletes: BVR `optim.py`, the loop halves of BHF
`optimize_motion/optimize_dynamics`, `sdf_fit`/`polish` loops (with G6's snapshot/NaN-guard).

**G4 — Batch/vectorize `Model.integrate/difference`** by grouping same-kind joints (one batched
`so3.log/exp` over all sphericals, etc.); keep the per-joint API for exotic mixes. Effort **S-M**.
Risk: low (numerically identical; contract-testable). Deletes BVR `tangent_difference` and
revives BR models in the kinematic stage. Measured headroom: ≥2.4× CPU at 24 joints, more at 52+/GPU.

**G5 — Vision/scene residual pack + per-term robust kernels.**
- `CameraProjectionResidual(K, extrinsics, targets, weights, kernel=…)` — pinhole, per-point
  score weights (spec: BVR `losses.py:39-106`, BHF `losses.py:43-70`);
- point-cloud terms: masked/padded chamfer, point-cloud-SDF penetration/attraction/clearance
  sharing one detached NN pass (spec: BVR `losses.py:135-407`), backed by a batched NN utility
  (adopt/port BHF `SpatialHashNN`);
- `kernel=` per `CostStack` item (incl. Geman-McClure) applied at residual level, not only
  solver IRLS;
- shared-intermediate mechanism (lazy state entries) so several residuals reuse one expensive
  pass.
Effort **L** total, shippable piecewise (projection **S**, chamfer **M**, SDF **M**).
Risk: medium — scope creep into "vision library"; keep terms geometric and camera-model-thin.

**G6 — Solver-quality upgrades (from BHF's documented workarounds).** Relative damping
(`λ = α·max(diag(JᵀJ))`), optional per-block step bounds/trust region on GN, relative +
step-based convergence criteria, batched solves over leading dims, no per-iteration `.item()`/
`float()` syncs or `torch.eye` allocations, best-iterate snapshot + NaN rollback (spec:
`icp.py:160-341`, `sdf_fit.py:498-567`). Make `nv` public in the problem protocol. Effort **M**.
Risk: low.

**G7 — Joint-ordering ergonomics.** Either `build_kinematic_tree_model(order="given")`
(topologically-valid caller order) or a first-class `LayoutMap` (name-keyed q/v/body permutation
objects, invertible, tensor-applicable) replacing the four hand-written remappers
(BVR `model.py:87-128`, BHF `motion.py:71-181`). Effort **S-M**. Risk: layout invariants
touch FK internals — needs contract tests.

**G8 — Inverse contact-force task.** `solve_contact_forces(model, q_traj, contacts, gravity, weights)`
wrapping the duplicated BVR/BHF pattern (§3.7): fext scatter + base-wrench/force-magnitude/
smoothness objective + solver. Effort **M** (mostly assembly of existing pieces). Risk: low;
API design should wait for G2/G3 so forces are just another var block.

**G9 — Small utilities with outsized dedup value.** `so3.from_euler/to_euler`
(removes BHF's last pypose call, `motion.py:61`); `se3.from_matrix/to_matrix` (4×4 interop —
currently hand-rolled twice); weighted/batched Umeyama (`transforms.py:125-275` is ready to
upstream); mesh→(mass, com, inertia) helper (trimesh-optional, spec: `smpl_inertia.py:76-149`,
`inertia.py:53-118`); trajectory kernel-smoothing (upstream BHF `smoothing.py`, de-loop the
batch dim). Effort **S** each. Risk: minimal.

**G10 — Viewer playback hooks.** Public per-frame update API (per-body color/scale, overlay
data streams) so BHF `playback.py` stops importing `viewer._scene/_backend`; implement the
`TrajectoryPlayer` stubs it needs (`seek/step/pause`). Effort **M**. Risk: low.

**Deliberate non-goals** (things consumers hand-roll that BR should *not* absorb): visibility/
front-facing masks, Sapiens/Goliath keypoint mappings, contact-label detection heuristics,
nvdiffrast silhouette rendering, RANSAC plane fitting — these are perception/project-specific.
BR's job is the parameter blocks, residual scaffolding, robust kernels, solvers, and Lie/FK/
dynamics kernels under them.

---

## 8. Open questions

1. **Which BHF pipelines are load-bearing?** `tools/smplx_alignment/` (BR-free) vs the newer
   `object_align` static flow — if the former is retiring, its gaps are corroborating evidence
   rather than requirements.
2. **Scalar-loss support vs least-squares purity.** Many consumer terms (GM means, hinge
   penalties, mask MSE) are natural *scalars*, not residual vectors. Should BR's cost layer
   accept scalar terms for first-order phases (as BVR's engine does) while keeping vector
   residuals for GN/LM phases — or force everything into (possibly weighted) residual vectors?
   This decision shapes G2/G3/G5.
3. **Where does the camera live?** A projection residual needs intrinsics/extrinsics types.
   Minimal `(K, T_cam)` tensors (my recommendation) vs a camera abstraction — risk of scope
   creep either way.
4. **Ragged per-frame data.** BVR pads per-frame clouds + validity masks (`losses.py:109-132`)
   — should BR standardize a padded-batch convention (tensors + mask) for data-carrying
   residuals, or stay out of it?
5. **`torch.compile` vs Python-loop kernels.** The roadmap plans `@torch.compile` on FK; would
   compile close the `Model.difference` gap (§4.4) enough to skip G4, or is same-kind batching
   still needed for eager mode? (My measurement says do G4 regardless — eager is the consumer
   reality today.)
6. **Does anything need `solve_ik`?** Neither consumer does. If better_human's retargeting
   will, the IK layer earns its place; otherwise its API stability should not constrain the
   G2/G3 redesign underneath it.
