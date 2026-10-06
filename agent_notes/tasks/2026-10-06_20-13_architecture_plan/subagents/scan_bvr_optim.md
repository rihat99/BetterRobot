# BVR optimization scan — what a real downstream needs from `better_robot.optim`

Root: `/home/rikhat.akizhanov/better/BetterVideoReconstruction` (BVR), read-only.
All paths below relative to `tools/`. Line counts from `wc -l`.
Budget-limited scan: optim.py read fully; optimizer.py, kindyn.py, forces.py read at the
key sections; others by outline + grep. Unverified items are marked (unverified).

## 1. Library APIs BVR actually calls

**better_robot.optim: never imported.** `grep -rn better_robot tools/` hits only lie,
kinematics, dynamics. BVR built its own engine `optim.py` (193 lines) on `torch.optim`.
No comment names the reason; the design comments say what BVR needed instead:
named terms gated by per-phase weight columns (`optim.py:12-16`), shared lazy
intermediates (`optim.py:27-50`), DOF freezing by gradient masks that work for Adam AND
L-BFGS (`optim.py:103-105`), mixed Adam warm-up → L-BFGS polish (`optimizer.py:951-956`),
fresh optimizer per phase "no stale momentum leaks across a DOF unfreeze" (`optim.py:148`).
None of these maps onto LM/GN over fixed-width `Residual`s: most losses are robust
scalar sums (Geman-McClure, Huber, chamfer with top-k NN), not residual vectors.

Grouped list of what is called:
- **lie**: `se3`, `so3` (`human_optim/motion.py:34`, `human_optim/camera.py:18`,
  `body.py:29`); `so3.to_matrix` for wrench transforms (`kindyn.py:229-233`).
- **kinematics**: `forward_kinematics`, `update_frame_placements` (`kindyn.py:214-217`,
  `forces.py:129`, `body.py:28`); `kinematics.jacobian.compute_joint_jacobians`,
  `get_frame_jacobian` (`kindyn.py:370-380`, `forces.py:129-160`) — called in a Python
  loop over the 12 contact frames, then `torch.stack` (`kindyn.py:378-379`).
- **dynamics**: `rnea(model, q, v, a, fext=...)` (`kindyn.py:250-257`, `kindyn.py:375`,
  `forces.py:128,152`). No ABA/CRBA/centroidal use. v/a come from BVR's own
  `smplx_robot/dynamics.py:12` `velocity_acceleration_from_trajectory` (50 lines,
  manifold central differences) — not a library function.
- **model surgery**: `dataclasses.replace(model, values=replace(model.values, gravity=...))`
  to set world gravity (`forces.py:138-140`); reads `model.values.body_inertias` for body
  weight (`forces.py:144`).
- **better_human**: `bh.SMPLX`, `bh.ParametricBody`, `SMPLXClassic`, `SMPLX_CONFIG`
  (`body.py:31-33,136,209-223`). `body.py` (353 lines) wraps them as `Body` / "SMPLXMid":
  52 joints, `q (N,211)` (free-flyer 7 + 51 quats·4), `shaped(betas)` cache returning a
  posed robot with baked inertias, `forward(betas, qn)` → joints/vertices/frames_world.
- **viewer**: none in tools/human_optim (BVR viewer is separate, `tools/viewer/`).
- **residuals / tasks / collision / optim**: none.

## 2. BVR's own optimization machinery (`optim.py`, 193 lines)

- **Variables**: `Problem.add_leaf(name, init)` → detached clone with `requires_grad`
  (`optim.py:66-70`). Leaves: `q (T,211)`, optional `betas (10,)`, kindyn adds
  `f_free (T,12,3)` and `gen_u (T,12,4)` friction-pyramid activations
  (`optimizer.py:724-737`). Prefit: `q`, `log_s (1,)` (`prefit.py:257-258`).
  q is an unconstrained Euclidean tensor; manifold handled by `normalize_q` inside the
  forward (`optimizer.py:668`, `human_optim/qmath.py:15`) — quaternions re-normalised each eval.
- **Forward / shared work**: one user `forward()` returns a `State`; eager `put` for posed
  products, `put_lazy` for expensive optional products computed at most once per
  iteration and only if an active term reads them (`optim.py:27-50`). Examples: finite-
  difference chain `d1 → accel → jerk → snap` (`optimizer.py:715-718`), visibility mask,
  scene NN signatures (`optimizer.py:711-714`), RNEA `dyn`, force splits
  (`kindyn.py:283-287`). This is exactly the library's "evaluation-scoped Node" idea, done
  in 25 lines with string keys.
- **Losses**: `add_term(name, fn(state) -> scalar | (scalar, diag_dict))` (`optim.py:53-74`).
  Term returns a raw scalar; engine multiplies by weight. Diagnostics stay tensors until
  log time to avoid per-iteration GPU sync (`optim.py:79-81`).
- **Weights / staging**: a term is active iff its phase weight ≠ 0 (`optim.py:86-89`).
  Weights live in TOML as length-4 columns `[root, kinematics_1, kinematics_2, kindyn_1]`
  (`configs/human_optimization.toml`); the column IS the objective. fps rescaling of
  difference-term weights for kindyn only (`optimizer.py:990-997`, measured reason in comment).
- **Phases**: `Phase(name, iters, weights, optimizer="adam"|"lbfgs", grad_mask, on_start)`
  (`optim.py:99-115`). BVR wraps it in `PhaseSpec(name, weights, iters, dofs, scene,
  optimizer)` (`optimizer.py:83-98`) and translates DOF group names → gradient masks
  (`optimizer.py:101 _dof_mask`, `999-1004`). Budgets (toml lines 14-52): root 150 Adam;
  kinematics_1 full 300 / refine 200 / lbfgs 30; kinematics_2 root 100 / full 200 / refine
  250 / lbfgs 0; kindyn_1 one L-BFGS of 150. lr 0.01, lr_decay 0.1, lr_lbfgs 1.0, history 10.
- **Optimizers**: Adam (fresh per phase, MultiStepLR at 1/2 and 3/4, `optim.py:180-191`);
  L-BFGS strong-Wolfe, one `step` with `max_iter=iters` (`optim.py:161-176`). No LM/GN
  anywhere in BVR. Separate solvers: kindyn warm start = force-only L-BFGS on a frozen-pose
  affine model (`kindyn.py:382-430`); force witness = projected Adam with ExponentialLR to
  1e-3 and `clamp_(min=0)` projection (`forces.py:264-276`), its own loop "since its weights
  multiply inside squared norms" (BVR CLAUDE.md).
- **Freezing**: gradient masking after backward (`optim.py:118-125`), per leaf, per phase.
  Fingers stay frozen even in the "all DOFs free" L-BFGS phase (`optimizer.py:986-988`).
  betas trainable only in phases naming it; forward switches to a "live" betas path
  (`optimizer.py:670-672`, `1001-1002`).
- **Phase hooks**: `on_start` rebuilds scene candidates, detects and freezes auto contact
  labels, builds contact patches under `no_grad` (`optimizer.py:959-996`). Mutable cells
  (`betas_cell`, `scene_cell`, `patch_cell`) carry phase state into the forward closures.
- **Warm start**: kindyn forces seeded by the frozen-pose solve, copied into leaves under
  `no_grad` before `run_phases` (`optimizer.py:1008-1015`).
- **Convergence**: none in the engine. Fixed iteration counts; L-BFGS uses its internal
  tolerances only in the kindyn warm start (`kindyn.py:410-414`). Force witness checks
  KKT after a decaying schedule (`forces.py:264-266`) and raises on non-finite output (`forces.py:283`).
- **Logging**: `print` of every term `raw·w=weighted` + diag scalars, every quarter of an
  Adam phase, every 5th L-BFGS closure (`optim.py:128-134`, `172`, `192`). No history kept,
  no callbacks, no returned trace.

## 3. Generic functionality living in BVR that belongs in a library

| Item | Where | Lines | Library home |
|---|---|---|---|
| Problem / State(lazy) / Phase / run_phases engine | `optim.py` | 193 | optim core |
| PhaseSpec + DOF-group → grad mask | `optimizer.py:83-121` | ~40 | optim (variable masks / groups) |
| quaternion renormalisation of free q | `qmath.py` | 30 | Model.integrate / manifold variable |
| tangent difference on q (vel), its accel/jerk/snap chain | `motion.py` + `optimizer.py:715-718,824-841` | 51+~30 | trajectory residuals (smoothness of any order) |
| v/a from trajectory (manifold central diff) | `smplx_robot/dynamics.py` | 50 | dynamics/trajectory utils |
| robust kernels (Geman-McClure, Huber) | `losses.py:25,290` | ~20 | optim robust losses |
| reprojection with confidences (GM) | `losses.py:38-122` | ~85 | residuals (camera) |
| velocity/accel/pose/orient priors | `losses.py:493-643` | ~150 | residuals (temporal, prior) |
| frame forces → parent-joint `fext` wrenches (lever arm, scatter_add) | `kindyn.py:220-240` | ~25 | dynamics (`fext_from_frame_forces`) |
| frozen-pose affine dynamics `tau_free − Σ J_cᵀ f_c` | `kindyn.py:357-380`, `forces.py:146-160` | ~45 (2 copies) | dynamics (contact Jacobian stack + affine split) |
| friction cone / pyramid parametrisation, cone violation | `kindyn.py:182-210`, `friction.py` | ~60 | residuals (contact) |
| force smoothness of order 1/2/3, magnitude/direction split | `kindyn.py:259-349` | ~90 | residuals (contact/force) |
| DynamicsSpec: whole kin-dyn problem block | `kindyn.py:68-555` | ~490 | tasks recipe (contact-rich ID) |
| projected Adam + KKT stop | `forces.py:181-300` | ~120 | optim (bound constraints) |
| hysteresis/min-span/close-gaps on boolean series | `contacts.py:165,382-460`, `forces.py:369-425` | ~170 (duplicated) | utils, not optim |
| chunked NN / SDF to point cloud | `optimizer.py:149-194`, `losses.py:149-360`, `contacts.py:64` | ~300 | collision (point-cloud SDF) |
| per-term logging | `optim.py:128-134` | 7 | optim callbacks / trace |

## 4. Pain points

- **Library optim unusable for this**: zero imports of `better_robot.optim`; a 193-line
  engine replaces it. The shapes BVR needs (scalar robust terms, phases, masks, Adam+L-BFGS)
  are not expressible as Problem/Variable/Residual/LM.
- **Duplicated physics**: frozen-pose `rnea + frame Jacobians` is written twice
  (`kindyn.py:357-380`, `forces.py:146-160`). Comment `forces.py:150` "once, else every
  frame redoes the pass" — `get_frame_jacobian` per frame in a Python loop recomputes
  unless `compute_joint_jacobians` is called first; no batched multi-frame Jacobian API.
- **No `fext` helper**: BVR transforms world forces into joint-local wrenches by hand
  (`kindyn.py:220-240`), with an explicit note that contributions must accumulate.
- **Gravity setting** requires nested `dataclasses.replace` on Model/values (`forces.py:138-140`).
- **v/a finite differences** live in BVR (`smplx_robot/dynamics.py`), recomputed every
  forward with gradient (`kindyn.py:251`).
- **Manifold**: q optimised as raw R^211 and renormalised in the forward; no retraction.
- **Gradient scale**: comment `optimizer.py:951-953` — metre translations vs quaternions vs
  pixel reprojection vs metre chamfer stall cold L-BFGS; Adam used as a preconditioner.
  Kindyn: "Adam can't co-adapt N-scale forces" (`toml:52`), so kindyn is L-BFGS only.
  A scaled LM/GN with Jacobian scaling could address this; BVR has no such option.
- **Performance tricks**: chunking for NN/SDF/chamfer (`optimizer.py:176` "~4M pair scores
  per chunk"; `chunk` knob `optimizer.py:510`); 15+ `no_grad` sections for detached
  measurements (vis mask, NN signatures, patch building, label detection); `detach()` of
  vertices for visibility; diagnostics kept as tensors to avoid sync (`optim.py:79-81`);
  lazy State to skip unused intermediates; frozen-pose affine trick replacing RNEA calls
  (`kindyn.py:362-366`). No float64 in the torch path (float64 only in numpy
  `vertex_weights.py`). No TODO/HACK/FIXME markers found.
- **Closure-cell state**: `betas_cell`, `scene_cell`, `patch_cell`, `auto_drift_pre=[None]`
  are mutable cells threading phase state through closures (`optimizer.py:979`) — a sign
  the engine lacks a phase context object.
- `optimize_person` is one ~650-line function (`optimizer.py:371-1031`).

## 5. Problem sizes

- Per person, sequential over P people (`optimize_person`, `optimizer.py:371`; `(P,N,…)`
  layout in CLAUDE.md). Whole trajectory optimised jointly: leaf `q (T,211)`, T = clip
  frames (hundreds typical; exact T unverified). Unknowns ≈ 211·T (+10 betas), kindyn adds
  12·3·T + 12·4·T ≈ 84·T force unknowns → ~300·T total, e.g. ~90k for T=300.
- Loss terms: kinematic stage registers 15 base terms + 8 annotation terms
  (`optimizer.py:800-948`); kindyn adds 15 (`kindyn.py:337-349`); prefit 6 (`prefit.py:278-293`).
  ~38 named terms, ~10-20 active per phase.
- Body: 52 joints, 12 contact frames, SMPL-X vertices (10475) subsampled (`sub`).
- Scene: fused point cloud, NN chunk sized to ~4M pairs.
- Batch: no batch axis over problems; the "batch" is the time axis T. Device CUDA
  (`CUDA_VISIBLE_DEVICES=3` convention), float32.
- Iterations: 150–780 per stage (§2), L-BFGS 30–150.

## 6. Wish list and ideal stage code

Needs, in order of evidence: (a) named scalar AND residual terms in one problem;
(b) per-phase weight dict where 0 = off; (c) lazy shared intermediates; (d) per-phase
variable/DOF masks incl. index groups; (e) any `torch.optim` optimizer or LM/GN per
phase, fresh state per phase, schedules; (f) phase hooks with a context object instead of
closure cells; (g) manifold variables (free-flyer + quaternion joints) with retraction;
(h) bound/projection constraints (λ ≥ 0); (i) logging/trace callback returning history;
(j) library helpers: frame-force → fext, batched multi-frame Jacobian, affine RNEA split,
trajectory v/a, smoothness residuals of order k, robust kernels.

Sketch of what `optimize_person` could reduce to:

```python
from better_robot import optim as bo
from better_robot.residuals import temporal, robust
from better_robot.dynamics import contact

P = bo.Problem()
q   = P.variable("q", q_init, manifold=body.config_manifold)      # retraction, no normalize_q
b   = P.variable("betas", betas0, active=False)
f   = P.variable("f", contact.zero_forces(T, 12), bounds=contact.cone(normals, mu=0.6))

@P.node                                                            # lazy, once per eval
def posed(q, betas): return body.forward(betas, q)
@P.node
def tau(q, f): return contact.rnea_with_frame_forces(model, q, f, frames, dt=1/fps)

P.term("reproj",  lambda posed: robust.gm(cam.project(posed.joints) - kp2d, kp_conf, s=100))
P.term("chamfer", lambda posed: scene.chamfer(posed.vertices[:, sub], depth, chunk=4_000_000))
P.term("pose_prior", lambda q: prior.pose(q, q_init))
for k in (1, 2, 3, 4):
    P.term(f"smooth{k}_root",  temporal.diff(q, order=k, dofs="root"))
    P.term(f"smooth{k}_joint", temporal.diff(q, order=k, dofs="joints"))
P.term("base_wrench", lambda tau: tau[:, :6], kind="residual")    # LM-able
P.term("force_smooth", temporal.diff(f, order=1))

W = cfg.weight_columns                                            # {term: [root, k1, k2, kd]}
plan = [
  bo.Phase("root",   W.col(0), bo.Adam(lr=1e-2, decay=(.5, .75)), 150, free={"q": "root"}),
  bo.Phase("full",   W.col(1), bo.Adam(lr=1e-2),  300, free={"q": "joints-fingers"}, on_start=build_scene),
  bo.Phase("polish", W.col(1), bo.LBFGS(history=10), 30, free={"q": "all-fingers"}),
  bo.Phase("kindyn", W.col(3), bo.LM(damping=1e-3), 150, free={"q": "all", "f": "all"},
           warm_start=contact.solve_forces_frozen_pose),
]
result = bo.run(P, plan, log=bo.log_terms(every=0.25))           # result.trace per term
```

## Open questions

- Would BVR accept LM/GN for kindyn (base wrench and force terms are true residuals) if
  the library offered Jacobian scaling? BVR comments only rule out cold L-BFGS and Adam.
- Should the engine own T-chunking/multi-person batching, or stay single-problem?
- Typical T per clip not verified (would need data inspection).
