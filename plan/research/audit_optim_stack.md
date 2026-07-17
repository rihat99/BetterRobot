# Audit: Optimization Stack (residuals, costs, optim, tasks, collision)

Auditor: read-only architecture audit, 2026-07-16.
Modules: `src/better_robot/{residuals,costs,optim,tasks,collision}/` — 4,222 LOC total
(residuals ≈ 1,291; optim ≈ 1,428; tasks ≈ 1,018; costs ≈ 210; collision ≈ 242).
Tests for this stack: `tests/{optim,tasks,residuals}` = 1,139 LOC, 58 tests, all passing
(`uv run pytest tests/optim tests/tasks tests/residuals -q` → `58 passed in 24.37s`).

## 1. Scope & method

- Read every file in the five modules plus the coupling points:
  `kinematics/jacobian.py::residual_jacobian`, `kinematics/jacobian_strategy.py`,
  `optim/state.py`, `costs/stack.py`, `costs/factory.py`.
- Ran runtime probes (CPU; the box's CUDA driver is too old for this torch build, so no GPU
  numbers) under `uv run python`: batched `solve_ik`, batched `SolverState.from_problem`,
  `JacobianStrategy.AUTODIFF` dispatch, analytic-vs-FD Jacobian timing, `costs.factory`,
  dtype handling, B-spline quaternion norms, LM convergence with/without box bounds,
  `dataclasses.replace` on `_ChainRuleProblem`.
- Checked consumer reality: `BetterVideoReconstruction/tools/human_optim/stages.py` (what BR
  would need to replace), and compared against `references/design/{pyroki,crocoddyl,jaxopt}.md`
  + `BEST_PRACTICES.md`.
- Cross-checked docs claims (`docs/concepts/tasks.md`, `docs/reference/roadmap.md`,
  module CLAUDE.md files) against code. Several are false; flagged below.

Severity scale: **Critical** = blocks the library's stated purpose for consumer projects;
**High** = wrong result / large perf or extensibility cost; **Medium** = dead weight,
misleading API, latent bug; **Low** = taste/consistency.

---

## 2. Confirmed problems (with evidence)

### 2.1 CRITICAL — The decision variable is hardwired to "one robot configuration q"

This is the gap blocking `BetterVideoReconstruction` / `BetterHumanForce` adoption, and it is
structural, not incidental:

- `ResidualState` (`residuals/base.py:28-45`) is `{model: Model, data: Data, variables: Tensor}`.
  `model` and `data` are **mandatory** — a residual over camera extrinsics, SMPL betas, or
  contact forces has no honest way to receive its variables; it must ride along in a fake FK
  `Data`.
- `variables` is a single flat tensor with a single retraction. `LeastSquaresProblem`
  (`optim/problem.py:22-40`) has one `x0`, one scalar `nv`, one `retract`, one box
  `(lower, upper)`. There is no concept of variable *blocks* (q + extrinsics; per-frame qs +
  shared betas; forces per contact).
- Built-in residuals interpret `state.variables` **as q by fiat**: `JointPositionLimit.__call__`
  reads `state.variables` directly as the configuration (`residuals/limits.py:56-62`);
  `RestResidual` likewise (`regularization.py:48-54`). If you pack `[q, camera]` into `x` with a
  custom `state_factory`, every built-in residual silently applies joint semantics to camera
  parameters.
- The AUTO Jacobian fallback hardwires q even harder: `residual_jacobian`
  (`kinematics/jacobian.py:263-284`) perturbs `model.integrate(q, v)` with
  `v0 = torch.zeros(model.nv)` and returns a `(dim, model.nv)` matrix. For any variable that is
  not exactly one robot configuration (trajectory `(T, nq)`, q+extrinsics, forces) the FD
  fallback returns a Jacobian of the **wrong shape/meaning** without raising.
- Real consumer code confirms the mismatch: `BetterVideoReconstruction/tools/human_optim/stages.py:776-905`
  optimizes contact forces `f_world (N, C, 3)` through `br.rnea` with `torch.optim.LBFGS` and a
  closure — BR kinematics/dynamics are used, the BR optimization stack is not, because it
  *cannot express the problem*. Same story for camera+betas+q in `human_optim/motion.py`.

Why it matters: the user's #1 adoption goal ("BR should replace the hand-rolled loops") is
impossible under the current `ResidualState`/`LeastSquaresProblem` shape. Contrast: pyroki
residuals are plain functions over typed variable objects (`jaxls.Var`), so `pose_cost(vals,
joint_var, cam_var, …)` is trivial (`references/design/pyroki.md:160-210`); Ceres has parameter
blocks; Crocoddyl separates state manifolds from residual models.

### 2.2 CRITICAL — Batched solving does not exist, but the docs promise it

`docs/concepts/tasks.md:161-174` ("Warm-started batched IK") shows
`solve_ik(model, targets, initial_q=torch.randn(128, model.nq))` returning `(128, nq)`.
Reality (runtime probe):

```
solve_ik(model, {...}, initial_q=q0.expand(4, -1))
→ RuntimeError: mat1 and mat2 shapes cannot be multiplied (4x33 and 4x33)
```

Root causes, all in the solver loop:

- `SolverState.from_problem` (`optim/state.py:95`): `0.5 * (r0 @ r0)` — a matmul that only
  works for 1-D residuals. Crashes immediately on `(B, dim)`.
- `LevenbergMarquardt.minimize` (`optim/optimizers/levenberg_marquardt.py:86-131`): scalar
  Python damping (`state.damping: float`), scalar cost (`cost = float(state.residual_norm)`),
  one global accept/reject branch, one global convergence check (`float(Jtr.norm()) < tol`).
  Even if the shape bugs were fixed, this is *one* problem with stacked residuals, not B
  independent problems — no per-batch-element damping, acceptance, or early stopping.
  `GaussNewton`, `Adam`, `LBFGS` share the same scalar structure
  (`gauss_newton.py:65`, `adam.py:96-100`, `lbfgs.py:114-134`).
- FK, residuals, and analytic Jacobians *are* batch-shaped (`(B..., dim, nv)`), so the batching
  story dies exactly at the optim layer. cuRobo/pyroki treat batched IK-with-per-element
  convergence as the core feature; here it is fiction.

Also contradicts `CLAUDE.md §Batching Rules` ("All tensors carry a leading batch dimension. No
unbatched mode") and `residuals/base.py:44` (`variables: (B..., nx)`).

### 2.3 CRITICAL — Box-bounded LM stalls even when the solution is strictly interior

Bounds are handled by clamping the trial point after the LM step
(`levenberg_marquardt.py:103-107`). Runtime probe on Panda, pure pose cost, target generated
from `qt = 0.7·lower + 0.3·upper` (strictly inside bounds), start at clamped neutral:

```
unbounded  LM: converged in 25 iters, final cost 2.4e-14
bounded    LM: maxiter at 300 iters, final cost 3.4e-02   (same problem!)
near-start LM: converged in 17 iters                       (local convergence is fine)
```

Consequence at the facade level: `solve_ik` **with default config fails an easy feasible
target** — `lm: 100 iters, converged=False, pos_err=0.31 m (1.2 s)`;
`gn: pos_err=1.21 m` (probe scripts in scratchpad; targets from FK of an in-bounds q).
The projected point breaks LM's ratio test: the unconstrained step keeps pointing out of the
box, the clamped point doesn't decrease cost, `Adaptive.reject` doubles λ to its 1e8 cap
(`strategies/adaptive.py:24-25`), and the loop burns `max_iter` without ever setting
`status="stalled"` (only LBFGS ever sets it — `lbfgs.py:144`; LM/GN grep-clean).
The existing test suite never sees this because `test_solve_ik_reachable_target` uses a
one-joint 0.3 rad offset with `limit_weight=0.0, rest_weight=0.0` and a 1 cm tolerance
(`tests/tasks/test_ik_regression.py:89-105`).

Proper treatments: scipy TRF-style interior reflection, Ceres' projected line search, or at
minimum a gradient-projection acceptance test. Today the flagship API silently returns bad
solutions with `converged=False` at best.

### 2.4 CRITICAL — `collision/` is 100 % stub, and the roadmap actively misrepresents it

Every executable body in the module raises `NotImplementedError`:
`geometry.py::colldist_from_sdf` (line 66), `pairs.py::distance` (line 40),
`closest_pts.py::point_to_segment/segment_to_segment` (lines 23/36), and all four
`RobotCollision` methods incl. `from_model` (`robot_collision.py:47-73`). The only real code is
dataclass field declarations and an unused `@register_pair` dict.

`docs/reference/roadmap.md` says *"If a symbol is not on this page, it is implemented and
tested"* — but only `SelfCollisionResidual`/`WorldCollisionResidual` appear on the page; the
entire collision substrate they depend on is absent from the list. The workspace CLAUDE.md's
"collision/ … (port of old capsule mode)" reads as if a port happened; it did not.
Answer to audit Q6: **not capsule-only — capsule-nothing**; unusable for humanoid
self-collision costs today. `solve_ik(..., robot_collision=...)` takes the argument and ignores
it (`tasks/ik.py:157`, docstring line 171: "unused in this version").

### 2.5 HIGH — Extensibility: the advertised plug-in path is closed or broken at every door

Walkthrough of "add a keypoint-reprojection residual and use it" today:

1. **Write the class** — genuinely small (protocol is structural: `name`, `dim`,
   `__call__(state)`, `jacobian(state)->Tensor|None`; ~20 lines). Good.
2. **`costs.factory` (the pyroki-style plain-function wrapper) is a stub** —
   `costs/factory.py:28` raises `NotImplementedError`. Verified at runtime. The one-line path
   advertised by the docstring does not exist.
3. **`@register_residual` is pure ceremony**: `get_residual` has **zero callers** in `src/`
   (grep) — nothing instantiates by name, no config-driven loading, no serialization. Its only
   observable behaviors are overwriting `cls.name` and raising on duplicate names
   (`registry.py:23-27`), which makes two independent downstream projects that both pick
   `"keypoint"` mutually incompatible at import time. The registry is not earning its keep.
4. **You cannot plug the residual into `solve_ik` at all.** `solve_ik(model, targets,
   initial_q, cost_cfg, optimizer_cfg, robot_collision)` (`tasks/ik.py:150-158`) builds its
   own fixed CostStack (pose+limits+rest, lines 190-209) with no `extra_costs=`/stack hook.
   The custom residual forces you down to the raw layer: build `CostStack`, write a
   `state_factory` that runs FK with `compute_frames=True` (forget it and frame residuals
   throw), construct `LeastSquaresProblem` with 7 fields including `retract=lambda q, dv:
   model.integrate(q, dv)` and `nv=model.nv`, instantiate optimizer + linear solver + damping
   strategy. ≈ 30-35 lines of boilerplate that `solve_ik` itself duplicates.
5. **If you skip the analytic Jacobian you get the FD fallback**, which is (a) 42× slower than
   analytic on Panda CPU (measured: 62.6 ms vs 1.48 ms per Jacobian — it re-runs full FK
   2·nv times per residual per iteration, `kinematics/jacobian.py:266-283`) and (b) simply
   wrong for anything that isn't a single robot configuration (§2.1).
6. **The autodiff option is a lie**: `JacobianStrategy.AUTODIFF` and `.FUNCTIONAL` are
   documented as `torch.func.jacrev/jacfwd` (`jacobian_strategy.py:14-18`) but
   `residual_jacobian` has no branch for them — any strategy other than a successful
   analytic call **silently falls through to finite differences**
   (`kinematics/jacobian.py:246-259`). Verified at runtime. For a "PyTorch autograd
   throughout" library, autograd is unreachable from the Jacobian dispatcher.

Comparison: pyroki = write a plain function, wrap with `Cost.factory`, done
(`references/design/pyroki.md:192-207`); crocoddyl = subclass `ResidualModelAbstract`
(`calc`/`calcDiff`) — more ceremony but every piece works. BR has crocoddyl-level class
ceremony with less capability than pyroki's plain functions.

### 2.6 HIGH — Robust-kernel IRLS is inconsistent with the step-acceptance test

`_apply_kernel` (`levenberg_marquardt.py:27-44`) reweights residual and Jacobian rows by
`sqrt(kernel.weight(r_i²))` — fine as IRLS. But step acceptance and gain ratio compare **raw
L2 costs**: `cost_new = float(0.5 * (r_new @ r_new).sum())` (line 110). With a non-L2 kernel,
LM minimizes the robustified objective while accepting/rejecting steps against the plain
quadratic one. Every kernel implements `rho()` (`kernels/huber.py:14-27` etc.) precisely for
this purpose and **`rho` has zero callers** (grep). Ceres evaluates the robustified cost for
acceptance. Consequences: good robust steps get rejected near outliers, damping schedules are
driven by the wrong objective, and Tukey (weight→0) can accept steps that increase the robust
cost. Additional semantic wrinkles: weights are applied per scalar *row*, not per residual
block (a 6-D pose residual gets its rows down-weighted independently — Ceres/g2o apply the
loss to the block's squared norm), and rows arriving at the kernel are already scaled by
`CostItem.weight`, so a user's Huber `delta` is in "weighted units" — undocumented.

### 2.7 HIGH — Matrix-free machinery exists but nothing in the shipped stack uses it

- `LeastSquaresProblem.gradient` / `CostStack.gradient` / five `apply_jac_transpose`
  overrides (`smoothness.py`, `regularization.py`, `temporal.py`, `contact.py`, `base.py`) are
  consumed **only by tests**. Adam builds the dense Jacobian anyway
  (`adam.py:79-80`), directly contradicting its own module docstring ("the
  `problem.gradient(x)` route is preferred", `adam.py:5-7`); LBFGS same (`lbfgs.py:81-82`).
- `tests/optim/test_matrix_free.py:1-11` claims "Adam and L-BFGS run … *only* through
  `problem.gradient` (we monkeypatch `problem.jacobian` to fail)"; the actual test body says
  "We don't actually swap Adam yet (P10-C task)" and just evaluates the gradient once
  (lines 81-89). The test docstring documents a test that was never written.
- `ResidualSpec` (`optim/jacobian_spec.py`, 55 lines of structure/time-coupling/affected-knots
  metadata): **zero consumers** in `src/` and zero in `tests/` (grep). Pure dead weight,
  exported in `optim/__init__.py:51`.
- `CostItem.slice` ("populated by CostStack at finalise time", `costs/stack.py:34`) is never
  populated — there is no finalise. `CostItem.kind = "constraint_leq_zero"`
  (`stack.py:21,33`) is never read anywhere: "constraints" are cosmetic strings.
- `scheduler=` is threaded through every optimizer signature
  (`optimizers/base.py:42`, all five `minimize`s, `optim.solve`) and never used by anything.

### 2.8 HIGH — Ornamental solver options that crash when selected

`OptimizerConfig` advertises `linear_solver: "cholesky"|"lstsq"|"cg"` and
`damping: "constant"|"adaptive"|"trust_region"` (`tasks/ik.py:65-67`), and the factory tables
happily build them (`ik.py:102-147`), but:

- `CG.solve` → `NotImplementedError` (`optim/solvers/cg.py:19`)
- `SparseCholesky.solve` → `NotImplementedError` (`solvers/sparse_cholesky.py:15`)
- `TrustRegion.init/accept/reject` → `NotImplementedError` (`strategies/trust_region.py`)

So `OptimizerConfig(linear_solver="cg")` or `damping="trust_region"` passes validation and
explodes mid-solve. Of the advertised 4 linear solvers, 2 are real (Cholesky, LSTSQ — both
~15 lines around one torch call); of 3 damping strategies, 2 are real (both trivial float
maps). The pluggable-protocol layer (4 sub-packages, 4 Protocols, ~250 LOC of scaffolding) is
mostly wrapping `torch.linalg.cholesky` and `lam * 2`. Note `Cholesky.solve` catches bare
`except Exception` (`solvers/cholesky.py:17-19`).

### 2.9 HIGH — Per-iteration host-device syncs and re-allocations throughout the loop

Every optimizer iteration calls `float(...)` on device tensors: LM ×4 (`levenberg_marquardt.py:
86,110,116,126`), GN ×2, Adam ×2, LBFGS ×6 incl. inside the line-search loop
(`lbfgs.py:114,134,152`). Each is a GPU sync point. `torch.eye(nv)` is allocated per iteration
(`levenberg_marquardt.py:96`, `gauss_newton.py:53`), and `PoseResidual` allocates its weight
vector via `r.new_tensor([...])` on **every call and every Jacobian call** (`pose.py:65,102`)
— the exact per-call H2D-copy pattern commit 93b8c03 just removed from `so3_inverse`. This
directly contradicts `CLAUDE.md §torch.compile Friendliness` ("No `.item()` calls in hot
paths"). Not measurable here (no working CUDA), but each sync is ~5-15 µs + pipeline stall,
×4-6 per iter ×100 iters ×(number of IK calls); worse, it forbids CUDA-graph capture ever
working on this loop.

### 2.10 MEDIUM — Trajectory residuals: stateful `dim`, shape ambiguity, dense O(T²) Jacobians

- `VelocityResidual`/`AccelerationResidual` set `self.dim = 0` at construction and mutate it
  during `__call__` (`smoothness.py:56-65,116-125`); `TimeIndexedResidual` mutates `self.dim`
  too (`temporal.py:80`). `CostStack.total_dim()`/`slice_map()` (`stack.py:77-100`) are wrong
  until every residual has been evaluated once, and the "residual must be a pure function of
  ResidualState — no side effects" rule in `residuals/CLAUDE.md` is violated by its own
  library.
- Trajectory-vs-batch ambiguity: a `(T, nq)` `state.variables` means "trajectory" to
  smoothness/contact residuals (`smoothness.py:23-30` requires exactly `dim()==2`) and
  "batch" to point residuals (`PoseResidual` broadcasts). The same tensor shape has two
  incompatible semantics decided per-residual by convention; batched trajectories
  `(B, T, nq)` are expressible nowhere.
- Analytic trajectory Jacobians are built dense with Python loops over knots:
  `(nv·(T−2), T·nv)` filled block-by-block (`smoothness.py:74-81,134-142`;
  `regularization.py:135-139`; `temporal.py:93-95`; `contact.py:116-134` — the latter loops
  `K × T` in Python and calls `float(w_pair[t,k])` per block, a sync per pair per step). For
  a 30-DOF humanoid at T=240: (7 k × 7 k) per residual, several of them, per LM iteration.
  The `apply_jac_transpose` sparse paths exist but nothing calls them (§2.7).

### 2.11 MEDIUM — `solve_trajopt` B-spline path: three latent breakages

- **Bounds silently dropped**: the B-spline branch constructs the problem with
  `lower=None, upper=None` (`tasks/trajopt.py:207`) even when the caller passed limits.
  No warning.
- **Quaternion control points are linearly mixed**: `expand = B @ z`
  (`parameterization.py:153`) — for free-flyer models the expanded trajectory's base
  quaternions are not unit (measured norms 0.28-0.98 on a G1 trajectory with varied base
  orientation). FK then operates on denormalized quaternions; `_retract` is Euclidean on
  control points (`trajopt.py:169`), so nothing ever renormalizes. B-spline + floating base
  is unusable.
- **`MultiStageOptimizer` crashes on the B-spline problem**: `minimize` clones the problem via
  `dataclasses.replace(problem, x0=…)` (`multi_stage.py:85`), which re-invokes `__init__`
  without the subclass's required `dq_dz` kwarg. Verified:
  `TypeError: _ChainRuleProblem.__init__() missing 1 required keyword-only argument: 'dq_dz'`.
  So `solve_trajopt(parameterization=BSplineTrajectory(...), optimizer=LMThenLBFGS(...))`
  cannot run. Also `_ChainRuleProblem` overrides only `jacobian` (`trajopt.py:56-65`) —
  `gradient()` and `jacobian_blocks()` still return q-space quantities of the wrong dimension
  in z-space (latent, since nothing uses them; see §2.7).

### 2.12 MEDIUM — Convergence/termination logic is thin

- LM checks `‖Jᵀr‖ < tol` only after **accepted** steps, using the gradient at the *previous*
  iterate (`levenberg_marquardt.py:126`); a long run of rejections never terminates early and
  never reports `stalled` (§2.3). No step-size (`‖δ‖`) or relative cost-decrease criteria —
  jaxopt/Ceres use all three.
- GN accepts unconditionally (no cost check at all, `gauss_newton.py:56-67`) — it happily
  diverges (measured pos_err 1.2 m in §2.3's probe).
- `SolverState.residual_norm` actually stores `0.5‖r‖²` (`state.py:40-44,95`) — misnamed,
  and per-iteration `history` dicts append unconditionally with no cap.

### 2.13 MEDIUM — `solve_ik` silently casts to float32 and other facade warts

- `initial_q.clone().detach().float()` and `model.lower_pos_limit.float()`
  (`tasks/ik.py:186-188,221-222`): pass float64 in, get a float32 solve and float32 result
  out (verified). No other task does this; `solve_trajopt` preserves dtype. Undocumented.
- `IKCostConfig` has `collision_margin`/`collision_weight` fields (`ik.py:46-47`) that are
  never read — dead config.
- `pos_weight`/`ori_weight` are multiplied by `pose_weight` and baked *inside* the residual
  while `CostStack` has its own per-item weight (`ik.py:193-200`): two weighting systems, and
  the kernel sees pre-weighted rows (§2.6).
- Docs drift: `CLAUDE.md` examples use frame name `"panda_hand"`; actual frames are prefixed
  (`body_panda_hand`) — the README-level example `result.frame_pose("panda_hand")` raises
  `KeyError` (verified while probing).

### 2.14 MEDIUM — Protocol violations inside the library's own residuals

`Residual.jacobian` contract: "return `None` to fall back to autodiff" (`base.py:63-65`). But
`JointVelocityLimit.jacobian` **raises** `NotImplementedError` (`limits.py:110`) — it only
works because `residual_jacobian` also catches `NotImplementedError`
(`kinematics/jacobian.py:251-252`), which in turn means a *genuinely broken* analytic Jacobian
that raises `NotImplementedError` internally is silently swallowed and replaced by FD. Stubs
(`NullspaceResidual`, `JerkResidual`, `JointAccelLimit`, `YoshikawaResidual`,
`SelfCollisionResidual`, `WorldCollisionResidual` — 6 of 17 exported residual classes) raise
from `__call__`, so `registered_residuals()` and `residuals/__init__.py` advertise a library
that is ~1/3 unimplemented.

### 2.15 LOW — Misc

- `CostStack.gradient` return-shape comment is wrong (`stack.py:171`: says
  `(B..., total_dim, nv)`, returns `(B..., nv)`), and its zero-item path returns a flattened
  `zeros_like(variables).reshape(-1)` (`stack.py:170`) — inconsistent with batched shapes.
- `_ChainRuleProblem.jacobian` comment claims `variables=z` (`trajopt.py:61-62`); the
  state_factory actually sets `variables=q_traj` (`trajopt.py:157`).
- `optim/state.py` docstring markets the struct as avoiding "the mjwarp trap" — noise.
- Registry decorator overwrites `cls.name` after the class already defines it (all built-ins
  define both), so the two can drift.

---

## 3. Suspected problems needing verification

- **GPU behavior unverified**: the box's NVIDIA driver (12060) is too old for this torch
  build, so all sync/alloc findings (§2.9) are code-reading + CPU timings. Verify per-iteration
  sync counts with `torch.profiler` or `CUDA_LAUNCH_BLOCKING` comparisons on a working GPU box.
- **LM float32 conditioning**: `solve_ik` forces float32 (§2.13) and forms explicit normal
  equations `JᵀJ` (squares the condition number). For near-singular humanoid Jacobians this
  may need QR on `J` instead. Suspected, not measured.
- **`TimeIndexedResidual._slice_state`** constructs a `Data` by hand and force-promotes the
  cache level via `object.__setattr__` (`temporal.py:55-75`) — likely to silently break when
  `Data` gains fields (velocities, Jacober caches). Fragility suspected; not observed failing.
- **`CostStack.jacobian` with mixed nv-vs-T·nv items**: a stack mixing a plain residual
  (`(dim, nv)`) and a trajectory residual (`(dim, T·nv)`) would fail only at `torch.cat`
  (`stack.py:144`) with an opaque shape error. Not verified which error surfaces first.
- **`Adaptive` damping halving-on-accept** (`strategies/adaptive.py:21-25`) is the weakest of
  the standard schemes (no gain-ratio-based update à la Nielsen/Madsen, even though the gain
  ratio *is* computed and stored, `levenberg_marquardt.py:116-117` — computed then unused for
  control). Suspected slower convergence vs Ceres-style; needs a benchmark.

---

## 4. What is actually good and should be kept

- **The analytic residual Jacobians are correct and fast.** `PoseResidual.jacobian`
  (Jr⁻¹·R-rotated LWA frame Jacobian, `pose.py:68-103`) matches FD to 7.7e-5 (float32) and is
  42× faster (1.48 ms vs 62.6 ms per Jacobian, Panda CPU). Orientation/Position/Limits/Rest
  analytics are similarly clean. `tests/residuals/test_smoothness.py` verifies smoothness
  Jacobians against FD. This is real, hard-won value — keep the math.
- **Unconstrained LM core works**: 17-25 iterations to 1e-14 cost on Panda pose IK — the
  Gauss-Newton algebra, manifold retraction through `model.integrate`, and tangent-space
  handling of the free-flyer are sound.
- **Manifold-aware design choices**: residuals in tangent space (`model.difference`), a single
  `retract` hook on the problem, `nq != nv` handled uniformly (e.g. the `dq/dv` projection in
  `JointPositionLimit.__init__`, `limits.py:44-54`) — this is the right architecture for
  floating-base and matches Pinocchio/pyroki practice.
- **`CostStack` as named weighted composition** (add/remove/set_active/set_weight,
  `stack.py:47-73`) mirrors Crocoddyl's `CostModelSum` and is the right shape; the
  `MultiStageOptimizer` snapshot/restore context manager (`multi_stage.py:113-145`) is careful,
  correct, tested code (`tests/optim/test_multi_stage.py` covers the raise path).
- **Residual-as-protocol (not base class)** is the right call — a residual is duck-typed, no
  inheritance required; the ceremony problems come from what's *around* it, not the protocol.
- **`TrajectoryParameterization`** (knot vs B-spline as a compress/expand pair with a chain-rule
  Jacobian) is the right abstraction shape — pyroki does the same; fix its bugs (§2.11) rather
  than discard it.
- **`Trajectory`** (`tasks/trajectory.py`) is a decent value type: validated shapes, sclerp
  resampling, `to_data`. Its 13 tests pass.
- The **stubs are honest one-liners** (`raise NotImplementedError` pointing at docs) — no
  half-implemented misbehavior *inside* the stub bodies. The dishonesty is in the roadmap's
  "everything else is implemented" claim and the config Literals that reach them (§2.4, §2.8).

---

## 5. Recommendations

Ordered by leverage. Effort: S ≤ 1 day, M ≤ 1 week, L > 1 week.

1. **Redesign around variable blocks (the pyroki/Ceres lesson).** [L, high risk, highest value]
   Replace the single flat `x` with named blocks: `{name: (tensor, manifold, bounds)}`, where
   manifold supplies `retract`/`difference`/`tangent_dim` (Euclidean, SO3, SE3, robot-q are
   instances). `ResidualState` → a values container residuals index by block name; `model/data`
   become optional context (an FK provider keyed on the q-block), not mandatory fields.
   Residuals declare which blocks they touch → the problem assembles per-block Jacobian
   columns; §2.1's camera+q, betas, forces, per-frame-q + shared-statics all become
   expressible, and better_human gets its hook for free. This is a rewrite of
   `problem.py`+`base.py` and a mechanical port of ~10 residuals; the Jacobian math (§4) is
   untouched.
2. **Make the solvers batched for real** [M-L, medium risk]. Vectorize LM over a leading batch:
   `(B, dim)` residuals, `(B, nv, nv)` normal equations (torch batches cholesky natively),
   per-element damping tensor, boolean accept/converged masks, `torch.where` updates. Delete
   all `float()`/scalar bookkeeping from the loop (also fixes §2.9). Convergence = all-elements
   mask or per-element early freeze. This is the single feature that justifies "PyTorch-native
   GPU-ready" vs just calling scipy.
3. **Fix bounded LM before anything ships** [S-M, low risk]. Minimum: evaluate acceptance on
   the *projected* step and add a stall exit (λ ceiling → `status="stalled"`); better: TRF-style
   reflective steps or Ceres-style projected line search. Add a regression test: interior-
   solution bounded IK must converge (§2.3's probe is the test).
4. **Robust kernels: accept on `rho`** [S]. Use `kernel.rho(r²).sum()` for `cost`/`cost_new`
   and gain ratio; apply the loss per residual *block* (or document per-row semantics); state
   that `delta`/`c` are in weighted-residual units.
5. **Delete or implement the ornamental surface** [S, zero risk, big honesty win]:
   `ResidualSpec` (no consumers), `scheduler=` kwarg, `CostItem.slice`, `CostItem.kind`,
   `IKCostConfig.collision_*`, `robot_collision=` on solve_ik, registry (`get_residual` has no
   callers — a residual is just an object you pass to `CostStack.add`; keep a plain dict only
   if config-file loading is actually planned). Remove `"cg"`/`"trust_region"` from
   `OptimizerConfig` Literals until the classes exist. Update the roadmap to list *all* of
   collision/ as stubbed.
6. **Open `solve_ik` and implement `costs.factory`** [S]. `solve_ik(..., extra_costs:
   dict[str, Residual] | CostStack = ...)` plus a working `factory(fn, dim=...)` closes 80 %
   of the extensibility complaint with ~40 lines. Then make `solve_ik`/`solve_trajopt` thin
   sugar over one `solve(problem)` entry point (they already nearly are).
7. **Implement AUTODIFF properly and demote FD** [M]. `torch.func.jacrev` over
   `v ↦ residual(state_factory(retract(x, v)))` gives tangent-space autodiff Jacobians for any
   variable shape — this should be the AUTO fallback (the backend is stated to be
   autograd-clean); keep FD as an explicit opt-in for verification. Make raising
   `NotImplementedError` from `.jacobian` an error, not a silent FD trigger (§2.14).
8. **Trajectory residual hygiene** [S-M]: `dim` must be computable at construction (pass T) or
   the CostStack must size lazily; kill the `(T, nq)`-vs-`(B, nq)` ambiguity by making the
   trajectory a distinct block/axis in the redesign (rec 1); wire `gradient()`/
   `apply_jac_transpose` into Adam/LBFGS or delete the matrix-free layer (currently it's
   untested-in-anger dead weight, §2.7).
9. **B-spline fixes** [S]: thread `lower/upper` through (project expanded traj or clamp
   control points with a warning); slerp-aware expansion or post-expand quaternion
   normalization for free-flyer; make `_ChainRuleProblem` replace-safe (store `dq_dz` as a
   dataclass field with a default).
10. **Port the collision module for real** [M-L] — the old capsule-mode code the docstrings
    reference (pyroki's `collision.py` is ~700 LOC and directly portable) — or cut the module
    and its residuals from the public surface until it exists.
11. **Test the solver, not just the plumbing** [M]: kernel weight/rho unit tests, bounded-LM
    interior-solution regression, batched-solve tests, a "custom residual end-to-end"
    test that a downstream project can copy as a recipe, and a convergence-rate benchmark
    pinned against scipy `least_squares` on the same problems.

## 6. Open questions

1. **Is per-problem batching (B independent IKs) or one-big-problem (trajopt) the priority?**
   The redesign differs: per-element damping masks vs block-sparse linear algebra. cuRobo-style
   batched IK suggests the former is what "GPU-ready" buyers expect; the consumer projects'
   video pipelines need the latter (T ≈ hundreds, shared statics).
2. **Should LM be differentiable (backprop through the solve)?** Nothing in the current stack
   supports it (`x0.clone().detach()`, `float()` everywhere kill the graph), and no consumer
   asked; but learning pipelines (SMPL fitting inside a training loop) would want implicit-diff
   à la jaxopt/Theseus. Decide before the rewrite — it changes whether the loop must stay
   tensor-pure. If yes, Theseus (PyTorch, ~stable) is the reference to study, not Ceres.
3. **Is the CostStack-level `kind="constraint_leq_zero"` a real roadmap item** (augmented
   Lagrangian / interior point) or should constraints stay "weighted penalties"? If real,
   it interacts with rec 3 (bounds) and should be designed once, together.
4. **How much of `optim/`'s pluggability should survive?** Two working linear solvers of 15
   lines each behind a Protocol + factory + config-string plumbing is negative-value today;
   but a genuine sparse/CG solver for trajopt would justify the seam. Depends on Q1.
5. **Who owns FK caching in the redesign?** Today every `state_factory` call re-runs full FK:
   per LM iterate that is `jacobian(x)` (one FK at x, re-doing the FK already computed when x
   was accepted as the previous `x_new`) plus `residual(x_new)` (one FK) — and 2·nv more under
   the FD fallback. A values-keyed context cache (crocoddyl's calc/calcDiff split, pyroki's
   `(residual, cache)` return) should be part of the block redesign, not bolted on later.
