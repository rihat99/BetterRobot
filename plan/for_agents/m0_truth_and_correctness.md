# M0 — Truth and correctness: Agent Execution Instructions

> **Implementation log (2026-07-17, `dev`):** T0.1's requested zero
> gradient is mathematically incorrect (`sum(so3.exp(w))` has gradient
> `[0.5, 0.5, 0.5]` at zero); safe finite derivatives are implemented.
> Robust kernels use `weight = 2·rho'`, and the front-page snippet also had
> an unresolved path plus an undefined target. T0.10 uses an explicit
> per-call debug flag. Per owner decision, T0.3 rejects non-identity mimics
> but exempts identity tags; those coordinates remain independent until M3.
> CUDA passed with `torch 2.13.0+cu126`; uv now selects that compatible build
> through an explicit CUDA 12.6 package source.

> Read `plan/for_agents/README.md` first. It carries the standing rules
> (deletions land with/after replacements, honesty rule, batched-parity
> tolerances, test commands, consumer-repo grep list). This file assumes
> you have read it.

## Mission

Before BetterRobot grows a new architecture, it must stop lying. M0 is ten
small, independent repairs: kill the NaN gradients at the Lie identity
(θ=0), make the small-angle cutoff safe in fp32, and — everywhere the
library currently advertises a capability it does not have (batched
solving, mimic joints, autodiff Jacobians, `cg`/`trust_region` solvers,
frame names, dtype preservation) — either fix it or make it fail fast with
an honest error and correct docs. Nothing here changes the architecture;
every task is a bug fix or a truthfulness fix that leaves the 897-test
suite green (`uv run pytest tests/ -v`, ~52 s). Later milestones (M2b, M3)
do the real repairs these tasks point at.

## Prerequisites

- **None** — M0 is the first milestone; every task is independent of the
  others (see file-overlap note below). It depends on nothing from M1+.
- Confirm the working tree matches these anchors before you start (they
  were verified 2026-07-17; the code moves):
  - `grep -n "_TAYLOR_THETA2 = 1e-8" src/better_robot/lie/_torch_native_backend.py` → line 26.
  - `grep -n "residual_norm=0.5 \* (r0 @ r0)" src/better_robot/optim/state.py` → line 95.
  - `grep -n "AUTODIFF\|FUNCTIONAL" src/better_robot/kinematics/jacobian_strategy.py` → lines 22–23.
  - `uv run python -c "import torch; from better_robot.lie import so3; w=torch.zeros(3,requires_grad=True); so3.exp(w).sum().backward(); print(w.grad)"`
    prints `[nan, nan, nan]` **today** — this is the T0.1 repro that must
    print zeros after the fix.
  - Panda loads with `nq=nv=9` and frames named `body_panda_hand`
    (not `panda_hand`), `lower_pos_limit.shape == (nq,)` — all verified live.

## Sizing & parallelism

Every task is **effort S** (≤ 1 day) and logically independent — the
roadmap explicitly says "immediate, any order". They can be handed to
parallel agents, **but four groups of tasks edit the same files** and must
either go to one agent or be merged carefully to avoid clobbering:

- **`lie/_torch_native_backend.py` + `lie/tangents.py`:** T0.1 (safe-where)
  and T0.2 (dtype cutoff) both edit these. Do T0.1's structural change
  first, then T0.2 threads the dtype-dependent cutoff through the same
  sites. Recommend one agent for T0.1+T0.2.
- **`optim/optimizers/levenberg_marquardt.py`:** T0.5 (bounded-LM docstring
  / status) and T0.6 (robust IRLS acceptance) both edit it. T0.6 changes
  one line of logic; T0.5 is docstring + a `stalled` status decision.
  Recommend one agent for T0.5+T0.6.
- **docs + per-package `CLAUDE.md`:** T0.3, T0.4, T0.5, T0.8, T0.9 each
  invalidate a doc/CLAUDE.md claim. Each task owns its own doc edits
  (listed per task); T0.9 is the catch-all sweep + the front-page snippet.
  If parallelized, T0.9 runs **last** and re-greps for any missed claims.
- **`tasks/ik.py`:** T0.4 (batched guard), T0.7 (dtype casts), T0.8
  (`OptimizerConfig` Literals). Small, non-overlapping line ranges; safe to
  parallelize with a final merge.

Dependency list: **all independent.** No task blocks another. The only
coupling is file-edit contention above.

## Tasks

### T0.1 — Kill NaN gradients at the Lie identity (safe-`where`)  [S]

**Goal / done-when:**
- `torch.autograd.gradcheck` **and** `torch.autograd.gradgradcheck` pass
  for `so3_exp`, `so3_log`, `se3_exp`, `se3_log`, `right_jacobian_so3`,
  `right_jacobian_inv_so3` **at exactly θ=0** and at θ=1e-9 (fp64).
- `so3.exp(zeros(3, requires_grad=True)).sum().backward()` gives a finite
  (zero) gradient, not `[nan, nan, nan]`.
- A SMPL-like rest-residual backward at the rest pose has **zero NaNs** in
  the gradient (today the core audit measured 96/99 components NaN).
- `JointSpherical.difference(q, q)` backward is NaN-free (identical
  quaternions → log at identity).
- The false claim in `lie/CLAUDE.md` ("stay smooth and differentiable
  across the singularity") and the identical claim in the
  `_torch_native_backend.py` module docstring (":16–17", "so the autograd
  graph stays smooth across θ = 0") are corrected.

**Current state:** `so3_exp` computes `theta = theta2.clamp(min=0.0).sqrt()`
(`lie/_torch_native_backend.py:148`) — `sqrt` has infinite slope at 0 — and
then `qw = torch.cos(half)` (`:159`) sits **outside** any `torch.where`
Taylor switch. In backward, `sqrt`'s `inf` slope meets a `0` upstream
gradient → `0·inf = NaN`, and the NaN leaks through `torch.where`'s
gradient (both branches are always evaluated; `where` only selects the
*value*, not the *gradient*). Same pattern at:
- `so3_log:174` `sin_half = sin_half2.clamp(min=0.0).sqrt()`, switch at `:178`.
- `se3_exp:270` `theta = theta2.clamp(min=0.0).sqrt()`, switch at `:271`.
- `se3_log:296` `theta = theta2.clamp(min=0.0).sqrt()`, switch at `:297`;
  also `cot_half = cos(half)/sin(half).clamp(...)` at `:301`.
- `lie/tangents.py:86` `theta = torch.sqrt(theta2.clamp(min=0.0))` in
  `_so3_jac_coefficients` (feeds `A_full`/`B_full`, selected at `:100–101`);
  `:129` same in `right_jacobian_inv_so3` (`C_full` division at `:135`,
  selected at `:138`).
- `JointSpherical.difference` (`data_model/joint_models/spherical.py:58–62`)
  is downstream of `so3.log` at identity — it should be **fixed by the
  `so3_log` change**, but its test is a separate acceptance criterion.

Verified live: `so3.exp(0)` grad `[nan, nan, nan]`.

**Implementation plan (jaxlie "safe where" idiom):** the rule is *substitute
a harmless dummy value into the quantity under the sqrt/division BEFORE
computing it, then select the correct branch after.* Concretely, replace
the shared `theta`/`sin_half` with a dummy-substituted "safe" version used
only by the full-formula branch, and give **every** output term (including
value-correct ones like `qw`) its own Taylor branch selected by the same
`use_taylor` mask.

1. In `so3_exp`, rewrite so the full branch never sees 0 and `qw` gets a
   Taylor branch (sketch — direction, not final):
   ```python
   theta2 = (omega * omega).sum(dim=-1, keepdim=True)
   use_taylor = theta2 < cutoff                       # cutoff from T0.2
   theta2_safe = torch.where(use_taylor, torch.ones_like(theta2), theta2)
   theta = theta2_safe.sqrt()                         # >= 1 wherever it's "live"
   half = theta / 2.0
   sin_half_over_theta = torch.where(
       use_taylor, 0.5 - theta2 / 48.0, torch.sin(half) / theta)
   qxyz = sin_half_over_theta * omega
   qw = torch.where(use_taylor, 1.0 - theta2 / 8.0, torch.cos(half))  # cos(θ/2)≈1−θ²/8
   return torch.cat([qxyz, qw], dim=-1)
   ```
   The load-bearing change vs today: `theta` uses `theta2_safe` (dummy 1.0
   where small), and `qw` is now inside a `where` with a Taylor branch.
2. Apply the same idiom to the five other sites. For each, introduce a
   `theta2_safe`/`sin_half2_safe` = `where(use_taylor, 1.0, x)` and take the
   sqrt of that; keep the existing Taylor branches; and audit for any
   value-correct term (like `qw`, or `theta` used only in the non-Taylor
   branch) that currently sits outside the `where`. In `so3_log`, `theta =
   2·atan2(sin_half, qw)` is only consumed by `factor_full`, so compute it
   from `sin_half_safe`; the garbage value in the Taylor region is unused.
3. In `tangents.py`, `_so3_jac_coefficients` and `right_jacobian_inv_so3`
   already have `safe = theta2 > cutoff` masks and `torch.where` selects —
   they only need the dummy substitution before `sqrt`/division so the
   unselected full branch has finite gradient.
4. Leave the `.clamp(min=1e-30)` guards in place — they are belt-and-braces
   against the (now-unreachable) 0 denominators and do not hurt; the dummy
   substitution is what fixes the *gradient*.
5. `JointSpherical.difference`: **do not** add a local workaround first —
   re-run its gradcheck after the `so3_log` fix; it should pass. Only if it
   still NaNs (it should not) apply the idiom locally.
6. Fix the two false doc claims: `lie/CLAUDE.md` "Numerics" paragraph and
   the `_torch_native_backend.py:16–17` module docstring. New wording, e.g.
   "…stitched via `torch.where` against a θ² cutoff, with jaxlie-style
   safe-`where` dummy substitution so gradients (and second-order
   gradients) are finite **at** θ=0, not merely near it."

**What to test:**
- Extend `tests/lie/test_torch_backend_singularities.py`: add
  `gradcheck` + `gradgradcheck` (fp64) for each of the six functions with
  input **exactly** at the identity (`omega=zeros`, `xi=zeros`, `q=[0,0,0,1]`,
  `T=identity`) and at 1e-9. Assert `torch.isfinite(grad).all()`. The
  existing `test_se3_log_finite_at_theta_zero` only tests *near* zero via an
  exp→log round-trip — add the direct at-zero cases; do not weaken it.
- Add to `tests/lie/test_tangents.py`: gradcheck of `right_jacobian_so3`
  and `right_jacobian_inv_so3` at `omega=zeros`.
- Add a spherical-joint test (under `tests/data_model/`): gradcheck /
  finite-grad of `JointSpherical.difference(q, q)` with identical unit
  quaternions.
- Add a SMPL-like NaN guard: in `tests/io/test_smpl_like.py` (or a new
  `tests/residuals/` test), build an SMPL-like spherical-joint model, put it
  at rest, evaluate `RestResidual` backward through
  `model.difference`/`so3.log`, and assert `torch.isfinite(grad).all()`.
- Existing `tests/lie/test_torch_backend_gradcheck.py` (fp64 gradchecks on
  random inputs) must stay green.

**Pitfalls / do-not-forget:**
- `write-tests` skill: this repo's default test dtype is **fp32**, but
  gradcheck/gradgradcheck need fp64 for finite-difference precision — this
  is the sanctioned exception (the T0.1 acceptance is explicitly fp64 at 0;
  the fp32 near-cutoff gradcheck is T0.2's job). State the dtype in each
  test.
- `torch.where(cond, a, b)` back-propagates into **both** `a` and `b`;
  masking the *output* is not enough — you must mask the *input to sqrt/div*.
- The pinocchio-parity suite (`tests/test_pinocchio/`) must stay at
  `atol=2e-6` — the Taylor branches are value-correct, but re-verify the
  round-trip test at θ=π−1e-6 still passes (that path is unchanged).
- `so3.log` folds the double cover by flipping when `qw<0` (`:168–169`) —
  keep that; it is orthogonal to the singularity fix.

---

### T0.2 — Dtype-dependent Taylor cutoff  [S]

**Goal / done-when:** an **fp32** gradcheck near the cutoff passes; the
single `_TAYLOR_THETA2 = 1e-8` constant becomes dtype-aware (fp32 switches
to Taylor at a larger θ² than fp64).

**Current state:** `_torch_native_backend.py:26` has one dtype-independent
constant `_TAYLOR_THETA2 = 1e-8`, with a comment claiming it "cover[s] the
dtype precision range (fp32 ≈ 1e-7, fp64 ≈ 1e-14)". `so3_log:178` derives
its own `_TAYLOR_THETA2 / 4.0`. `lie/tangents.py:88` and `:130` use a
**different** hard-coded cutoff `theta2 > 1e-10` (plan doc does not mention
this second constant — flag it). At θ²=1e-8 (θ=1e-4) the full expression
`(1−cos θ)/θ²` in fp32 loses all significance to catastrophic cancellation
(`1 − cos(1e-4)` rounds to 0 in fp32), so fp32 needs to switch to Taylor
*earlier* (larger θ²), around θ²≈1e-5 (θ≈3e-3).

**Implementation plan:**
1. Replace the module constant with a small helper that returns the cutoff
   for a tensor's dtype, e.g.:
   ```python
   _TAYLOR_THETA2_FP32 = 1e-5   # tune against the fp32 gradcheck below
   _TAYLOR_THETA2_FP64 = 1e-8

   def _taylor_theta2(dtype: torch.dtype) -> float:
       return _TAYLOR_THETA2_FP32 if dtype == torch.float32 else _TAYLOR_THETA2_FP64
   ```
2. Thread it through every `use_taylor = theta2 < …` site touched in T0.1
   (`so3_exp`, `so3_log` incl. its `/4.0` scaling, `se3_exp`, `se3_log`) and
   the two `tangents.py` thresholds — all keyed on the *input* dtype.
3. **Tune, don't guess:** the exact fp32 cutoff is evidence-driven. Sweep a
   handful of θ values straddling the cutoff, run the fp32 gradcheck, and
   pick the smallest cutoff that passes for both branches. If a single value
   can't make both the value round-trip and the fp32 gradcheck pass, **stop
   and report the tradeoff** rather than silently picking one.

**What to test:**
- New fp32 gradcheck (in `tests/lie/test_torch_backend_singularities.py`
  or a new `tests/lie/test_taylor_cutoff.py`) for `so3_exp`/`so3_log`/
  `se3_exp`/`se3_log` at θ just below and just above the fp32 cutoff, using
  `gradcheck(..., eps=…, atol=…, rtol=…)` tolerances appropriate to fp32
  (looser than fp64 — document them).
- fp64 behavior must be unchanged: existing fp64 gradchecks stay green, and
  the fp64 round-trip at small θ keeps `atol=1e-10`.
- Pinocchio parity (`atol=2e-6`) stays green.

**Pitfalls / do-not-forget:**
- Update the `:24–26` comment — the "1e-7/1e-14" claim was already wrong.
- `so3_log` uses `sin(θ/2)²` not `θ²` — keep the `/4.0` relationship when
  you make the cutoff dtype-aware.
- Do not introduce a Python `if dtype ==` inside the hot forward path in a
  way that breaks `torch.compile` — a module-level float lookup keyed on
  `x.dtype` (a static property) is fine; a tensor-value branch is not.

---

### T0.3 — Mimic joints: reject at build with a clear error  [S]

**Goal / done-when:** loading a URDF with a mimic joint **raises** with
guidance pointing at the roadmap; the false claim in `data_model/CLAUDE.md`
is removed. Real enforcement is **M3** (say so in the error and in
Out-of-scope).

**Current state (messier than the assessment — verified):** the mimic
metadata pipeline exists but is computationally dead. The URDF parser reads
mimic tags (`io/parsers/urdf.py:246–257`), `build_model.py` resolves them
into `mimic_multiplier`/`mimic_offset`/`mimic_source` arrays
(`:486–504`), and `Model` carries them (`model.py:75–78`). A zero-DOF
`JointMimic` placeholder class exists (`joint_models/mimic.py`) but is never
selected. **No kinematics/dynamics file reads the mimic arrays** (grep: zero
consumers). Verified on the bundled Panda: `nq=nv=9`, both fingers get
independent DOFs, and `mimic_source == (0,1,…,12,12)` — i.e. joint 13
*declares* it mimics joint 12 (multiplier 1.0) but still owns its own q
coordinate, so any gripper the user believes is coupled produces silently
wrong kinematics.

**Implementation plan:**
1. Place the rejection in `build_model.py` at the mimic-resolution loop
   (`:490–502`), inside `if ir_j.mimic_source is not None:` — this is the
   single point where a mimic joint is detected from any parser (URDF, MJCF,
   programmatic). Replace the array population with a raise. Draft message:
   ```python
   raise NotImplementedError(
       f"Joint {ir_j.name!r} is a mimic joint (mimics {ir_j.mimic_source!r}, "
       f"multiplier={ir_j.mimic_multiplier}, offset={ir_j.mimic_offset}). "
       f"BetterRobot does not enforce mimic constraints yet: doing so needs a "
       f"reduced-coordinate map spanning FK, Jacobians, limits, and "
       f"RNEA/ABA/CRBA — scheduled for milestone M3 (see plan/04_roadmap.md, "
       f"M3 item 3). Loading this model today would give every mimic joint an "
       f"independent DOF and produce wrong kinematics. Remove the <mimic> tag "
       f"from the URDF, or track M3."
   )
   ```
   Using `NotImplementedError` (not a silent skip) also makes the joint show
   up in the T0.9 `NotImplementedError`-grep roadmap page for free. (A
   `BetterRobotError` subclass such as the currently-unraised
   `UnsupportedJointError` at `exceptions.py:126` is an acceptable
   alternative if you prefer taxonomy consistency — it accepts a message and
   is otherwise dead code; note the choice in your report.)
2. Once the raise is in place, the `mimic_mult`/`mimic_off`/`mimic_src_list`
   array-population code below the raise becomes unreachable for mimic
   joints — leave the array *allocation* (all-ones/all-zeros/identity) so
   non-mimic models keep their existing `Model.mimic_*` fields with harmless
   defaults; do **not** delete the `Model` fields (that is M3 structural
   work, and a live-consumer grep is required before removing any field).
3. Remove the false "Mimic Joints" section from `data_model/CLAUDE.md`
   (currently: "Handled via tensors … Mimic joints have nq=0, nv=0." — both
   false). Replace with one honest line: "Mimic joints are **rejected at
   build** (`NotImplementedError`); reduced-coordinate enforcement is
   scheduled for M3."

**What to test:**
- New test in `tests/io/` (e.g. extend `test_build_model.py` or
  `test_urdf.py`): build a tiny IR / minimal URDF with a `<mimic>` joint and
  assert `pytest.raises(NotImplementedError, match="mimic")`. The
  programmatic `ModelBuilder` path is the cheapest way to construct an IR
  with a mimic source without shipping a URDF fixture — check whether it can
  set `mimic_source` on an `IRJoint`; if not, construct the `IRModel`
  directly.
- Regression guard: loading the **Panda** must now raise (it has a mimic
  finger). This means any existing test that loads the bundled Panda will
  break — **audit `tests/` for Panda loads first** (`grep -rln
  "panda_description\|_get_panda_urdf" tests/`). This is the one M0 task
  that can turn the suite red across many files. Two honest options, decide
  and report: (a) the Panda genuinely has a mimic finger and *should* now
  reject — update those tests to load a mimic-free variant or a fixed-finger
  Panda; or (b) if the fingers' `multiplier=1.0, offset=0.0` is considered a
  benign identity mimic, scope the rejection to *non-identity* mimics only.
  **Stop and get owner review on (a) vs (b)** — it changes whether the
  flagship test robot loads at all. Do not silently pick one.

**Pitfalls / do-not-forget:**
- The pinocchio-parity suite loads the Panda; option (a) above would break
  it. That suite is the crown jewel — do not weaken it. This is why the
  decision above is owner-gated.
- Do **not** attempt real mimic enforcement here (FK gather etc.) — the
  assessment (§1.3) and codex (A3) both stress it spans Jacobians, limits,
  and all of dynamics; a partial FK-only fix would make FK and dynamics
  disagree. That is `m3_parametric_model_breadth.md` T-mimic.

---

### T0.4 — Batched `solve_ik`: honest `NotImplementedError`  [S]

**Goal / done-when:** a batched call fails with an honest message pointing
at M2b, **not** the `mat1 and mat2 shapes cannot be multiplied` crash at
`optim/state.py:95`. Docs stop claiming batched solving.

**Current state:** `solve_ik(model, {...}, initial_q=q_batch_4)` crashes at
`SolverState.from_problem` (`optim/state.py:95`): `residual_norm=0.5 * (r0 @
r0)` assumes a 1-D residual and does a matmul (verified live:
`RuntimeError: mat1 and mat2 shapes cannot be multiplied (4x33 and 4x33)`).
Even if that line were fixed, all four optimizers use scalar Python
damping/cost/accept logic — batched solving genuinely does not exist. Docs
(`docs/concepts/tasks.md:161–173`, "Warm-started batched IK", a `(128, nq)`
example) and `CLAUDE.md` "Batching Rules" advertise it.

**Implementation plan:**
1. Primary guard in `solve_ik` (`tasks/ik.py`), right after `x0` is built
   (`:186–188`), before constructing the problem:
   ```python
   if x0.dim() > 1:
       raise NotImplementedError(
           f"solve_ik received a batched initial_q with batch dims "
           f"{tuple(x0.shape[:-1])}. The optimizer stack is single-problem "
           f"only today — LM/GN/Adam/LBFGS use scalar damping, cost, and "
           f"accept/reject logic. Batched solving (per-element damping and "
           f"convergence) is scheduled for milestone M2b "
           f"(see plan/04_roadmap.md). Pass a single (nq,) configuration, or "
           f"loop over the batch."
       )
   ```
2. Backstop guard in `SolverState.from_problem` (`optim/state.py`, after
   `r0 = problem.residual(x0)`): `if r0.dim() > 1: raise NotImplementedError(
   "batched residuals are not supported; see roadmap M2b")` — so the
   dishonest matmul at `:95` can never be reached by any optimizer path, not
   just `solve_ik`. Keep it terse; the user-facing message lives in
   `solve_ik`.
3. Docs: in `docs/concepts/tasks.md` replace the "Warm-started batched IK"
   section (`:161–173`) with an honest note that batched solving is a
   roadmap item (M2b), or remove the example. In root `CLAUDE.md` the
   "Batching Rules" section describes FK/residual/Jacobian batching (which
   *is* real) — leave that, but do not let it imply the *solver* is batched;
   add "solvers are single-problem in v0; see roadmap M2b". (T0.9 sweeps for
   any missed batched-solve claims.)

**What to test:**
- New test in `tests/tasks/` (e.g. `test_ik_regression.py`): assert
  `pytest.raises(NotImplementedError, match="M2b|batch")` for
  `solve_ik(model, targets, initial_q=q0.expand(4, -1))`.
- Unbatched `solve_ik` must stay green (existing IK regression tests).

**Pitfalls / do-not-forget:**
- `solve_trajopt` flattens a `(T, nq)` trajectory into a 1-D vector, so the
  `state.py` backstop must not break it — verify `problem.residual` there is
  1-D (it is; trajopt is unaffected). Run `tests/tasks/test_trajopt_param.py`.
- The consumer repos do **not** call `solve_ik` batched (they hand-roll
  their own loops) — no consumer breakage, but grep both repos to confirm
  (`grep -rn "solve_ik" ../BetterVideoReconstruction ../BetterHumanForce`).

---

### T0.5 — Bounded-LM honesty + correct terminal status  [S]

**Goal / done-when:** the `LevenbergMarquardt` docstring and `docs/` state
the active-bounds weakness using the **corrected** mechanism; no false
`converged`; the terminal status distinguishes what the code actually does.

**Current state (mechanism corrected by codex — the old story was wrong):**
The old explanation ("the step is clamped *after* acceptance, wrecking the
ratio test") is **false**. Read `levenberg_marquardt.py:99–117`:
- The trial point **is projected onto the bounds before the trial residual
  is evaluated** — `x_new = problem.step(...)` then `x_new =
  x_new.clamp(min=lower, max=upper)` (`:103–107`), then `r_new =
  problem.residual(x_new)` (`:109`).
- Acceptance is a bare `cost_new < cost` comparison (`:114`) on raw L2 cost.
- The gain ratio is computed (`:116–117`) but the default `Adaptive`
  strategy ignores it — it just halves λ on accept, doubles on reject
  (`strategies/adaptive.py:19–25`).
- There is **no active set, no projected gradient, no KKT test** anywhere.
So at active bounds the projected step direction is repeatedly poor,
progress stalls, and the loop exits `maxiter` with real error left. (codex
probe: 0.259 m error, 3 joints pinned, status `maxiter` — not a false
`converged`.)

**`stalled` status — verify and correct the plan claim:** the assessment
§1.4 and the task brief both say "`stalled` exists in the enum and is never
set". **Code says otherwise:** `LBFGS` *does* set `stalled`
(`optim/optimizers/lbfgs.py:144`). It is **LM and GN specifically** that
never set it — they only ever set `converged` or `maxiter`. Write the honest
version: *the LM path cannot report a distinct "stalled at active bounds"
status; a bounds-limited run exits as `maxiter`.* (Record this as a
plan-vs-code discrepancy.)

**Implementation plan:**
1. Rewrite the `LevenbergMarquardt` module docstring
   (`levenberg_marquardt.py:1–17`) and the `docs/concepts/solver_stack.md`
   bounded-LM paragraph to state, precisely: box bounds are enforced by
   projecting the trial point onto `[lower, upper]` **before** evaluating
   it; acceptance is a bare cost comparison; there is no active-set /
   projected-gradient / KKT treatment; therefore at active bounds the solver
   can stall and exit `maxiter` with residual error remaining. Point at the
   real fix in **M2b** (active-set LM or reflective trust region with KKT
   termination).
2. Do **not** invent a fake `stalled` for LM by adding a "λ hit its ceiling"
   heuristic that relabels the same bad steps — codex explicitly warns a
   projected-gradient stop or a status patch "keeps the same bad steps under
   a better label". The honest M0 change is documentation + keeping the
   truthful `maxiter`. If you want a *correct* stall signal, it must be a
   real KKT/projected-gradient test — that is M2b scope, not M0.
3. Optional, low-risk: in `SolverState`'s docstring (`state.py:49–51`) note
   that `stalled` is currently set only by `LBFGS`.

**What to test:**
- A regression test in `tests/tasks/` that documents the *current* honest
  behavior: an in-bounds-solution bounded IK that today stalls must return
  `converged is False` and `status == "maxiter"` (not a false `converged`).
  Use the audit's probe: target from FK of an in-bounds `q =
  0.7·lower + 0.3·upper`, start at the clamped neutral, default config. This
  test *documents the weakness*; M2b will flip it to `converged is True`.
  Mark it clearly (`# documents M0 bounded-LM weakness; M2b makes it pass`).
- Existing unbounded-LM convergence tests stay green (17–25 iters to ~1e-14).

**Pitfalls / do-not-forget:**
- Do not "fix" the bounded solve here — the repair is M2b
  (`m2b_batched_second_order_solvers.md`). M0 only makes the docs honest and
  the status truthful.
- `optim/CLAUDE.md` "LM Details" says "After each accepted step: clamps
  `x_new` to `[lower, upper]`" — that is *accurate* to the code (the clamp
  is at `:103–107`, before the residual eval, on every trial not just
  accepted). Tighten the wording to "clamps every trial point before
  evaluation" so it doesn't reinforce the old post-acceptance story.

---

### T0.6 — Robust-kernel IRLS: accept on ρ, not raw L2  [S]

**Goal / done-when:** a Huber solve **converges** on an outlier dataset
where accepting on raw L2 stalls; step acceptance and the gain ratio use the
robustified cost ρ, consistent with the IRLS reweighting.

**Current state (verified):** `_apply_kernel`
(`levenberg_marquardt.py:27–44`) reweights residual/Jacobian rows by
`sqrt(kernel.weight(r²))` for the normal equations — correct IRLS. But
acceptance compares **raw L2**: `cost = float(state.residual_norm)` (`:86`,
which is `0.5‖r‖²`) and `cost_new = float(0.5 * (r_new @ r_new).sum())`
(`:110`). With a non-L2 kernel, LM *minimizes* the robust objective but
*accepts/rejects* against the plain quadratic. Every kernel implements
`rho(r²)` precisely for this, and `rho` has **zero callers** (grep). Result:
good robust steps get rejected near outliers; Tukey (weight→0) can accept
steps that *increase* the robust cost.

**Implementation plan:**
1. Define the robust cost as `Σ ρ(r_i²)` (via `kernel.rho`) with L2's
   `ρ(s)=0.5·s` fallback when `kernel is None`. Add a small helper:
   ```python
   def _robust_cost(r, kernel):
       if kernel is None:
           return 0.5 * (r * r).sum()
       return kernel.rho(r * r).sum()
   ```
2. Use `_robust_cost` for both `cost` (initial, replacing the `:86`
   `residual_norm` read) and `cost_new` (`:110`), and keep the accept test
   `cost_new < cost`. The gain ratio's *predicted* decrease
   (`:116`) is computed from the reweighted normal equations — for M0 it is
   sufficient that both actual and predicted are on the robust objective;
   the exact Madsen predicted-reduction on the robust surrogate is M2b
   polish. Keep the raw `residual`/`residual_norm` fields storing raw
   quantities (other code reads them) — introduce a separate `cost` scalar
   for the acceptance test rather than overloading `residual_norm`.
3. Confirm `rho` semantics across kernels (`kernels/huber.py`,
   `cauchy.py`, `tukey.py`): `weight(s)` must equal `ρ'(s)` for the IRLS
   equivalence to hold. If any kernel's `rho`/`weight` pair is inconsistent,
   flag it (do not silently "fix" a kernel — report the mismatch).

**What to test:**
- New test in `tests/optim/` (e.g. `test_robust_kernels.py`): construct a
  1-D-ish least-squares problem with a few gross outliers where an L2-accept
  LM stalls or converges to the outlier-biased solution, and a Huber-accept
  LM reaches the inlier solution. Assert the Huber run's final robust cost
  and inlier fit beat the L2-accept baseline. Keep it a small synthetic
  residual (does not need FK) so it is fast and deterministic (seed the RNG).
- Add a unit test that `kernel.rho` is now actually called during a solve
  (e.g. monkeypatch/spy, or assert a Tukey solve does not accept a
  robust-cost-increasing step).
- Existing L2 IK regression tests must be unchanged (L2's ρ = 0.5·s
  reproduces today's behavior exactly).

**Pitfalls / do-not-forget:**
- `residual_norm` is documented as `0.5‖r‖²` and read by callers
  (`IKResult`, tests) — do not change its meaning. Add a distinct robust
  `cost` local/scalar.
- Rows arriving at the kernel are already scaled by `CostItem.weight`
  (audit §2.6), so a user's Huber `delta` is in "weighted units" — out of
  scope to fix here, but do not make it worse; note it.
- This shares the file with T0.5 — coordinate edits.

---

### T0.7 — `solve_ik` dtype preservation  [S]

**Goal / done-when:** a float64 model + float64 `initial_q` yields a float64
solve and a float64 `result.q`; a dtype-preservation test passes.

**Current state (verified):** `tasks/ik.py` hard-casts to float32:
`x0 = initial_q.clone().detach().float()` (`:186`), `model.q_neutral.clone().float()`
(`:188`), and `lower=model.lower_pos_limit.float()` / `upper=...float()`
(`:221–222`). Pass float64 in, get a float32 solve and float32 result out.
No other task does this (`solve_trajopt` preserves dtype).

**Implementation plan:**
1. Drop the four `.float()` calls. Derive the working dtype from the inputs:
   - `x0 = initial_q.clone().detach()` (or `model.q_neutral.clone()` when
     `initial_q is None`), preserving its dtype.
   - `lower = model.lower_pos_limit.to(x0.dtype)`,
     `upper = model.upper_pos_limit.to(x0.dtype)` (a no-op when they already
     match; avoids a needless cast, and never *downcasts*).
2. Guard the mixed case: if `initial_q.dtype != model.<tensors>.dtype`,
   either cast the *initial_q* to the model dtype or raise a clear
   `DtypeMismatchError` (`exceptions.py:55`, currently never raised — wiring
   it here is a bonus honesty win). Prefer: solve in `initial_q`'s dtype and
   `.to()` the model-derived limits to match; document that the solve dtype
   follows `initial_q` (or `q_neutral`).

**What to test:**
- New test in `tests/tasks/`: `model64 = model.to(dtype=torch.float64)`;
  `res = solve_ik(model64, targets, initial_q=q0_f64)`; assert
  `res.q.dtype == torch.float64` and `res.fk().joint_pose_world.dtype ==
  torch.float64`. Also assert float32 in → float32 out (default path
  unchanged).
- Existing IK regression tests (float32) stay green.

**Pitfalls / do-not-forget:**
- The library's declared dtype policy is a later item (03 §9); M0 only makes
  `solve_ik` *preserve* input dtype, matching `solve_trajopt`. Do not
  attempt a global dtype-policy overhaul.
- Shares `tasks/ik.py` with T0.4 and T0.8 — coordinate line edits.

---

### T0.8 — Delete the enum lies + document FD cost  [S]

**Goal / done-when:** every advertised option either works or cannot be
selected. Specifically: `JacobianStrategy.AUTODIFF`/`FUNCTIONAL` are
removed (they silently ran FD); `OptimizerConfig`'s `"cg"` linear-solver and
`"trust_region"` damping options are removed (they crash when selected); the
FD fallback is documented as costing **2·nv + 1** residual evaluations.

**Current state (verified):**
- `kinematics/jacobian_strategy.py:22–23` defines `AUTODIFF = "autodiff"`
  and `FUNCTIONAL = "functional"` with docstrings (`:15–16`) claiming
  `torch.func.jacrev`/`jacfwd`. `residual_jacobian`
  (`kinematics/jacobian.py:246`) only checks `ANALYTIC`/`AUTO`, then
  **unconditionally** central-FD (`:259–284`) — `jacrev`/`jacfwd` appear
  nowhere. So both enum values are unreachable lies (verified: all
  strategies give bit-identical FD Jacobians).
- FD cost: `r0 = _fn(v0)` (1 eval) + a Python loop over `nv` doing `_fn(v_p)`
  and `_fn(v_m)` (2·nv evals) = **2·nv + 1** full FK-with-frames passes
  (`jacobian.py:274,280–283`). codex counted 19 at nv=9. It is also unbatched
  (`dim = r0.numel()` flattens batch into the residual dim; crashes on
  batched states).
- `tasks/ik.py:65` `linear_solver: Literal["cholesky", "lstsq", "cg"]` and
  `:67` `damping: Literal["constant", "adaptive", "trust_region"]`; the
  factory tables (`:108`, `:142`) build `CG`/`TrustRegion`, whose methods
  raise `NotImplementedError` mid-solve.

**Implementation plan:**
1. `jacobian_strategy.py`: delete `AUTODIFF` and `FUNCTIONAL` enum members
   and their false docstring lines. Keep `ANALYTIC`, `FINITE_DIFF`, `AUTO`.
   Add a one-line note that AUTO = analytic-then-FD today, and that a real
   `torch.func.jacrev` fallback is a roadmap item (residual redesign, M2 /
   03 §6.2) — "re-add these values when implemented".
2. `residual_jacobian`: no logic change needed (it already ignored the two
   values), but update its docstring/comments to state the AUTO fallback is
   central FD at **2·nv + 1** evaluations, eps `1e-3` (fp32) / `1e-7`
   (fp64), unbatched. Do **not** implement jacrev here — that is M2.
3. `tasks/ik.py`: narrow the Literals to `Literal["cholesky", "lstsq"]` and
   `Literal["constant", "adaptive"]`. Remove `"cg"` from the
   `_make_linear_solver` table (`:108`) and `"trust_region"` from
   `_make_damping_strategy` (`:142`) so an unknown name raises the existing
   `ValueError` at config time instead of `NotImplementedError` mid-solve.
   Do **not** delete the `CG`/`SparseCholesky`/`TrustRegion` classes
   themselves (they are the future seam; leave them stubbed and off the
   selectable surface — "re-add when implemented" per the standing rule).
4. Docs/CLAUDE.md: `kinematics/CLAUDE.md` "Jacobian Strategy" block
   reproduces the false enum (with `AUTODIFF = torch.func.jacrev`) — update
   it. `optim/CLAUDE.md` "Pluggable Components" lists `CG`, `SparseCholesky`,
   `TrustRegion` as if usable — mark them "stubbed, not selectable". The
   `docs/index.md:49–51` "what ships today" prose advertises "CG, sparse
   Cholesky" solvers and "TrustRegion" damping — remove them (T0.9 also
   sweeps this; whichever task lands first, coordinate).

**What to test:**
- Update any test that references `JacobianStrategy.AUTODIFF`/`.FUNCTIONAL`
  (grep `tests/` first). Add a test asserting the enum no longer has those
  members (`assert not hasattr(JacobianStrategy, "AUTODIFF")`).
- Add a config test: `OptimizerConfig(linear_solver="cg")` and
  `damping="trust_region")` must now be rejected — either a `ValueError`
  from the factory when solve_ik runs, or (better) a type-check the Literal
  enforces. Assert the crash is *not* a mid-solve `NotImplementedError`.
- A test documenting FD cost: instrument `_fn` call count (monkeypatch
  `forward_kinematics` or count via a spy) and assert exactly `2·nv + 1`
  calls for one `residual_jacobian` on a no-analytic-Jacobian residual.
- Contract tests (`tests/contract/`) stay green; `tests/kinematics/` stays
  green.

**Pitfalls / do-not-forget:**
- Grep the consumer repos for `AUTODIFF`/`FUNCTIONAL`/`linear_solver="cg"`/
  `damping="trust_region"` before deleting (per the standing deletion rule).
  None expected, but confirm.
- The frozen-API contract (`tests/contract/test_public_api.py`) does not
  include these enum values in its 26-symbol set — but check
  `test_pluggable_protocols.py` doesn't assert `CG`/`TrustRegion` are
  *reachable*.

---

### T0.9 — Docs truth pass  [S]

**Goal / done-when:** the front-page snippet **executes**; frame naming,
`lower_pos_limit` shape, the roadmap stub-inventory page, and the per-package
`CLAUDE.md` claims are corrected/regenerated. Also fold in the
accepted-but-ignored `solve_ik` options (assessment §1.6).

**Current state (verified live):**
- **Front-page snippet is broken.** `docs/index.md:36,39` uses
  `solve_ik(model, {"panda_hand": ...})` and
  `result.frame_pose("panda_hand")` — frames are named `body_panda_hand`
  (verified: `"panda_hand" not in model.frame_names`); both raise `KeyError`.
  Same `"panda_hand"` bug in repo-root `CLAUDE.md:97,100`. (`README.md`
  already uses `body_panda_hand` at `:34,38` — partially fixed.)
- **`lower_pos_limit` shape.** Root `CLAUDE.md:138–139` documents `(nv,)`;
  actual is `(nq,)` (verified; `model.py:65–66` already says `(nq,)`). They
  differ only for nq≠nv joints (free-flyer/spherical) — on free-flyer Panda
  nq=16, nv=15, limits are `(16,)`.
- **`data.J`.** Root `CLAUDE.md:91` says `compute_joint_jacobians … fills
  data.J` — `J` is a *deprecated alias*; the field is `data.joint_jacobians`
  (the naming contract bans `J` from source).
- **Roadmap page lies.** `docs/reference/roadmap.md:8` says "If a symbol is
  not on this page, it is implemented and tested" — false for ~10 stub
  symbols absent from the page (`RobotCollision` methods, `collision/pairs`,
  `TrustRegion`, `SparseCholesky`, `costs/factory.factory`,
  `kinematics/chain.get_chain`, `data_model/indexing.build_name_to_id`,
  `utils/*` stubs, `spatial/force`).
- **Ignored `solve_ik` options.** `IKCostConfig.collision_margin` /
  `collision_weight` (`ik.py:46–47`) and the `robot_collision=` param
  (`ik.py:157`, docstring "unused in this version") are accepted and never
  read.

**Implementation plan:**
1. **Front-page snippet executes.** Fix `docs/index.md:36,39` and root
   `CLAUDE.md:97,100` to `body_panda_hand`. Then make it *mechanically*
   true: add an executed-snippet test (a small `tests/docs/` pytest that
   imports the exact code from `docs/index.md`, or a myst-nb/doctest run) so
   the front-page IK example runs against the real Panda and returns without
   `KeyError`. This is the roadmap's headline done-when for M0 docs.
2. **`lower_pos_limit` shape.** Fix root `CLAUDE.md:138–139` to `(nq,)`
   (both lower and upper). Grep docs for other `(nv,)` limit claims.
3. **`data.J`.** Fix root `CLAUDE.md:91` to `data.joint_jacobians`.
4. **Roadmap page regenerated from a grep.** Run
   `grep -rn "raise NotImplementedError" src/better_robot/` to produce the
   canonical stub list, and either (preferred) add a small generator/test
   under `tests/` that regenerates `docs/reference/roadmap.md`'s inventory
   from that grep and asserts the page matches, or at minimum add every
   missing stub symbol to the page so the `:8` completeness rule becomes
   true. Note that T0.3's mimic `NotImplementedError` and any others land in
   this same inventory.
5. **Ignored options.** Either remove `IKCostConfig.collision_margin`/
   `collision_weight` and the `robot_collision=` param from `solve_ik`
   (they read nothing; collision is 100% stub — assessment §1.6), or, if a
   consumer passes them (grep both repos first), keep the param but document
   "accepted and ignored until collision lands (M4)". Prefer removal —
   nothing reads them.
6. **CLAUDE.md sweep (do last if parallelized).** After T0.1/T0.3/T0.4/T0.5/
   T0.8 land, re-grep all `CLAUDE.md` and `docs/` for: `panda_hand`,
   `data\.J\b`, `(nv,)` limit claims, "batched" solve claims, `AUTODIFF`/
   `FUNCTIONAL`, `trust_region`/`\bCG\b`/"sparse Cholesky", "smooth and
   differentiable across the singularity", the mimic "nq=0, nv=0" claim.
   Fix any survivors. `docs/index.md:25–27` also calls the `Backend`
   Protocol "the architectural core" / "a backend that does not leak" — that
   is M1's deletion target, not M0; leave it, but do not add new
   backend-centric claims.

**What to test:**
- The executed front-page snippet test (item 1) — the load-bearing
  acceptance.
- If you add a roadmap-regeneration test, it asserts the `docs/reference/
  roadmap.md` inventory equals the live `NotImplementedError` grep.
- Existing doc/contract tests stay green; `tests/contract/test_naming.py`
  (bans `data.J` in *source*) is unaffected by doc edits.

**Pitfalls / do-not-forget:**
- `sphinx-docs` skill (MyST/Markdown) applies to `docs/` edits;
  `diataxis-docs` if you restructure a page (you shouldn't need to).
- Do not "fix" the `body_` prefix by renaming frames in the builder — that
  is a breaking API change and an open owner question (audit Q2). M0 fixes
  the *docs* to match the code, not vice versa.
- Coordinate with T0.4 (batched claims) and T0.8 (enum/solver prose) so the
  same lines aren't edited twice.

---

### T0.10 — Remove free-flyer `bool()` quaternion check from the FK hot path  [S]

**Goal / done-when:** free-flyer `forward_kinematics_raw` compiles with
`torch.compile(fullgraph=True)`; the per-call host-sync quaternion-norm
validation becomes an opt-in debug check.

**Current state (verified):** `_validate_q` (`kinematics/forward.py:35`)
runs `if bool(((norm - 1.0).abs() > _QUAT_NORM_TOL).any())` (`:64`) for
free-flyer models — a forced GPU→host sync and a `torch.compile(fullgraph=
True)` graph breaker (fixed-base FK compiles at ~5× today; free-flyer fails
on exactly this line). Worse, `_validate_q` runs **twice per FK**: once in
`forward_kinematics_raw` (`:97`) and again in `forward_kinematics` (`:181`).

**Implementation plan:**
1. Make the quaternion-norm check opt-in. Options (pick the simplest that
   satisfies the compile test; prefer a module-level flag over an env read
   in the hot path):
   - Gate the `bool(...)` block behind a module-level debug flag (e.g.
     `better_robot.set_debug_checks(True)` / a `_DEBUG_CHECKS` module global),
     default **off**. The shape and device checks (`:50–59`) are cheap and
     sync-free — keep those always on.
   - Or split `_validate_q` into `_validate_q_shape_device` (always on,
     sync-free) and `_validate_q_quat_norm` (debug-only).
2. Remove the **double** validation: `forward_kinematics_raw` (`:97`) is the
   tensor-only primitive the backend calls and is the compile target — it
   should *not* run the norm check on the hot path. `forward_kinematics`
   (`:181`, the public boundary) is the right place for an opt-in check.
   Ensure the quaternion check runs at most once, at the public boundary,
   and only when debug is enabled.
3. Keep the value-correctness contract intact: document that free-flyer q is
   assumed pre-normalized on the hot path, and the debug check exists to
   catch mistakes.

**What to test:**
- New test in `tests/kinematics/`: load a free-flyer model
  (`br.load(g1.urdf, free_flyer=True)` or the free-flyer Panda used
  elsewhere) and assert
  `torch.compile(forward_kinematics_raw, fullgraph=True)(model, q)` runs
  without a graph break (and matches eager output). This is the done-when.
  Guard with `pytest.importorskip` on torch.compile availability; run on CPU
  (CUDA is broken on this box).
- A test that the debug check still *catches* a denormalized free-flyer
  quaternion when enabled (raises `QuaternionNormError`), and does **not**
  raise / sync when disabled (default).
- Existing FK/pinocchio-parity tests stay green — the shape/device
  validation must still fire.

**Pitfalls / do-not-forget:**
- `write-tests` skill: the compile test is a legitimate CPU test; state that
  it needs a torch version supporting `fullgraph=True` (repo floor is
  currently `>=2.1`; the compile test may need a skip on older versions).
- Do not remove the shape/device checks — only the `bool(...).any()` norm
  sync becomes opt-in.
- This is item 7-step-1 of 03 §7 ("fix the syncs and graph breaks") scoped
  to just the FK breaker; the optimizer `float()`/`bool()` syncs and the
  hot-path-lint extension are **M1** (`m1_two_lane_seam_and_hygiene.md`),
  not M0. Do not widen the lint here.

---

## Milestone acceptance checklist

- [ ] `so3/se3 exp/log`, `tangents.py` right-Jacobians, and
  `JointSpherical.difference` pass `gradcheck` **and** `gradgradcheck` at
  θ=0 and 1e-9 (fp64); `so3.exp(0)` backward is finite (T0.1).
- [ ] SMPL-like rest-residual backward has zero NaNs (T0.1).
- [ ] `lie/CLAUDE.md` and the backend module docstring no longer claim
  "smooth and differentiable across the singularity" (T0.1).
- [ ] fp32 gradcheck near the (now dtype-dependent) Taylor cutoff passes;
  fp64 behavior unchanged; the `tangents.py` `1e-10` cutoffs are dtype-aware
  too (T0.2).
- [ ] Loading a mimic URDF raises `NotImplementedError` with roadmap
  guidance; `data_model/CLAUDE.md`'s mimic claim is removed; the Panda-load
  fallout decision (reject vs identity-mimic-scope) is owner-reviewed (T0.3).
- [ ] Batched `solve_ik` raises an honest `NotImplementedError` (points to
  M2b), not the `state.py:95` matmul crash; docs stop claiming batched
  solving (T0.4).
- [ ] LM docstring + `docs/` describe the *corrected* bounded-LM mechanism
  (projected-before-eval, bare cost compare, no active set/KKT); no false
  `converged`; the LM/GN-never-set-`stalled` reality is documented (T0.5).
- [ ] A Huber solve converges on an outlier dataset where L2-accept stalls;
  acceptance uses ρ; `kernel.rho` is actually called (T0.6).
- [ ] float64 `solve_ik` returns float64; dtype-preservation test passes
  (T0.7).
- [ ] `JacobianStrategy.AUTODIFF`/`FUNCTIONAL` deleted; `"cg"`/
  `"trust_region"` no longer selectable; FD documented as 2·nv+1 evals
  (T0.8).
- [ ] The front-page snippet executes in a test; frame naming
  (`body_panda_hand`), `lower_pos_limit` shape `(nq,)`, `data.J`→
  `joint_jacobians`, and the roadmap stub inventory are corrected; ignored
  `solve_ik` collision options removed/documented (T0.9).
- [ ] Free-flyer `forward_kinematics_raw` compiles with `fullgraph=True`;
  the quaternion-norm sync is opt-in debug (T0.10).
- [ ] `uv run pytest tests/ -v` is green (except the deliberately-added
  "documents-the-weakness" T0.5 test, which is marked as such), and the
  pinocchio-parity suite is untouched at `atol=2e-6`.

## Out of scope

- **Real bounded least-squares** (active-set LM / reflective trust region,
  KKT termination, a genuine `stalled` status): **M2b**
  (`m2b_batched_second_order_solvers.md`). T0.5 only documents the weakness.
- **Batched solving** (per-element damping/cost/accept, batched Cholesky):
  **M2b**. T0.4 only raises an honest error.
- **Real mimic enforcement** (reduced-coordinate map across FK/Jacobians/
  limits/RNEA/ABA/CRBA): **M3** (`m3_parametric_model_breadth.md`). T0.3 only
  rejects at build. Do **not** implement an FK-only gather (FK and dynamics
  would disagree).
- **A real `torch.func.jacrev` autodiff Jacobian fallback**: residual
  redesign in **M2** (03 §6.2). T0.8 only deletes the lying enum values and
  documents FD cost.
- **The other host-sync fixes and the hot-path-lint extension** (optimizer
  `float()`/`bool()`, `torch.eye` per iter, per-call `new_tensor` in
  `residuals/pose.py`, constant hoisting): **M1** step 1/6/7
  (`m1_two_lane_seam_and_hygiene.md`). T0.10 fixes only the FK free-flyer
  breaker.
- **Deleting `backends/`, `utils/`, `costs/`, the `Data` alias shim,
  `ResidualSpec`, the registry, `rich`**: **M1** (with consumer-migration
  sequencing). M0 deletes nothing with a live consumer.
- **Dtype/units policy, quaternion double-cover convention, the
  differentiation contract**: written policies in **M1** (03 §9). T0.7 only
  makes `solve_ik` preserve input dtype.
- **Renaming the `body_` frame prefix**: open owner question (audit Q2);
  breaking change. T0.9 fixes docs to match code, not the reverse.
- **Setting up CI**: **M1** item 10. M0 changes are validated by the local
  suite only (no `.github/` exists yet).

## References

- `plan/04_roadmap.md`, **M0 table** — the ten done-when criteria this file
  expands (one row per task).
- `plan/01_assessment.md` §1.1 (NaN gradients), §1.2 (batched crash), §1.3
  (mimic), §1.4 (bounded-LM, *mechanism corrected*), §1.5 (enum/FD),
  §1.6 (collision + ignored options), §2.4 (perf/syncs), §2.6 (docs drift).
- `plan/03_architecture.md` §6 (residual-library / enum deletion rationale,
  "delete the enum values that lie"), §7 step 1 (fix syncs and graph
  breaks — the FK free-flyer breaker), §4 (optimizer stack: robust ρ
  acceptance, bounded algorithm — the M2b target), §9 (dtype policy — the
  Taylor-cutoff + preserve-dtype notes).
- `plan/research/codex_plan_review.md` — **A1** (NaN blast radius narrowed),
  **A4** (bounded-LM: the corrected "projected-before-eval, no active set"
  mechanism; use this, not the old clamp-after-acceptance story), **A5**
  (FD = 2·nv+1; analytic ~39× FD but only ~9× real autodiff), **A12**
  (robust IRLS accepts on wrong objective), **A3** (mimic: reject-at-build is
  the only honest quick fix), **A11** (float32 cast).
- `plan/research/audit_optim_stack.md` §2.2 (batched crash), §2.3
  (bounded-LM probe: the in-bounds target that stalls — reuse as the T0.5
  test), §2.6 (robust IRLS + `rho` has zero callers — the T0.6 evidence),
  §2.8 (ornamental crashing options), §2.13 (float32 cast + ignored options).
- `plan/research/audit_quality_perf.md` §2.5/§2.6 (the `_validate_q` sync,
  the free-flyer `fullgraph` breaker — T0.10), §2.8 (the 6-of-10 doc-drift
  table: front-page snippet, `lower_pos_limit` shape, roadmap completeness,
  `data.J` — the T0.9 checklist), §2.9 (dead exceptions incl. the unused
  `UnsupportedJointError` T0.3 could wire).
- `references/design/jaxlie.md` — the safe-`where` idiom source (dummy under
  sqrt/division, select after) for T0.1.
- Sibling instruction files: the honest errors drafted here are *repaired*
  in `m2b_batched_second_order_solvers.md` (batched + bounded, T0.4/T0.5),
  `m3_parametric_model_breadth.md` (mimic coordinate map, T0.3), and the
  residual redesign in M2 (real autodiff fallback, T0.8).
- Consumer repos (grep before any deletion, per README standing rule):
  `../BetterVideoReconstruction`, `../BetterHumanForce`.
