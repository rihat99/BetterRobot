# M2b — Batched Second-Order Solvers: Agent Execution Instructions

> Read `plan/for_agents/README.md` first. It carries the standing rules
> (deletion ordering, honesty rules, kernel requirements, test commands).

> **Update (owner decision 2026-07-17 — branch strategy):** BHF/BVR stay
> on the pre-redesign branch until the M4 migration, so the "preserve the
> legacy `minimize` path / kernels for BHF" constraint in this file is
> lifted. The legacy surface may be deleted once no BR-internal caller
> consumes it; record removed symbols for M4's migration table.

## Mission

Rebuild LM/GN as batched, capture-ready solvers on the jaxopt
`init_state/update/run` split, running on M2a's `Problem`/`Values` — not the
old `LeastSquaresProblem`. After this milestone, one call solves B independent
problems with per-element damping, per-element accept/reject, per-element
terminal statuses, Madsen–Nielsen damping numerics on scaled blocks, a
`cholesky_ex` info-mask fallback so one indefinite element cannot fail the
batch, and a real bounded least-squares algorithm with KKT termination. This
is the single feature that justifies "PyTorch-native, GPU-ready" over calling
scipy — today batched solving does not exist at all (`plan/01_assessment.md
§1.2`), and bounds-active LM stalls at `maxiter` with real error left (§1.4).
`update` is written branch-free and sync-free from day one so M6 can capture
it into a CUDA graph without a rewrite.

## Prerequisites

- **M0** (`m0_truth_and_correctness.md`) must be complete. M2b relies on:
  - IRLS acceptance on the robustified cost ρ (M2b's gain ratio builds on it).
  - The `"cg"`/`"trust_region"` values removed from `OptimizerConfig` Literals
    and the factory tables (M2b deletes the stub *classes* behind them).
  - The batched-input `NotImplementedError` guard on `solve_ik`/legacy
    optimizers (M2b provides the repair at the solver layer; the facade-level
    guard is removed in M2c when `solve_ik` is re-based).
  - The documented bounded-LM weakness (M2b is the promised repair; remove
    the caveat wording in the same change that lands the fix).
  - **Verification (as of 2026-07-17 none of this has landed):**
    `grep -n '"cg"\|"trust_region"' src/better_robot/tasks/ik.py` must return
    nothing (today it hits lines 65, 67, 108, 142). If it still hits, M0 is
    not done — stop and report.
- **M2a** (`m2a_variable_blocks_and_slice.md`) must be complete. M2b consumes:
  - `VarSpec` (with `manifold`, state-space `bounds` + feasible retraction,
    per-coordinate `scale`, `mask` with eliminated fixed coordinates),
    `Values`, `Problem` with `residual(values)`, `gradient(values)`,
    `jacobian_blocks(values)` (03 §3 — sketches show direction, the landed
    signatures win).
  - The custom-residual contract's **semantic grouping for robust kernels**
    (03 §3 constraint 3) — T2b.4's robustified gain ratio needs to know what
    unit a kernel applies to.
  - The tangent-space autograd helper (needs M0's θ=0 fix) for AUTO Jacobians.
  - **Verification:** the M2a vertical-slice test passes; the symbols above
    import; `VarSpec` has a `scale` field (T2b.4 is blocked without it).
- M1 is only an indirect prerequisite (M2a requires it). M2b touches the
  optim layer only; it does not depend on Warp kernels or the bridge.

## Sizing & parallelism

Roadmap effort: **M–L** overall.

| Task | Effort | Depends on |
|------|--------|------------|
| T2b.1 solver pattern (`init_state/update/run`) | M | M2a |
| T2b.2 batched LM/GN semantics | M | T2b.1, T2b.6 |
| T2b.3 `cholesky_ex` info-mask fallback | S | T2b.2 (lands inside its solve step) |
| T2b.4 Madsen–Nielsen on scaled blocks | S–M | T2b.2 |
| T2b.5 bounded algorithm (survey → **gate** → implement) | M–L | survey: none; implementation: T2b.2 + T2b.4 + owner sign-off |
| T2b.6 linear-solver contract + stub deletion | S | M0 |
| T2b.7 capture-readiness enforcement | S–M | continuous; final check after T2b.2–T2b.5 |
| T2b.8 solver-quality probe suite | M | tests written early (xfail), pass by milestone end |

Parallelizable from day one: T2b.6, the T2b.5 *survey*, and T2b.8's test
skeletons. Everything else is ordered as listed. T2b.7 is a discipline applied
throughout plus a final enforcement pass.

## Tasks

### T2b.1 — The `init_state/update/run` solver pattern  [M]

**Goal / done-when:** Each second-order solver is a frozen dataclass of
hyperparameters plus two pure functions and a derived loop:

```python
state = solver.init_state(values, problem)              # -> LMState
values, state = solver.update(values, state, problem)   # one step, pure
values, state = solver.run(values, problem)             # derived loop
```

State is a plain value (a `NamedTuple` or frozen dataclass whose fields are
all tensors — no Python floats, no lists of dicts). Consumers can own the
outer loop (call `update` themselves, interleave logging/viewer updates) and
warm-start by passing a previous `(values, state)` pair back in. Done when
this contract is tested: a hand-rolled external loop over `update` produces
the same result as `run`, and a warm-started second solve reuses damping
state.

The state-shape and graph-lifetime requirements in
`docs/conventions/engineering.md` § “Differentiation contract” are binding
on this task; implicit backward remains an M6 deliverable.

**Current state:** All four optimizers are `minimize(problem, *, max_iter,
linear_solver, kernel, strategy, scheduler)` monoliths returning a mutable
`SolverState` (`src/better_robot/optim/optimizers/base.py:26-44`).
`SolverState.damping` is a Python `float`, `history` is a list of dicts with
Python floats (`src/better_robot/optim/state.py:63-70`), and
`SolverState.from_problem` crashes on batched input at `state.py:95`
(`0.5 * (r0 @ r0)` — 1-D matmul). There is no step-level API; consumers
cannot embed a step (BVR/BHF hand-roll `torch.optim` loops for exactly this
reason — see `references/design/jaxopt.md §11.1`).

**Implementation plan:**
1. Create the new solver module(s) **beside M2a's `Problem`** — follow
   whatever package layout M2a landed (coordinate via
   `m2a_variable_blocks_and_slice.md`; if it reorganized `optim/`, follow
   that; use the category-first file-naming convention, e.g.
   `solver_lm.py`). Do NOT modify
   `src/better_robot/optim/optimizers/*.py` — the legacy `minimize` path has
   live consumer imports (BHF `tools/geometry/icp.py:58,330-332` calls
   `GaussNewton(...).minimize(...)`; `tools/object_align/sdf_fit.py:40-41`
   imports the kernels) and is retired in M2c/M4, not here (standing rule 1).
2. Define the solver as `@dataclass(frozen=True)`: `max_iter`, `tol`,
   `damping_parameter`, `linear_solver`, `kernel`, plus the T2b.5 bound
   options. Hyperparameters only — everything mutable per-iteration lives in
   the state (jaxopt discipline, `references/design/jaxopt.md §2, §9`).
3. Define `LMState` as a NamedTuple/frozen dataclass of tensors, all shaped
   with the batch: `iter_num ()` or `(B,)`, `cost (B,)`, `residual (B, dim)`,
   `mu (B,)`, `increase_factor (B,)`, `grad_norm (B,)`, `converged (B,)
   bool`, `status (B,) int8`. Fields are tensors from birth
   (`torch.zeros(B, dtype=...)`, not `0.0`) so the pytree structure never
   changes between iterations.
4. `init_state` evaluates the residual once at the initial values (so
   `update` never special-cases iteration 0) and initializes μ per T2b.4.
5. `run` is the only place allowed to touch the host: a bounded Python `for`
   loop that may call `bool(state.converged.all())` **once per iteration**
   for early exit. Keep the loop driver separable from `update` — M6 swaps in
   a fixed-trip captured driver (jaxopt digest §3 and lesson 9: two loop
   drivers, one `update`).
6. Per-iteration history: do not port `SolverState.history` (per-iteration
   dicts of Python floats = a sync per field). If diagnostics are wanted,
   `run` may optionally stack per-iteration tensors at the end; default off.
7. Warm start: `run(values, problem, state=None)` — if `state` is given, skip
   `init_state`.

**What to test** (new file, e.g. `tests/optim/test_solver_pattern.py`):
- External-loop equivalence: N calls of `update` == `run` with
  `max_iter=N` (identical `values` bitwise — same code path).
- Warm start: solve, perturb targets slightly, re-run with the returned
  state; asserts it converges in fewer iterations than a cold start.
- State purity: calling `update` twice returns states with identical pytree
  structure (same fields, shapes, dtypes, devices); the input state object's
  tensors are not mutated in place.
- Hyperparameter frozenness: `dataclasses.FrozenInstanceError` on attribute
  assignment.
- Existing tests that must stay green: everything under `tests/optim/`,
  `tests/tasks/`, `tests/contract/` — the legacy path is untouched.

**Pitfalls / do-not-forget:**
- No message from a plan doc overrides the landed M2a signatures — read the
  actual M2a code before writing `update`'s signature.
- `tests/contract/test_solver_state.py` asserts the *legacy* contract; do not
  weaken it — the new state type gets its own tests.
- The `scheduler=` kwarg of the legacy API is dead weight (audit §2.7); do
  not reproduce it in the new API.

### T2b.2 — Batched LM/GN semantics  [M]

**Goal / done-when:** The LM `update` is correct for `(B, ...)`-batched
`Values`: per-element damping `mu (B,)`, per-element cost `(B,)`, B
independent normal systems, per-element accept/reject as a 0/1 `torch.where`
blend of already-computed tensors, per-element convergence mask, zero host
syncs inside `update`. GN is a **preset of the same update** (fixed small μ),
not a second code path — jaxopt's separate GN is a toy that diverges
(`references/design/jaxopt.md §10.5`, lesson 13; audit §2.12 measured 1.21 m
error from GN's unconditional accept).

**Current state:** `optim/optimizers/levenberg_marquardt.py` is scalar
throughout: `cost = float(state.residual_norm)` (line 86), fresh
`torch.eye(nv)` every iteration (line 96), one global accept branch
`if cost_new < cost:` (line 114), convergence checked only inside the accept
branch (line 126) so a run of rejections never terminates early, 4
`float()` host syncs per iteration (lines 86, 110, 116, 126). GN
(`gauss_newton.py:56-71`) accepts unconditionally. Batched input dies at
`state.py:95` before any of this runs.

**Implementation plan:**
1. Assemble the dense system from M2a: flatten `Problem.jacobian_blocks`
   into `J (B, dim, nt)` and residual into `r (B, dim)` where `nt` = total
   tangent dim after mask elimination. Dense assembly is correct for IK-sized
   problems (03 §3); sparsity is M5's job.
2. Normal system per element: `JtJ (B, nt, nt)`, `g = Jᵀr (B, nt)`,
   `H = JtJ + mu[..., None, None] * I` with `I` allocated **once** in
   `init_state` (or hoisted to a module-level per-(nt,dtype,device) constant)
   — never per iteration. Sharing one factorization across elements would
   simply be wrong; `torch.linalg` batches Cholesky natively.
3. Solve for `delta (B, nt)` via the T2b.6 contract; retract:
   `values_cand = retract(values, delta)` through the M2a manifolds; apply
   the feasible projection (T2b.5) so candidates are always in-bounds.
4. **Evaluate ONE candidate.** Cache the current residual/cost in the state;
   evaluate `r_cand`, `cost_cand` (robustified, T2b.4). Do **not** evaluate
   the candidate Jacobian — defer it to the top of the next iteration, so
   rejected elements never pay for it (codex correction, 03 §4: "branchless"
   does not mean evaluating two residual branches; `torch.where` blends
   already-computed tensors). The price: next iteration recomputes J at the
   unchanged x of rejected elements inside the batched op — that is the
   accepted riding-along tax.
5. Accept mask `accept (B,) bool` from the gain-ratio test (T2b.4). Blend:
   `x_next = torch.where(accept[..., None], x_cand, x)` per block (mind
   per-block event shapes), same for `residual`, `cost`. Never index-select
   by mask (`x[accept] = ...` is dynamic-shape work and capture-hostile).
6. Per-element convergence: `state.converged (B,) bool`, updated branch-free,
   e.g. `converged | (grad_inf_norm < tol) | (step_norm < xtol)`. Add the
   criteria the audit found missing (§2.12): gradient norm, step size, and
   relative cost decrease — all per element, all evaluated every iteration
   (not only after accepts). Converged elements keep riding along in the
   batched ops but stop moving: freeze via
   `x_next = torch.where(converged[..., None], x, x_next)`. Compaction of
   converged elements is a later optimization, not correctness — do not build
   it.
7. `status (B,) int8` codes, fixed enum: `0=running, 1=converged,
   2=stalled_at_bounds, 3=maxiter, 4=failed`. `update` writes them as tensor
   blends; `run` translates to readable form at exit (a host sync at the run
   boundary is fine).
8. Host-sync policy: `update` **never** syncs — no `float()`, `bool()`,
   `.item()`, `.cpu()`, no Python comparison on tensor values. `run` may sync
   once per iteration for early exit (T2b.1 step 5).
9. GN preset: constructor/classmethod configuring the LM update with constant
   tiny μ and the same gain-ratio accept test (so it can no longer silently
   diverge). Do not write a second update loop.

**What to test** (e.g. `tests/optim/test_solver_lm_batched.py`; float32 per
workspace test rules; follow the `write-tests` skill):
- Shape correctness: B=1, B=128, and a multi-axis `(2, 3)` batch smoke test
  run without error and return `(B...,)`-shaped `cost/converged/status`.
- Per-element independence: build a batch where element 0 has an easy target
  and element 1 a hard one; assert element 0's `converged` flips iterations
  before element 1's, and element 0's solution stops changing after it
  converges (frozen ride-along).
- Accept/reject blending: engineer one element to reject (e.g. huge μ won't
  help — use a poisoned candidate via a residual spike) while another
  accepts in the same iteration; assert the rejected element's x is bitwise
  unchanged and its μ escalated while the accepted element moved.
- The 128-target batched-vs-sequential parity test — the milestone
  acceptance; protocol defined in T2b.8 probe P6.
- No-sync test: see T2b.7.

**Pitfalls / do-not-forget:**
- `(B..., feature)` is the repo-wide convention — support arbitrary leading
  batch axes, not a hard-coded single `B` (broadcasting via `[..., None]`,
  never `.view(B, ...)`).
- NaN discipline: a NaN candidate cost must resolve to reject
  (`gain_ratio > 0` is False for NaN — rely on that, and add a test).
- The M0 batched-`NotImplementedError` guard on the *legacy* optimizers stays
  in place; only the new solvers accept batches. `solve_ik` stays guarded
  until M2c re-bases it (`m2c_first_order_phases_tasks.md`).
- Update `optim/CLAUDE.md` ("LM Details") and root `CLAUDE.md` ("LM Solver
  Notes": "Adaptive damping: starts at 1e-4, doubles on reject, halves on
  accept…") in the same change — after M2b those claims describe only the
  deprecated legacy path and must say so.

### T2b.3 — `cholesky_ex` with info-mask fallback  [S]

**Goal / done-when:** One indefinite/singular element cannot fail the batch,
raise, or poison other elements with NaN — and the fallback is fixed tensor
work (identical op sequence regardless of which elements fail; no dynamic
element selection), so it is capture-safe.

**Current state:** `optim/solvers/cholesky.py:16-20` wraps
`torch.linalg.cholesky` in `try/except Exception` with a whole-matrix
`torch.linalg.lstsq` fallback (dtype-cast fix landed in commit 91c33ad —
`b.to(A.dtype)` / result cast back; verified present). This is per-matrix
Python control flow: on a batch, one bad element throws for everyone, and the
except-path is a host-side branch that can never be captured.

**Implementation plan:**
1. In the new batched Cholesky path use
   `L, info = torch.linalg.cholesky_ex(H)` (default `check_errors=False`).
   `info (B,) int32` is nonzero for failed elements.
2. `ok = (info == 0)`. Solve with the factor as-is, then neutralize failures
   *before* retraction: `delta = torch.where(ok[..., None],
   torch.nan_to_num(delta_chol), zeros)`. A zero step makes the failed
   element's candidate equal its current point, so `cost_cand == cost`, the
   gain-ratio test rejects naturally, and T2b.4 escalates its μ — which is
   exactly the right response to an indefinite `H` (more damping). No lstsq
   rescue inside `update`.
3. Fold `ok` into the accept mask (`accept &= ok`) and into the status logic:
   an element whose factorization still fails with μ at `mu_max` gets
   `status=failed` (branch-free: a where over `(~ok) & (mu >= mu_max)`).
4. Keep the legacy `Cholesky.solve` try/except untouched for the legacy
   `minimize` path (BHF depends on its behavior across "thousands of ICP
   refinement steps" — see the 91c33ad commit message). It is retired with
   the legacy path in M2c/M4.

**What to test** (extend `tests/optim/test_solver_lm_batched.py`):
- Construct a batch of normal systems where one element's `H` is made
  indefinite (e.g. force `mu=0` and a rank-deficient J for that element);
  assert: no exception, other elements converge normally, the bad element's
  μ escalates, and its x never becomes NaN.
- All-elements-fail batch: statuses go to `failed` once μ hits the cap; loop
  exits by maxiter without raising.
- `nan_to_num` guard: with a NaN-producing factor, the blended `delta` is
  finite for every element.

**Pitfalls / do-not-forget:**
- `cholesky_ex` on CUDA defers the info check by design — never call
  `.item()`/`bool()` on `info` inside `update`.
- Don't "improve" the fallback into a per-element lstsq rescue: that is
  dynamic element selection, explicitly ruled out by 03 §2.6. Rank-deficient
  problems that need lstsq should select the LSTSQ solver for the whole run.

### T2b.4 — Madsen–Nielsen numerics on scaled blocks  [S–M]

**Goal / done-when:** Damping follows Madsen & Nielsen (Algorithm 6.18; port
from `references/design/jaxopt.md §5.2`), per element, computed on **scaled**
blocks, with the gain ratio on the **robustified** cost. Done when the update
rules below are implemented branch-free and the probe suite's convergence
tests (T2b.8) show ≤ the iteration counts of the current halve/double scheme
on the unbounded probes (it should be strictly better; if not, report — do
not tune silently).

**Current state:** `optim/strategies/adaptive.py:16-25` halves λ on accept
(floor 1e-10) and doubles on reject (cap 1e8) — the weakest standard scheme;
the gain ratio is computed (`levenberg_marquardt.py:116-117`) and then used
for nothing. Acceptance is raw-L2 `cost_new < cost` (line 114) even under a
robust kernel — M0 fixes the acceptance objective; M2b carries the fix into
the gain ratio. Every kernel already implements `rho()`
(`optim/kernels/huber.py:17-27` etc.); before M0, `rho` had zero callers in
`src/` (only `tests/contract/test_pluggable_protocols.py:45-52` touches it).

**Implementation plan:**
1. **Scaling first (prerequisite).** Apply the per-block `VarSpec.scale`
   from M2a before forming the normal system: `J_s = J · diag(s)`, solve for
   `delta_s`, unscale `delta = s · delta_s`. Damping magnitudes then act on
   `diag(J_sᵀJ_s)`, which is comparable across heterogeneous blocks —
   radians, meters, pixels, newtons must not share one unscaled μ (03 §4).
   Default `scale=None` ⇒ ones; document that mixed-unit problems without
   scales get whatever the dominant block dictates.
2. Init: `mu0 = damping_parameter * max_j diag(J_sᵀJ_s)` per element,
   `(B,)`-shaped (`amax` over the diagonal). Expose `damping_parameter` as a
   frozen hyperparameter; default `1e-4` (Madsen–Nielsen suggest `1e-6` for
   good initial guesses, `1e-3`–`1` otherwise; IK warm starts are usually
   decent — pick `1e-4` and let the probe suite veto).
3. Gain ratio, per element, robustified:
   `rho_gain = (cost - cost_cand) / (0.5 · Σ delta·(mu·delta - g))`
   (Madsen–Nielsen Eq. 6.16; jaxopt digest §5.2). `cost` is the robust
   objective Σ kernel.rho(·) over the M2a semantic groups (actual decrease =
   true robust cost; predicted decrease = from the IRLS-weighted quadratic
   model — the standard, Ceres-consistent pairing; document it).
4. Accept iff `rho_gain > 0` (tensor mask). On accept:
   `mu *= max(1/3, 1 - (2·rho_gain - 1)³)` via
   `torch.maximum`/`torch.clamp`; reset `increase_factor` to 2. On reject:
   `mu = min(mu · increase_factor, mu_max)`;
   `increase_factor = min(2 · increase_factor, if_max)` — geometric
   escalation. Clamp both at `2**32` (jaxopt's explicit float32 overflow
   protection). All of this as `torch.where` blends keyed on the accept mask.
5. Keep the legacy `Adaptive`/`Constant` strategies untouched for the legacy
   path. The new solver does not take a pluggable damping-strategy object —
   Madsen–Nielsen *is* the LM update (a pluggable Protocol for two trivial
   float maps was negative-value, audit §2.8/§Q4).

**What to test** (e.g. `tests/optim/test_solver_lm_damping.py`):
- Unit-test the update rules on hand-fed tensors: accept with
  `rho_gain=0.5` multiplies μ by `max(1/3, 1-0³)=1/3`… (tabulate 3–4 gain
  values incl. `rho_gain>1` and `≈0⁺`); reject doubles then quadruples
  `increase_factor` across consecutive rejects; clamps hold at `2**32`.
- Per-element trajectories: a batch of two problems, one smooth one nasty —
  assert their μ trajectories differ (state, not hyperparameter).
- Scaling: a 2-block toy problem (one block in meters ~1e0, one in
  "millimeter-ish" units ~1e3) converges with scales set and stalls/creeps
  without — the test documents why scale exists.
- Robust gain ratio: a step that decreases Huber cost but increases raw L2
  is accepted; a Tukey step that increases robust cost while decreasing raw
  L2 is rejected (audit §2.6's failure modes, now as regression tests).

**Pitfalls / do-not-forget:**
- `max(diag(JTJ))` must be per element — a batched `amax(dim=-1)`, not a
  global max.
- The denominator `0.5·δᵀ(μδ − g)` is positive by construction when the
  solve succeeded; guard division with a tiny `clamp(min=eps)` rather than a
  Python branch, and make `rho_gain` NaN-safe (T2b.2 pitfall).
- When T2b.5 lands, the *predicted* decrease must use the actually-taken
  projected step (see T2b.5 step 4) — write the gain-ratio code so the step
  it sees is the post-projection one.

### T2b.5 — A real bounded algorithm  [M–L; evidence gate]

**Goal / done-when:** Bounds-active IK converges (or terminates with an
honest KKT-based status) where the M0-documented behavior stalled. The
algorithm provides: feasible retraction (candidates always in-bounds),
active-set tracking, predicted reduction consistent with the projection, KKT
(projected-gradient) termination, and per-element terminal statuses
`converged / stalled_at_bounds / maxiter / failed`. Probes P1–P3 of T2b.8
pass.

**Current state:** `levenberg_marquardt.py:100-110` projects the trial point
onto the box **before** evaluating it (the codex-corrected mechanism — the
old "clamped after acceptance" story in early plan drafts was wrong, see
01 §1.4), acceptance is bare `cost_new < cost`, and there is no active set,
no projected gradient, no KKT test anywhere. Result (three independent
probes, 01 §1.4): unbounded converges in ~25 iters to ~1e-14 cost; bounded
runs to the 300-iter limit with 3.4e-2 cost / 0.31 m error on a feasible
interior target. `status="stalled"` exists in the enum
(`optim/state.py:26`) and is set only by LBFGS (`lbfgs.py:144`) — LM/GN
never set it. Merely adding a projected-gradient *stopping criterion* would
keep the same bad steps under a better label (codex B3) — that is explicitly
not the fix.

**Implementation plan:**
1. **Survey (do this first, in parallel with T2b.1–T2b.2).** Compare, in
   writing, at minimum:
   - **Reflective trust region** (Branch–Coleman–Li 1999; scipy
     `least_squares(method="trf")` is the reference implementation — read
     scipy's `_lsq/trf.py` and its TRF paper notes);
   - **Projected/active-set LM** (Kanzow–Yamashita–Fukushima-style projected
     LM; Ceres' bounds handling; note jaxopt has *no* bounded NLS — its
     bounded solver is LBFGSB, whose projected-gradient/KKT machinery is
     still worth reading for the termination test).
   Evaluate each against: (a) batchability — can the per-iteration logic be
   expressed as fixed-shape tensor work with `torch.where` (no per-element
   Python control flow)?; (b) capture-readiness per 03 §2.6; (c) behavior on
   the T2b.8 probes P1–P3 (prototype on the actual Panda problems, CPU is
   fine); (d) implementation size/risk. Produce a short evidence document
   (numbers + trajectories, not adjectives) with a recommendation.
2. **STOP for owner sign-off** (standing rule 8). Do not silently pick a
   default and continue.
3. Implement the signed-off algorithm inside the T2b.2 update (bounds config
   comes from M2a `VarSpec.bounds` — **state-space**, nq-shaped for
   `RobotConfig`, enforced via the manifold's feasible retraction; SO3/SE3
   blocks have no box, per 03 §3 constraint 1; never conflate with
   trust-region step bounds).
4. Requirements regardless of choice:
   - Candidates are feasible by construction (projection/reflection inside
     the retraction step, before residual evaluation).
   - Active-set state: per-element boolean masks `(B, nt)` of coordinates at
     bounds with outward-pointing gradient; used to restrict/reflect the
     step, not merely to label the result.
   - The gain ratio's predicted decrease is computed from the step actually
     taken (post-projection δ), so acceptance is consistent (feeds T2b.4
     step 3).
   - KKT termination per element: projected-gradient criterion, e.g.
     `‖P_box(x ⊖ g) ⊖ x‖∞ < tol` in the tangent coordinates that carry
     bounds (free coordinates: plain gradient norm). An element that
     satisfies KKT with active bounds and nonzero residual gets
     `status=stalled_at_bounds` — distinct from `converged`
     (gradient ≈ 0 unconstrained) and from `maxiter`.
5. Delete the M0 caveat documentation ("bounded LM degrades at active
   bounds…") from docstrings/docs in the same change, replacing it with the
   new algorithm's honest description (standing rule 2 in reverse: the
   repair must also retire the warning).

**What to test:** probes P1–P3 in T2b.8, plus:
- Unit test the projection/reflection: a step that would exit the box lands
  exactly feasible; an interior step is untouched (bitwise).
- Active-set identification on a hand-built 2-DOF problem with a known
  constrained optimum: the mask matches the analytic active set; status is
  `stalled_at_bounds`; the KKT residual is below tol.
- Mixed batch: element A interior-optimal, element B bound-constrained, in
  one call → statuses `[converged, stalled_at_bounds]`.

**Pitfalls / do-not-forget:**
- Free-flyer blocks: the base pose coordinates have no bounds; masks must
  align with the tangent layout M2a defines for `RobotConfig` (nq ≠ nv).
- Reflective methods assume strict interior starts; the Panda's `q_neutral`
  violates joint 4's limits (`[-3.07, -0.07]`, root `CLAUDE.md`) — always
  start probes from `q_neutral.clamp(lower, upper)` and decide/document what
  the solver does with an infeasible x0 (project + warn is acceptable;
  silent is not).
- scipy is a test-time reference only — it must not become a runtime
  dependency.

### T2b.6 — Linear solvers: contract, keep two, delete the stubs  [S]

**Goal / done-when:** The linear-solve contract is
`solve(matvec_or_matrix, b, ridge=None) -> x`, batched, with damping passed
as `ridge` so it stays orthogonal to the solver (jaxopt digest §6). Cholesky
(on `cholesky_ex`, T2b.3) and LSTSQ survive; the CG and SparseCholesky stubs
and the TrustRegion strategy class are deleted.

**Current state (all verified 2026-07-17):**
- `optim/solvers/cg.py:19`, `optim/solvers/sparse_cholesky.py:15`,
  `optim/strategies/trust_region.py:16,19,22` — bodies raise
  `NotImplementedError`.
- Exported at `optim/solvers/__init__.py:8,11,13` and
  `optim/strategies/__init__.py:10,12`.
- Parametrized into contract tests: `tests/contract/test_protocols.py:84,91`
  and `tests/contract/test_pluggable_protocols.py:30,35`.
- `docs/concepts/solver_stack.md:162-170,203-209` advertises them;
  line 225 even documents a `"sparse_cholesky"` config Literal that does not
  exist in code (`tasks/ik.py:65` has no such value) — a docs lie to kill.
- Consumer grep (BVR + BHF, 2026-07-17): **zero** imports of
  `CG`/`SparseCholesky`/`TrustRegion` — deletion is consumer-safe. (BHF does
  import `GaussNewton`, the kernels, and `CostStack` — those stay.)
- M0 removes the *selectable* lies (`"cg"`/`"trust_region"` strings in
  `OptimizerConfig` and the factories at `tasks/ik.py:104-147`); T2b.6
  removes the stub *classes*. Verify M0 landed first (see Prerequisites); if
  executing before it, stop and report the ordering violation.

**Implementation plan:**
1. Define the new-path contract (protocol or plain convention — no new
   Protocol subpackage ceremony; two implementations is the floor, not an
   invitation): `solve(A, b, ridge)` where `A` is `(B, n, n)` (dense path;
   the matvec-callable form of the contract is *specified* now so M5 can add
   CG/banded solvers without changing LM, but only the dense form is
   implemented here — standing rule 5).
2. `ridge`: the solver applies `A + ridge[..., None, None] * I` itself (or
   wraps the matvec) — LM passes `mu` and never builds `H` twice.
3. Delete `cg.py`, `sparse_cholesky.py`, `strategies/trust_region.py`; fix
   both `__init__.py`s, both contract tests, and the
   `docs/concepts/solver_stack.md` sections (154-233) in the same commit.
   Also update the stale class list in `optim/solvers/base.py:4` docstring
   and `optim/strategies/base.py:4`.
4. LSTSQ: `torch.linalg.lstsq` batches natively; keep the 91c33ad dtype
   behavior in mind if the new path ever mixes dtypes (the engineering
   contract from M1 item 5 states the dtype policy — follow it).

**What to test:**
- Contract test: both solvers accept `(B, n, n)`/`(B, n)` and return
  `(B, n)`; `ridge` of shape `(B,)` shifts each element's spectrum.
- `grep -rn "CG\|SparseCholesky\|TrustRegion" src/ docs/ tests/` returns only
  historical mentions you deliberately kept (changelog); imports are gone.
- Full suite green after deletion.

**Pitfalls / do-not-forget:**
- Real CG / sparse / banded / Schur solvers are **M5**
  (`m5_sparse_trajectory_structure.md`) — do not reimplement the stubs
  "while you're there".
- `docs/reference/` API pages are auto-generated — regenerate rather than
  hand-edit.

### T2b.7 — Capture-readiness of `update` as an acceptance criterion  [S–M]

**Goal / done-when:** `update` satisfies the structural checklist of 03 §2.6,
enforced by tests/lint where enforceable — with the honest caveat, stated in
code comments and docs, that **certification is M6's actual capture/replay
parity test** (`m6_warp_fast_path_and_cuda_graphs.md`), not lint. A lint
cannot certify capture safety (codex round 2); it can only catch the known
violation patterns.

**The checklist (03 §2.6, applies to `update`):**
- fixed input buffers, updated with `copy_` rather than replaced — this is
  the *capture harness's* obligation at the run/replay boundary (M6); M2b's
  obligation is that `update` never requires shape/identity changes between
  iterations (pure function over fixed-shape tensors);
- warmup-then-record discipline (M6 harness; nothing in `update` may depend
  on iteration count via Python state);
- tensor addresses stable across replays (same: no data-dependent shapes, no
  Python-side caching keyed on tensor identity);
- branch-free logic (`torch.where`, not Python `if` on tensor values) —
  including the `cholesky_ex` fallback as fixed tensor work (T2b.3);
- host syncs only at outer boundaries (`run` may sync once per iteration;
  `update` never);
- stated eligibility rules for custom residuals (document: a residual is
  capture-eligible iff its forward is fixed-shape, sync-free, and
  allocation-bounded; residuals that don't comply still work in eager mode —
  write this in the M2a custom-residual author guide's solver section).

**Implementation plan:**
1. Extend `tests/contract/test_hot_path_lint.py` to watch the new solver
   module(s) (its `WATCHED` tuple currently covers `kinematics`, `dynamics`,
   `optim/optimizers` — add the new path) and to forbid, inside the new
   solver files: `float(`/`bool(`/`int(` on expressions, `.item()`, `.cpu()`,
   `torch.eye`/`new_tensor`/allocation calls inside loop bodies. Coordinate
   with M1 item 7 (the lint extension) — if it already landed, just add the
   path; don't duplicate rules.
2. Whitelist `run` (module-level or via the existing `# bench-ok: <reason>`
   escape) for its one early-exit sync per iteration.
3. Structural runtime test: run `update` twice; assert the state pytree is
   structurally identical (shapes/dtypes/devices) across iterations and that
   `update` did not mutate its inputs.
4. Compile smoke test (best-effort, CPU): `torch.compile(update,
   fullgraph=True)` on a tiny fixed-base model must not raise on
   data-dependent control flow. Mark it `slow`; skip cleanly if the compile
   stack is unavailable. Note free-flyer FK only became fullgraph-compilable
   with M0 item 10 — use a fixed-base model here.
5. Write the honesty line into the solver docstring:
   "capture-ready by construction (03 §2.6 checklist); capture-*certified*
   only by the M6 capture/replay parity test."

**Pitfalls / do-not-forget:**
- CUDA is broken on this box (standing rule 7) — never claim graph capture
  was validated locally. The compile smoke test is a proxy, not proof; say
  so in the test docstring.
- Don't contort `update` into in-place-buffer style to "help" capture —
  allocations during capture are legal under graph memory pools (03 §2.6);
  purity + fixed shapes + no syncs is the actual requirement.

### T2b.8 — The solver-quality probe suite  [M]

**Goal / done-when:** The probe set from `plan/research/audit_optim_stack.md`
is committed as tests and passes. Write the probes early (xfail/skip until
their tasks land) so every task loops against them. Suggested location:
`tests/optim/test_solver_quality.py` + a committed benchmark definition under
`tests/bench/` (standing rule 4: hardware, dtype, shapes, warmup, statistics
— checked in, never bare chat ratios).

The probes, extracted from the audits (float32 per workspace test rules;
tolerances below are stated defaults — if measurement forces a change,
change the number *in the test with a comment*, never silently weaken an
assertion to `> 0`):

- **P1 — Bounded-interior regression** (audit §2.3; the M0-documented
  stall). Panda, pure pose cost (`limit_weight=0, rest_weight=0` moral
  equivalent in the new Problem), target = FK of
  `qt = 0.7·lower + 0.3·upper` (strictly interior), start from
  `q_neutral.clamp(lower, upper)`. Old behavior: maxiter at 300 iters, cost
  3.4e-2, pos_err 0.31 m. **Assert:** `status == converged`, final cost
  < 1e-8, position error < 1e-3 m, iterations ≤ 60 (the unbounded solve
  measured 25; give 2.4× headroom for the bounded machinery).
- **P2 — Unreachable-target KKT probe** (codex probe, 01 §1.4). An
  unreachable Panda pose: generate the target from FK of a configuration
  pushed 0.5 rad *outside* selected joint limits (so the unconstrained
  optimum violates the box; codex measured 0.259 m error, three joints
  pinned, `maxiter` after 300). **Assert:** terminal status is
  `stalled_at_bounds` (not `maxiter`), the KKT/projected-gradient residual
  is < tol, at least one coordinate sits exactly at its bound, and the run
  terminates in ≤ 100 iterations. The residual error itself may remain large
  — the target is genuinely unreachable; the honesty of the status is the
  assertion.
- **P3 — Facade-level feasible target** (audit §2.3's consequence probe:
  default-config `solve_ik` failed an easy feasible target with 0.31 m
  error). In M2b, run this at the new solver layer (build the IK problem on
  M2a's `Problem` directly — pose + limit + rest residuals with default-ish
  weights); target = FK of a random in-bounds q (seeded). **Assert:**
  converged, pos_err < 1e-3 m. M2c re-runs this same probe through the
  re-based `solve_ik` facade.
- **P4 — GN non-divergence** (audit §2.12: GN measured 1.21 m error from
  unconditional accept). Run the GN preset on P3's problem. **Assert:** the
  final cost ≤ initial cost (monotone accept logic), and status is honest
  (converged or maxiter — never a silent divergence).
- **P5 — Robust-kernel consistency** (audit §2.6; builds on M0's ρ-accept
  fix). A point-fitting toy problem with 20% gross outliers: Huber converges
  to within 10% of the clean-data optimum where an L2 accept stalls or lands
  off; the T2b.4 accept/reject direction tests (Huber-accept /
  Tukey-reject) pass.
- **P6 — Batched parity, 128 targets, ONE call** (the roadmap's headline
  acceptance). Protocol:
  1. Seeded: sample 128 in-bounds Panda configurations, targets = FK of
     each; one shared initial q (clamped neutral).
  2. Batched run: one call with `(128, nq)`-batched Values.
  3. Sequential runs: 128 calls of the SAME solver with B=1 slices
     (identical hyperparameters). "Sequential" means B=1 through the same
     implementation — that is the honest comparator and doubles as a shape
     regression test.
  4. **Compare per element:** (a) statuses equal, except elements within the
     threshold carve-out — standing rule 3: an element whose convergence
     metric at exit is within 10× of `tol`, or whose iteration count
     differs between runs, may legitimately differ in status; assert ≥ 90%
     exact status agreement and 100% agreement on {converged ∪
     stalled_at_bounds} vs {failed}; (b) for elements converged in both:
     |cost_batched − cost_seq| ≤ 1e-6 + 1e-2·max(cost) (float32 batched
     linalg reassociation compounds over iterations — costs near zero
     dominate the atol term), and both solutions' pose errors < 1e-3 m
     (compare task-space error, NOT q — IK is multi-modal and bitwise-q
     equality is not the claim).
  5. Also assert the batched call returns `(128,)`-shaped
     `status/cost/converged`.
- **P7 — Convergence-rate benchmark vs scipy** (audit rec 11). A committed
  benchmark definition (`tests/bench/`), CPU, float32, comparing iterations
  and final cost against `scipy.optimize.least_squares` (TRF) on P1 and P3.
  This is a tracked reference, not a hard gate: record the numbers; alert
  only on gross regression (>2× scipy's iteration count). scipy is a
  dev/test dependency only.
- **P8 — Kernel rho/weight consistency** (audit rec 11). For each kernel:
  `weight(s) ≈ d rho/d s` via autograd or central differences at a grid of
  `s` values including the branch points (`delta²`, `c²`), rtol 1e-4.

**Pitfalls / do-not-forget:**
- Every probe seeds its RNG. Panda comes from `robot_descriptions` (already
  a test dependency); no mocking of FK (repo test policy).
- P1/P2's "old behavior" numbers are documentation for the test docstring —
  the legacy solver is not re-run in CI for comparison.
- Batched-vs-sequential parity uses stated tolerances and per-element
  statuses, never exact equality (standing rule 3).

## Milestone acceptance checklist

- [ ] New solver API: frozen hyperparams + `init_state` + pure `update` +
      derived `run`; external-loop equivalence and warm-start tests pass
      (T2b.1); built on M2a `Problem`/`Values`, legacy path untouched.
- [ ] Batched semantics: `mu (B,)`, `cost (B,)`,
      `mu[..., None, None]·I` per-element normal systems, accept/reject as a
      0/1 `torch.where` blend of already-computed tensors (one candidate
      residual; candidate Jacobian deferred), `state.converged (B,) bool`,
      converged elements ride along frozen (T2b.2).
- [ ] `cholesky_ex` info-mask fallback: one indefinite element cannot fail
      the batch; fallback is fixed tensor work, no dynamic element selection
      (T2b.3).
- [ ] Madsen–Nielsen: μ init from `max(diag(JTJ_scaled))`; accept
      `μ *= max(1/3, 1−(2ρ−1)³)`; geometric escalation on reject; gain ratio
      on the robustified cost; per-block scaling applied (T2b.4).
- [ ] Bounded algorithm: survey evidence produced, owner signed off, chosen
      algorithm implemented with feasible retraction, active-set updates,
      projection-consistent predicted reduction, KKT termination; statuses
      `converged / stalled_at_bounds / maxiter / failed` per element (T2b.5).
- [ ] Batched IK, 128 targets, one call, matches 128 sequential (B=1)
      solves per the P6 protocol, with per-element statuses.
- [ ] Bounds-active IK converges where the M0-documented behavior stalled
      (P1 green; P2's honest-status assertion green).
- [ ] The full solver-quality probe suite P1–P8 passes / is committed.
- [ ] `update` passes the extended hot-path lint + structural purity test +
      compile smoke test; the docstring states capture certification is
      M6's capture/replay parity test, not lint (T2b.7).
- [ ] CG/SparseCholesky/TrustRegion deleted; `solve(A_or_matvec, b, ridge)`
      contract in place; docs/contract tests updated (T2b.6).
- [ ] `optim/CLAUDE.md`, root `CLAUDE.md` "LM Solver Notes", and
      `docs/concepts/solver_stack.md` updated to describe the new stack and
      to mark the legacy path deprecated-pending-M2c.
- [ ] Full test suite green: `uv run pytest tests/ -v` (897 pass today;
      minus tests this file explicitly retires, plus the new ones).

## Out of scope

- **Matrix-free Adam, batched LBFGS, the phase engine, `solve_ik` re-base,
  `solve_trajopt` fixes, `costs/` shim retirement** — M2c
  (`m2c_first_order_phases_tasks.md`). Batched LBFGS is explicitly its own
  later item (per-element histories and line search are real design work).
- **Deleting the legacy `LeastSquaresProblem`/`minimize` path, `CostStack`,
  or anything BHF imports** (`tools/geometry/icp.py`, `sdf_fit.py`,
  `scripts/motion/optimize_motion.py:210`) — M2c/M4 migration
  (`m4_consumer_packs_and_migration.md`); standing rule 1.
- **Real CG, sparse/banded Cholesky, Schur elimination, block-sparse
  assembly** — M5 (`m5_sparse_trajectory_structure.md`). T2b.6 only states
  the matvec form of the contract.
- **Actual CUDA-graph capture, the fixed-trip loop driver, GPU benchmarks,
  capture/replay certification** — M6
  (`m6_warp_fast_path_and_cuda_graphs.md`).
- **Implicit differentiation of `run`** — later flagship; only the
  differentiation-contract decisions from M2a/03 §9 constrain state shape
  here. Do not add `create_graph` machinery.
- **Compacting converged elements out of the batch** — a later optimization;
  correctness does not need it.
- **Geodesic acceleration** (jaxopt §5.5) and **QR on the augmented system
  `[J; √μ·I]`** (jaxopt §5.4) — do not port speculatively; note them in the
  T2b.5 survey only if probe evidence demands an ill-conditioning fallback.
- **Redesigning kernels/manifolds/Problem** — M2a owns those surfaces.

## References

- `plan/04_roadmap.md` — the M2b paragraph; its done-when is this file's
  acceptance checklist source (quoted criteria: 128-target parity,
  bounds-active convergence, probe set, 03 §2.6 capture checklist).
- `plan/03_architecture.md §4` — optimizer-stack decisions (state pattern,
  batched semantics, Madsen–Nielsen, bounds, linear solvers); §2.6 — the
  capture-readiness checklist verbatim; §3 — the M2a `Problem`/`VarSpec`
  sketch (direction, not final signatures).
- `plan/01_assessment.md §1.2` (batched solving does not exist), §1.4
  (bounded-LM stall, corrected mechanism, three probe results), §2.4
  (host-sync inventory; fresh `torch.eye` per iteration).
- `plan/research/audit_optim_stack.md` — probe evidence: §2.2 batched crash,
  §2.3 bounded-interior probe (P1/P3 source), §2.6 robust-kernel
  inconsistency (P5), §2.8 ornamental options (T2b.6), §2.9 sync counts,
  §2.12 GN divergence + thin termination (P4), §5 rec 11 (probe list:
  kernel rho tests, bounded regression, batched tests, scipy benchmark).
- `plan/research/codex_plan_review.md` — A4 (corrected bounded-LM
  mechanism), A12 (robust acceptance), B3 (branchless precision: cache
  current residual, one candidate, defer candidate Jacobian; `cholesky_ex`
  info masks; a projected-gradient stop alone is not a bounds fix).
- `references/design/jaxopt.md` — the port source: §2 (Solver/OptStep
  pattern), §3 (bounded loop, two drivers), §5.2 (Madsen–Nielsen rules with
  book citations), §5.3 (blend semantics), §6 (`solve(matvec, b, ridge)`),
  §11 lessons 1–4, 8–9, 13 (GN as LM preset).
- `src/better_robot/optim/` — current scalar implementations (line refs in
  each task's Current state, verified 2026-07-17).
- Consumer evidence: `../BetterHumanForce/tools/geometry/icp.py:58,330-332`
  (live `GaussNewton.minimize` import — do not break),
  `../BetterHumanForce/tools/object_align/sdf_fit.py:40-41` (kernel imports).
- External: Madsen, Nielsen & Tingleff, *Methods for Non-Linear Least
  Squares Problems* (Alg. 6.18, Eq. 6.16); Branch–Coleman–Li 1999 (TRF);
  scipy `optimize/_lsq/trf.py`; Kanzow–Yamashita–Fukushima projected LM —
  survey material for T2b.5.
