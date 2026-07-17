# M2c — First-Order Path + Phases + Task Rebase: Agent Execution Instructions

> Read `plan/for_agents/README.md` first. It carries the standing rules
> (deletion ordering, honesty rules, kernel requirements, test commands).

> **Update (owner decision 2026-07-17 — branch strategy):** BHF stays on
> the pre-redesign branch until the M4 migration. T2c.5's shim-retirement
> sequencing simplifies: `costs/`, `LeastSquaresProblem`, the legacy
> `GaussNewton.minimize` path, and the solver-global kernels may be
> deleted outright once no BR-internal caller (`solve_ik`/trajopt)
> consumes them — no BHF-driven ordering. The migration-note deliverable
> stays: enumerate the legacy surface and its replacements for M4's table.

## Mission

Finish the optimization redesign on the consumer-facing side. M2a built the
block-world `Problem`/`Values`/`VarSpec`/provider stack; M2b built batched
second-order solvers on the `init_state/update/run` pattern. M2c adds the
**first-order path** (a matrix-free, batched, manifold-correct Adam — the
current Adam/LBFGS build the full dense Jacobian *every iteration*, verified at
`optim/optimizers/adam.py:79` and `lbfgs.py:81` re-checked 2026-07-17,
contradicting their own docstrings), the **phase engine** (BVR's 188-line
`tools/optim.py` Problem/Phase/run_phases engine upstreamed — residual / weight
/ variable-mask / optimizer overrides with snapshot-restore staging), the
**`solve_ik` rebase** to a thin preset over `Problem(vars=(q,),
residuals=pose+limits+rest)`, a **fix-or-honest-drop decision on `solve_trajopt`
+ the B-spline path** (three verified breakages including a silent
wrong-quaternion path that standing rule 2 forbids leaving in place), and the
**`costs/` shim retirement with a migration note** that enumerates the full
BetterHumanForce (BHF) legacy surface and sequences deletion per standing rule 1.
The milestone's two headline acceptance criteria, verbatim from
`plan/04_roadmap.md`: *"a joint q + camera-extrinsics toy problem solves without
touching BR internals; BHF's ICP workarounds (trust-region clip, relative
damping, external stopping) reproduce as configuration, not subclassing."*

## Prerequisites

M2c sits on top of **both** M2a and M2b; it touches only the `optim`/`tasks`
layers and the `costs` shim. It needs no Warp kernels and no `torch.compile`
work.

- **M2a complete** (`m2a_variable_blocks_and_slice.md`). M2c consumes:
  - `VarSpec`/`Values`/`Problem` with `residual(values)`, `gradient(values)`
    (per-block **tangent-space** gradients — the matrix-free Adam's input),
    `jacobian_blocks(values)`, and the dense-assembly path.
  - The manifolds (`Euclidean`/`SO3`/`SE3`/`RobotConfig`) with feasible
    retraction, and the **tangent-space autograd helper** (T2a.2 — needs M0's
    θ=0 fix; the first-order Adam step retracts through `SO3`/`RobotConfig` at
    identity every warm start).
  - The provider DAG + evaluation-local context (T2a.4 — the phase engine's
    lazy shared-state substrate; BVR's `put_lazy` is exactly this).
  - Mask-as-elimination semantics + the expand/gather utility (T2a.3 — the
    phase engine's per-phase mask override rides this).
  - Per-block `scale` and the `ResidualItem` (residual + weight + kernel + name)
    — the phase engine overrides the weight column and the active set.
  - The custom-residual author guide (T2a.9) — T2c.6's acceptance problems and
    the ported built-in residuals both follow it.
  - **Verify it landed:** the M2a vertical-slice test passes; `Problem`,
    `VarSpec`, `RobotConfig`, `SE3`, and `Problem.gradient` import from
    whatever package M2a created (`grep -rn "class Problem" src/better_robot/`
    hits a block-world module, not `optim/problem.py`).
- **M2b complete** (`m2b_batched_second_order_solvers.md`). M2c consumes:
  - The `init_state/update/run` solver pattern and the batched LM/GN preset —
    the phase engine drives these, and `solve_ik` runs on them.
  - Per-element statuses `converged/stalled_at_bounds/maxiter/failed`, the
    `cholesky_ex` info-mask fallback, Madsen–Nielsen damping on scaled blocks
    (`damping_parameter` init from `max(diag(JᵀJ_scaled))` — this **is** BHF
    ICP's "relative damping" knob, T2c.6b), the bounded algorithm, and the
    `solve(A_or_matvec, b, ridge)` linear-solver contract.
  - **Verify it landed:** the M2b solver-quality probes P1–P8 pass; the new
    LM `update` is a pure function of a tensor-only state; the legacy
    `minimize` path is still present and untouched (M2b preserved it for BHF
    ICP — `grep -n "def minimize" src/better_robot/optim/optimizers/gauss_newton.py`
    still hits).
- **M0 carried through** (`m0_truth_and_correctness.md`): the enum-lie
  deletion, the `solve_ik` dtype-preservation fix, and the batched-input
  `NotImplementedError` guard on the legacy facade. T2c.3 **removes** that
  facade-level guard when it re-bases `solve_ik` onto the batched solver — this
  is the sanctioned point (M2b's file says so explicitly).
- Consumer repos readable, read-only, never modified:
  `/data3/rikhat.akizhanov/better/BetterVideoReconstruction` (BVR) and
  `/data3/rikhat.akizhanov/better/BetterHumanForce` (BHF).

## Sizing & parallelism

Roadmap effort: **M** (≤ 1 week) overall, on top of the M2a/M2b foundations.

| Task | Effort | Depends on |
|------|--------|------------|
| T2c.1 matrix-free batched Adam | S–M | M2a (`Problem.gradient`, manifolds), M2b (solver pattern) |
| T2c.2 phase engine | M | T2c.1, M2b solvers, M2a (providers, masks, ResidualItem) |
| T2c.3 `solve_ik` rebase + built-in residual port | M | T2c.1, T2c.2, M2b (LM) |
| T2c.4 `solve_trajopt` + B-spline: **evidence gate** | S–M | assessment: none; fix/drop: owner sign-off |
| T2c.5 `costs/` shim retirement + migration note | S | survey: none; shim install: T2c.3 |
| T2c.6 acceptance problems as tests | M | T2c.1, T2c.2, T2c.3, M2b |

Dependency order:

```
T2c.1 → T2c.2 → T2c.3 → T2c.6      (the main spine, ordered)
In parallel from day one:  T2c.4 (assess → STOP for owner), T2c.5 (survey → note)
T2c.5's shim-install step joins after T2c.3 lands.
```

T2c.4 and T2c.5's survey are independent and can go to parallel agents
immediately. Everything on the spine is ordered as listed — the phase engine
drives the solvers, `solve_ik` uses the phase engine for its multi-stage
option, and the acceptance tests exercise the whole stack.

## Tasks

### T2c.1 — Matrix-free batched Adam on tangent gradients  [S–M]

**Goal / done-when:** A batched, manifold-correct Adam solver on the M2b
`init_state/update/run` pattern whose gradient comes from
`Problem.gradient(values)` (per-block tangent-space) — **J is never
materialized**. Moments live in tangent space per block; the step is applied
through each block's `manifold.retract`, so quaternions never see a Euclidean
step. `update` is branch-free and host-sync-free (capture-safe by construction,
consistent with M2b T2b.7). Batched LBFGS is **explicitly deferred** to a named
later milestone — do not half-port it.

**Current state:** `optim/optimizers/adam.py:79` computes
`J = problem.jacobian(state.x)` then `grad = J.mT @ state.residual` (line 80) —
the **full dense Jacobian every iteration**, on the *old* `LeastSquaresProblem`.
Its own module docstring (`adam.py:1-13`) says "The `problem.gradient(x)` route
is preferred… amortises the Jacobian build" while the code uses the dense route
(audit `plan/01_assessment.md §2.2`, `audit_consumer_gaps.md §4.2`). It is
scalar-only (`float(0.5*(r@r).sum())` at line 96, `float(grad.norm())` at line
100 — per-iteration host syncs), unbatched, and Euclidean-clamps after the step
(line 90-93). `lbfgs.py:81` and `:148` do the same dense-J build.
`tests/optim/test_matrix_free.py:1-11` documents a matrix-free swap that "was
never written" (audit §2.7) — this task finally makes that test real.

**Implementation plan:**
1. New solver module beside M2b's solvers (follow M2b's package layout;
   category-first file-naming, e.g. `solver_adam.py` — invoke `file-naming`).
   Do **not** modify `optim/optimizers/adam.py` (legacy path, retired with the
   rest of the legacy stack in M4 — standing rule 1).
2. `@dataclass(frozen=True)` hyperparameters: `lr`, `beta1`, `beta2`, `eps`,
   `tol`, `max_iter`. `AdamState` is a NamedTuple/frozen dataclass of **tensors
   only**, one per block plus scalars: `m: Values`, `v: Values`,
   `step: Tensor` (`(B,)` or `()`), `grad_norm: (B,)`, `converged: (B,) bool`.
   Fields are tensors from birth (`torch.zeros(...)`, never `0.0`) so the pytree
   structure is invariant across iterations (M2b discipline).
3. `init_state(values, problem)`: zero moments shaped
   `(B..., tangent_dim_free)` per block (tangent-dim **after mask elimination**,
   M2a T2a.3), matching `Problem.gradient`'s per-block output shape.
4. `update(values, state, problem)` — pure:
   - `g = problem.gradient(values)` — a per-block `Values`-shaped dict of
     tangent gradients (matrix-free; internally routed through T2a.2's tangent
     autograd or per-residual `apply_jac_transpose`). **Never call
     `problem.jacobian_blocks` here.**
   - Per block, standard Adam moment update with bias correction; `delta`
     tangent step.
   - `values_next[name] = spec.manifold.retract(values[name], delta[name])` —
     manifold retraction, **feasible** (M2a bounds ride the retraction).
   - Per-element convergence: `converged | (grad_inf_norm < tol)`, updated
     branch-free; frozen elements stop moving via `torch.where`.
   - **No `float()`/`bool()`/`.item()`** anywhere in `update`.
5. `run(values, problem, state=None)`: bounded Python loop; may
   `bool(state.converged.all())` **once per iteration** for early exit (the
   only place a host sync is allowed). Warm-start: reuse the passed `state`.
6. **Batched LBFGS — DEFER, do not build.** Add a docstring/roadmap note (and,
   if the phase engine's optimizer field can name it, an honest
   `NotImplementedError`), draft wording:

   > `NotImplementedError: batched LBFGS is not implemented — per-element
   > histories, per-element line search, and curvature-validity/history-reset
   > logic are a dedicated later milestone (see plan/04_roadmap.md §M2c and
   > §Optimizer stack). Use optimizer=Adam (first-order) or an LM/GN preset
   > (second-order) instead.`

   Do **not** port `optim/optimizers/lbfgs.py`'s two-loop recursion — it is
   scalar, dense-J, and its history/line-search logic does not batch (that is
   the whole reason it is deferred).

**What to test** (`tests/optim/test_solver_adam_matrix_free.py`, float32,
`write-tests` skill first):
- **Matrix-free proof:** monkeypatch `Problem.jacobian_blocks` (and any dense-J
  method) to raise; the Adam `run` completes and decreases cost — proving it
  went through `Problem.gradient` only. This is the test audit §2.7 says was
  never written; write it for real.
- **Manifold correctness at the singularity:** Adam on an `SO3` / `RobotConfig`
  block starting **at** identity (θ=0, requires the M0 fix) produces finite
  steps; quaternion coordinates stay unit after `retract` (roundtrip norm
  ≈ 1). A Euclidean-step Adam would drift off the sphere — assert it does not.
- **Batched:** B=1, B=8, and a `(2,3)` multi-axis batch run; per-element
  gradients/steps match B=1 slices within stated tolerance (standing rule 3);
  external-loop-over-`update` == `run(max_iter=N)`.
- **Warm start:** solve, perturb, re-run with the returned state; converges in
  fewer iterations than cold.
- **No-sync/purity:** extend the M2b hot-path lint `WATCHED` set to the new
  module (coordinate with M2b T2b.7 / M1 item 7 — add the path, don't duplicate
  rules); a structural test asserts the state pytree is shape/dtype-invariant
  across two `update` calls and inputs are not mutated.

**Pitfalls / do-not-forget:**
- `Problem.gradient` returns gradients on the **reduced (free)** tangent
  coordinates when a block is masked; the moments must match that shape, and
  the retraction must scatter the free-coordinate step back to full state via
  M2a's expand/gather utility. Do not size moments to full tangent dim.
- Adam is a **first-order** method: it takes no `linear_solver`/`kernel`/
  `strategy`/`damping` args. Do not reproduce the legacy warn-on-kwarg
  machinery (`adam.py:59-69`) — the new API simply doesn't accept them.
- `Problem.gradient` already applies per-item robust kernels (M2a records the
  kernel on `ResidualItem`; IRLS is solver-side per M2b). Do not re-weight.

### T2c.2 — The phase engine  [M]

**Goal / done-when:** BVR's `tools/optim.py` engine (188 lines, read it in
full) upstreamed to the block world, losing nothing it needs. A `Phase` =
active residual set + weight overrides + variable-mask overrides + optimizer
config, with MultiStage's **snapshot/restore** semantics kept (try/finally, so a
phase that raises does not leak overrides). A phase runner runs a list of phases
in order over one `Problem`/`Values`, mutating `Values` (or returning updated
`Values`) between phases, fresh solver state per phase.

**Feature-by-feature map (every BVR feature must land somewhere — no loss):**

| BVR `tools/optim.py` feature | file:line | BR M2c home |
|---|---|---|
| Lazy shared `State` (`put`/`put_lazy`, maker called ≤ once, only if an active term reads it) | `optim.py:27-51` | **M2a provider DAG + evaluation-local context** (T2a.4) — the phase engine reuses it; it does **not** build a second cache |
| Per-phase weight column (`Phase.weights: dict[str,float]`, 0 ⇒ term off) | `optim.py:112` | `Phase.weight_overrides` + the active set = residuals with nonzero effective weight |
| Term active iff weight ≠ 0 (weight column *is* the objective) | `optim.py:86-96` | active-set derivation from the merged weight column; inactive residuals (and providers only they read) skip evaluation (M2a laziness) |
| Per-DOF gradient mask (`Phase.grad_mask`, multiplies each leaf's grad, zeros freeze DOFs) | `optim.py:114,178-182` | `Phase.mask_overrides` → per-block `VarSpec.mask` override → **elimination** (M2a T2a.3). Note the upgrade: BVR *soft*-masks the gradient; BR *eliminates* the coordinate (a zeroed column makes JᵀJ singular — codex B2.2). Same effect (frozen DOF), sounder mechanism |
| `on_start` hook (runs once before a phase's first iter, even iters=0) | `optim.py:107,147-148` | `Phase.on_start: Callable \| None` |
| Fresh optimizer per phase ("no stale momentum leaks across a DOF unfreeze") | `optim.py:170` | phase runner calls `solver.init_state` fresh at each phase entry (never carries Adam moments across a mask change) |
| Per-phase optimizer choice (adam / lbfgs) | `optim.py:113,152-170` | `Phase.optimizer` = a solver **instance** (Adam from T2c.1 or an LM/GN preset from M2b), not a string. BVR's `"lbfgs"` maps to a **deferred** path — use Adam or LM until batched LBFGS lands (T2c.1) |
| MultiStep lr decay | `optim.py:171-173` | optional Adam hyperparameter (lr schedule as a frozen field). Do **not** port an optax-style preset zoo (03 §4: "do not port the preset explosion") |
| Sync-free logging (floats only at log time — "no per-iteration GPU sync sneaks in") | `optim.py:80-82,118-124` | diagnostics stay tensors inside the loop; host-sync only at the phase/log boundary (M2b host-sync policy) |

**Current state:** `optim/optimizers/multi_stage.py` is the closest existing
piece — `MultiStageOptimizer` + `_cost_stack_snapshot` (lines 113-146) already
snapshot `active`/`weight` and restore in a try/finally, and `OptimizerStage`
carries `disabled_items`/`weight_overrides`/per-stage solver+kernel+strategy
(lines 28-39). But it is bound to the old `LeastSquaresProblem`/`CostStack` and
knows nothing of variable masks or multiple blocks. No consumer imports it
(`audit_consumer_gaps.md §2.6`), so it is free to be superseded — but keep it
alive as the legacy shim (it imports `LeastSquaresProblem`) until M4 unless the
`solve_ik` rebase (T2c.3) leaves it with zero callers, in which case it may be
retired then (see T2c.3 sequencing note).

**Implementation plan:**
1. New module in the block-world package (name via `file-naming`, e.g.
   `phase.py` with `Phase` + a `run_phases`/`PhaseRunner`). `design-principles`
   before shaping the API (this is a real abstraction — apply the two-caller
   rule: BVR's engine + the M2c `solve_ik` `lm_then_lbfgs` preset are the two
   callers).
2. `@dataclass(frozen=True) Phase`: `name`, `iters`, `weight_overrides:
   dict[str,float]`, `mask_overrides: dict[str, Tensor]` (per var-block, tangent
   -dim-shaped bool/float, override semantics = which coordinates are eliminated
   this phase), `optimizer` (a solver instance), `on_start: Callable | None`.
3. A **snapshot/restore** context generalized from `_cost_stack_snapshot` to the
   block world: snapshot the `Problem`'s residual weights + active flags **and**
   the affected `VarSpec` masks; apply the phase's overrides; restore all three
   in a try/finally on exit (even on raise). Port the exact reverse-order
   restore discipline from `multi_stage.py:140-146`.
4. `run_phases(problem, values, phases, ...)`: for each phase — run `on_start`;
   skip the body if `iters<=0` (but `on_start` still ran, per BVR); enter the
   snapshot context; `state = phase.optimizer.init_state(values, problem)`;
   `values, state = phase.optimizer.run(values, problem, state=state)` bounded
   to `phase.iters`; carry `values` forward; restore. Return the final `values`
   (+ optional per-phase diagnostics).
5. Batched: everything flows through batched `Values`/solvers — phases run over
   the leading batch axis unchanged.

**What to test** (`tests/optim/test_phase_engine.py`):
- **Snapshot/restore incl. raise path:** after `run_phases`, the `Problem`'s
  residual weights, active flags, and block masks equal their pre-run values;
  a phase whose optimizer raises still restores them (port the raise-path
  coverage from `tests/optim/test_multi_stage.py`).
- **Weight column = active set:** a residual with a 0 override is not evaluated
  in that phase; a provider read only by that residual does not run (assert via
  M2a's provider-count monkeypatch from T2a.4) — proving BVR's laziness is
  preserved.
- **Per-phase mask override:** phase 1 freezes a DOF (assert those coordinates
  do not move); phase 2 unfreezes it (assert they move). The frozen coordinate
  is *eliminated*, not zero-columned (reuse M2a's nonsingular-normal-system
  assertion).
- **Fresh state per phase:** a two-phase Adam run where phase 2 unfreezes a DOF
  starts phase 2 with zero moments for that DOF (no momentum leak).
- **Mixed optimizers:** phase 1 = LM preset (M2b), phase 2 = Adam (T2c.1), both
  drive the same `Problem`; converges.
- **`on_start` runs once**, even for an `iters=0` phase.
- **Batched:** a `(B,)`-batched phased run matches B sequential runs within
  stated tolerance.

**Pitfalls / do-not-forget:**
- The phase engine is the **second concrete caller** of the block API that
  justifies the mask-override machinery — cite it if the `design-principles`
  review asks for the two-caller rule.
- Do not resurrect a per-phase FK/state cache — the M2a provider context is the
  sanctioned shared-state mechanism (constraint 2: caching is evaluation-local).
- Restore order matters: masks last-in/first-out, then weights, then active —
  mirror `multi_stage.py`.

### T2c.3 — `solve_ik` re-based as a thin preset + built-in residual port  [M]

**Goal / done-when:** `solve_ik` becomes a thin preset that builds
`Problem(vars=(q,), residuals=pose+limits+rest)` and runs it on the M2b LM
preset (default) or the T2c.2 phase engine (for the `lm_then_lbfgs`-style
multi-stage option), with **public behavior preserved** — every existing IK
test stays green, plus M2b's batched-IK probes now run *through the re-based
facade*. The old `LeastSquaresProblem` is no longer consumed by `solve_ik`; its
class-level deletion is sequenced (see the sequencing note below — some slips to
M4). This task also **ports the built-in kinematic residuals** (pose, position,
orientation, joint limits, rest) to the M2a block protocol (`reads=("q",)`,
reading FK from a `RobotStateProvider`), since the rebased `Problem` needs them
as block residuals.

**Current state:**
- `tasks/ik.py:217` builds a `LeastSquaresProblem` from a hand-assembled
  `CostStack` (`PoseResidual`/`JointPositionLimit`/`RestResidual`, lines
  196-209), a `_state_factory` that reruns FK per call (lines 212-214), and a
  string-dispatched optimizer/linear-solver/kernel/damping (lines 229-250).
- It silently casts float64→float32 (`initial_q.clone().detach().float()` at
  line 186, `.float()` on limits at 221-222 — M0 fixes this; carry the fix).
- The `lm_then_lbfgs` option (line 238) delegates to `LMThenLBFGS`
  (→ `MultiStageOptimizer`).
- The built-in residuals read `state.variables` as q by fiat
  (`residuals/limits.py:56-62`, `regularization.py:48-54`) and receive
  `(model, data, variables)` via `ResidualState` — M2a's out-of-scope
  explicitly deferred their port to `reads`/blocks to "M2b/M2c". **This is
  where it lands.**

**Implementation plan:**
1. **Port the kinematic built-in residuals to the block protocol.** For each of
   `PoseResidual`, `PositionResidual`, `OrientationResidual`, `JointPositionLimit`,
   `RestResidual`: add `reads=("q",)` (or the provider output name), read FK
   from the M2a `RobotStateProvider` context instead of a bare `ResidualState`,
   and return `jacobian_blocks(ctx) -> {"q": J}` keyed by var name (the analytic
   LWA-Jacobian math is correct and fast — **keep it unchanged**, 03 §6 change
   1). Follow the T2a.9 author guide verbatim; friction is a guide bug.
   **Coordinate with M2b:** if M2b already made a minimal port to run its P3/P6
   IK probes, consolidate onto that — do not create a parallel second port.
2. Parity-test the ported residuals against the legacy ones (identical residual
   and Jacobian values on a Panda pose, atol ~1e-5 fp32) before deleting/
   shimming anything.
3. Rewrite `solve_ik` (`tasks/ik.py`) as: build the `q` `VarSpec` (RobotConfig
   manifold, state-space bounds from `model.lower/upper_pos_limit`), a
   `RobotStateProvider(model, var="q")`, the pose/limits/rest `ResidualItem`s
   (weights from `IKCostConfig`), a `Problem`, then run the M2b LM preset. Map
   `optimizer="lm"|"gn"` → M2b presets; `optimizer="adam"|"lbfgs"` → T2c.1 Adam
   (LBFGS deferred — either alias to Adam with a `DeprecationWarning` or raise
   the T2c.1 honest error; pick one and document); `optimizer="lm_then_lbfgs"`
   → a **two-phase `Phase` spec** (T2c.2): LM phase then a refine phase (LM or
   Adam, not the deferred LBFGS), carrying `refine_disabled_items` as a phase
   weight/active override.
4. Preserve the `IKResult` public surface (`q`, `residual`, `iters`,
   `converged`, `.fk()`, `.frame_pose(name)`, `.q_only()`) bit-for-bit.
   `converged` is now per-element (M2b) — for the unbatched call reduce it to a
   scalar bool; for a batched call expose the `(B,)` mask (this is the *new*
   capability the M0 guard was blocking).
5. **Remove the M0 batched-input `NotImplementedError` guard** from the facade
   (M2b's file names this the sanctioned removal point) — batched `solve_ik`
   now works.
6. Delete dead `IKCostConfig` fields with no consumer (`collision_margin`,
   `collision_weight` — audit §2.13) and the `robot_collision=` param (unread,
   `tasks/ik.py:157,171`) **only if** M0/M1 did not already; if a collision
   pass is planned for M4, leave a one-line pointer instead.
7. Update `tasks/CLAUDE.md`, root `CLAUDE.md` (the "IK API" / "LM Solver Notes"
   sections), and `docs/concepts/tasks.md` in the same change — they describe
   the `LeastSquaresProblem`/string-dispatch design that no longer exists.

**Sequencing note on `LeastSquaresProblem` deletion (standing rule 1 — a
plan-vs-code correction):** the roadmap says *"the old `LeastSquaresProblem`
dies here"*, but **it cannot be fully deleted in M2c.** Verified consumers of
the class after this task:
- `tasks/trajopt.py` (`_ChainRuleProblem` subclass + direct use) — retired or
  kept per T2c.4's owner decision;
- the **legacy `GaussNewton`** (`optim/optimizers/gauss_newton.py:15,29,45`
  reads `problem._nv`), which M2b **preserved** because BHF's ICP calls
  `GaussNewton(...).minimize(...)` (`BHF tools/geometry/icp.py:58,330-332`) on a
  duck-typed problem — this deletion is sequenced to **M4** after BHF's ICP
  migration commit exists (coordinate with `m4_consumer_packs_and_migration.md`);
- the other legacy optimizers (LM/Adam/LBFGS/MultiStage/LMThenLBFGS), which
  share the same shim and have no external consumer.

So T2c.3's deletion scope is: **`solve_ik` stops importing/using
`LeastSquaresProblem`**; the class survives as a legacy shim consumed only by
the legacy optimizer stack (kept for BHF ICP). Full class deletion lands in M4.
Record this explicitly in the migration note (T2c.5).

**What to test:**
- **All existing IK tests stay green, unchanged:** `tests/tasks/test_ik_regression.py`,
  `tests/tasks/test_lm_then_lbfgs.py`, `tests/optim/test_config_wiring.py`,
  `tests/test_skeleton_signatures.py`, `tests/contract/test_public_api.py`
  (the `solve_ik` symbol stays public).
- **M0 dtype preservation** stays green (float64 in → float64 out).
- **Batched `solve_ik`:** re-run M2b probe **P3** (facade-level feasible
  target) and **P6** (128 targets, one call, batched-vs-sequential per-element
  parity) *through the re-based `solve_ik`* — M2b's file explicitly hands these
  to M2c. `converged` is now `(128,)`.
- **Built-in residual parity** test (step 2 above) committed.
- **No `LeastSquaresProblem` in the `solve_ik` path:** an AST/grep contract
  assertion that `tasks/ik.py` no longer imports `optim.problem`.
- Bounds-active IK (M2b P1) converges through the facade.

**Pitfalls / do-not-forget:**
- The LWA-Jacobian convention and the SE3 pose format `[tx,ty,tz,qx,qy,qz,qw]`
  are unchanged — do not "clean up" the residual math while porting.
- `q_neutral` for the Panda violates joint 4's `[-3.07,-0.07]` bound; the
  rebased facade must start from `q_neutral.clamp(lower, upper)` (root
  `CLAUDE.md`) — preserve that behavior.
- `frame_pose(name)` uses the prefixed frame name (`body_panda_hand`, not
  `panda_hand` — audit §2.13/§2.6); do not regress the front-page snippet.

### T2c.4 — `solve_trajopt` + the B-spline path: fix-vs-drop evidence gate  [S–M]

**Goal / done-when:** the three verified breakages in the trajopt/B-spline path
are each either **fixed** (quaternion-safe interpolation, bounds honored,
replace-safe) or **temporarily dropped with an honest error** — but a silent
wrong-quaternion path is **not** an option (standing rule 2). This is an
**evidence-gated owner decision**: assess the current code, reproduce the bugs
as failing tests, write a short fix-vs-drop evidence note (effort, risk, and
what a fix requires vs what dropping leaves the consumer), and **STOP for owner
review** (standing rule 8). Do not silently pick.

**Current state (all verified 2026-07-17):**
1. **Bounds silently dropped.** The B-spline branch constructs the problem with
   `lower=None, upper=None` (`tasks/trajopt.py:207`) even when the caller passed
   `lower`/`upper`. No warning.
2. **Quaternion control points linearly mixed.** `BSplineTrajectory.expand`
   returns `self._basis @ z` (`parameterization.py:153`) — for a free-flyer
   model the expanded base quaternions are **not unit** (audit §2.11 measured
   norms 0.28–0.98 on a varied-orientation G1 trajectory). FK then runs on
   denormalized quaternions, and `_retract` is Euclidean on control points
   (`trajopt.py:169` — `return x + dv`), so nothing ever renormalizes. This is
   the silent-wrong path standing rule 2 forbids. No test covers it (the
   existing `tests/tasks/test_trajopt_param.py` uses a fixed-base 3/4-DOF chain,
   so it never triggers the quaternion bug).
3. **MultiStage crashes on the B-spline problem.** `_ChainRuleProblem.__init__`
   requires a keyword-only `dq_dz` (`trajopt.py:49`); `MultiStageOptimizer`
   clones the problem via `dataclasses.replace(problem, x0=…)`
   (`multi_stage.py:85`), which re-invokes `__init__` without `dq_dz` →
   `TypeError`. So `solve_trajopt(parameterization=BSplineTrajectory(...),
   optimizer=LMThenLBFGS(...))` cannot run.

**Implementation plan:**
1. **Reproduce all three as failing tests first** (the evidence):
   - free-flyer B-spline `expand` → assert base-quaternion norms are ~1 (fails
     today);
   - B-spline `solve_trajopt` with `lower`/`upper` passed → assert the expanded
     trajectory respects the box (fails today — bounds dropped);
   - `solve_trajopt(parameterization=BSplineTrajectory(...),
     optimizer=LMThenLBFGS(...))` → assert it runs (raises `TypeError` today).
2. **Assess fix-vs-drop for each and write the evidence note** (numbers, not
   adjectives):
   - *Fix path*: quaternion-safe B-spline requires SLERP-of-control-points or
     post-expand per-joint renormalization **plus** a manifold `_retract` on the
     control points (not Euclidean) — this is genuinely a spline-on-a-manifold
     problem, and a naive renormalize breaks the chain-rule Jacobian
     `dq_dz = kron(B, I)` (`trajopt.py:201`). Bounds: thread `lower/upper`
     through the B-spline branch (project the expanded trajectory or clamp
     control points with a warning). Replace-safety: make `dq_dz` a dataclass
     field with a default, or stop subclassing.
   - *Drop path*: raise an honest `NotImplementedError` from `BSplineTrajectory`
     (or from `solve_trajopt` when a non-`KnotTrajectory` parameterization is
     given with a free-flyer model / with bounds), pointing at the roadmap.
     Draft wording:

     > `NotImplementedError: BSplineTrajectory does not support floating-base
     > (free-flyer) models — linear mixing of quaternion control points
     > produces non-unit rotations (see plan/04_roadmap.md §M2c). Use
     > KnotTrajectory, or a fixed-base model. Bounds and manifold-safe spline
     > control points are a future milestone.`

     Note the fixed-base **knot** path works today and stays supported either
     way.
3. **STOP for owner review** with the evidence note. Do not implement the fix
   or the drop until the owner chooses. Then implement exactly the chosen path;
   whichever is chosen, criterion (2)/(3)'s reproduction tests must go green
   (a fix makes them pass; a drop replaces the norm assertion with an
   honest-error assertion).
4. `solve_trajopt`'s use of `LeastSquaresProblem`: it may stay on the legacy
   shim (which survives to M4 for BHF ICP anyway) — M2c's done-when does **not**
   require rebasing `solve_trajopt` onto the block `Problem`; only the B-spline
   correctness lie must be closed. If the owner picks "drop", the
   `_ChainRuleProblem` subclass can be deleted with the B-spline path.

**What to test:** the three reproduction tests (failing → green per the chosen
path); the existing fixed-base knot tests in `tests/tasks/test_trajopt_param.py`
stay green; if "drop", a test asserting the honest error fires for free-flyer /
bounded B-spline.

**Pitfalls / do-not-forget:**
- This is the roadmap's own "produce the evidence and STOP" task — resist the
  urge to just fix it; the owner may prefer a clean drop (trajopt-on-manifold-
  splines is arguably M5 sparse-trajectory work).
- A "fix" that renormalizes quaternions *after* `B @ z` but leaves the Euclidean
  `_retract` and the linear `dq_dz` is still subtly wrong (the Jacobian no
  longer matches the forward map) — say so in the evidence note.

### T2c.5 — Retire the `costs/` shim + migration note  [S]

**Goal / done-when:** `CostStack`'s role is subsumed by the block-world
`Problem.residuals` + the T2c.2 phase engine (03 §3: "`CostStack` becomes the
`residuals` tuple with weights/kernels/activity — its snapshot/restore semantics
are kept for staging"). A **migration note** enumerates the *full* legacy
surface BHF touches, states what replaces each symbol, and sequences shim
retirement per standing rule 1 — **shims die only after BHF's migration commit
exists, so some retirement slips to M4** (coordinate with
`m4_consumer_packs_and_migration.md`). The `costs.factory` stub is deleted.

**The full BHF legacy surface (verified 2026-07-17 — the consumer constraint is
bigger than the M2c plan line states):**

| Symbol BHF imports | Site | M2c disposition |
|---|---|---|
| `better_robot.costs.stack.CostStack` | `scripts/motion/optimize_motion.py:210` | Role replaced by `Problem.residuals` + phase engine. **Keep `costs.stack.CostStack` as a thin re-export shim** (canonical home moves under `optim/`); deletion → M4 after BHF migrates |
| `better_robot.optim.optimizers.gauss_newton.GaussNewton` (`.minimize`) | `icp.py:58`, call at `:330-332` | Replaced by M2b `init_state/update/run` solver + config (T2c.6b). **Legacy `GaussNewton` shim stays** (M2b preserved it); deletion → M4 |
| `better_robot.optim.kernels.cauchy.Cauchy` / `.huber.Huber` (`.rho(d²)`) | `sdf_fit.py:40-41` | **Kept, not retired** — standalone robust kernels are a keep (audit §6.5); now also first-class per-`ResidualItem` (03 §6 change 3). Record that they stay importable |
| `residuals.{AccelerationResidual, ContactConsistencyResidual, ReferenceTrajectoryResidual}` + `residuals.base.ResidualState` | `optimize_motion.py:211-216` | `ResidualState` → M2a evaluation context; these residuals port to `reads`/blocks (trajectory residuals ride M2c/M4). **Keep `ResidualState` + the old-style residual entry points as shims**; deletion → M4 |
| `tasks.trajectory.Trajectory` | `motion.py:17,193,254`, `playback.py:26`, `smoothing.py:32,169,245`, `view_motion.py:53` | **Kept, not retired** — `Trajectory` is a keep (audit §6.6). Record that it stays |
| deprecated `Data.oMi` | `playback.py:115,180,190`, `motion.py:201` | M1 kept `Data` alias shims; `oMi` → current name (`joint_pose_world`). **Shim stays**; deletion → M4 |

**Implementation plan:**
1. **Move `CostStack` (and `CostItem`) into `optim/`** (03 §1: "`costs/` merges
   into `optim/`; CostStack is one small file, not a layer"). The canonical
   class lives under `optim/`; `better_robot.costs.stack` becomes a re-export
   shim (`from better_robot.optim.<newloc> import CostStack, CostItem`) so
   `optimize_motion.py:210` keeps importing cleanly. Keep the top-level
   `better_robot.CostStack` export (it is in the frozen public `__all__`).
2. **Delete `costs/factory.py`** — it is a `NotImplementedError` stub
   (`factory()` raises; verified) with **zero consumers** in either repo
   (grep-confirmed). Remove it from `costs/__init__.py` (line exports `factory`)
   and any docs. The block world's plain-function-residual path (M2a T2a.9
   guide) supersedes it.
3. **Write the migration note.** Put it where
   `m4_consumer_packs_and_migration.md` says the migration table lives; if that
   file predates yours, create `plan/migration/bhf_legacy_surface.md` and leave
   a one-line pointer for the M4 executor. The note is the table above, plus:
   which symbols die in M4 vs stay, the coordinated commit ordering (BHF's
   migration commit must land **before** each shim deletion), and the
   `LeastSquaresProblem` deletion sequencing from T2c.3.
4. Update the layer-DAG contract test if it references `costs` as a node
   (`tests/contract/test_layer_dependencies.py`) — 03 §1 keeps the DAG but drops
   `costs` as a separate layer. Do not weaken the test; adjust the expected
   node set deliberately.

**What to test:**
- **Both consumer repos still import cleanly** — grep both repos for
  `better_robot` imports and confirm every path above still resolves (standing
  rule 1). A shim test in BR: `from better_robot.costs.stack import CostStack`
  succeeds and is the same class as the `optim/` canonical one.
- `grep -rn "factory" src/better_robot/costs/` returns nothing; no consumer
  imports it (re-confirm in both repos).
- Full suite green after the move (`tests/contract/*` included).

**Pitfalls / do-not-forget:**
- Do **not** delete `costs.stack.CostStack`, `GaussNewton`, `ResidualState`,
  `Trajectory`, or the `Data` aliases — every one has a live BHF import. This
  task *installs shims and writes the plan*; the deletions are M4's, after BHF's
  migration commit (standing rule 1).
- The migration note is a deliverable, not a comment — the M4 executor consumes
  it as the symbol-by-symbol table (03 §9 "Migration as a deliverable").

### T2c.6 — Acceptance problems as tests  [M] — the gate

**Goal / done-when:** the two roadmap headline acceptance criteria land as
permanent BR tests on synthetic data (CPU, seconds). (a) A joint-`q` +
camera-extrinsics toy problem solves using **only the public block API** (no BR
internals). (b) BHF's ICP workarounds reproduce as **configuration, not
subclassing**.

**T2c.6(a) — q + camera-extrinsics toy** (`tests/optim/test_accept_q_extrinsics.py`):
- Two variable blocks: `q` (`VarSpec("q", ..., manifold=RobotConfig(model),
  bounds=...)`) and camera extrinsics `T_cam` (`VarSpec("T_cam", shape=(7,),
  manifold=SE3())` — SE3 manifold, no box bound; M2a raises if you pass one).
- A custom residual (written by following the T2a.9 author guide) that reads
  **both** blocks: e.g. a reprojection-style / relative-pose residual coupling a
  frame pose from FK (via the `RobotStateProvider` on `q`) with the camera
  extrinsics — `reads=("q", "T_cam")`, `jacobian_blocks` keyed by both, or an
  autodiff fallback. Sketch of the setup (direction, not final signatures):

  ```python
  model = <small fixed-base chain>
  vars = (VarSpec("q", (model.nq,), RobotConfig(model), bounds=...),
          VarSpec("T_cam", (7,), SE3()))
  # residual r = log( T_cam⁻¹ ∘ oMf(q)[frame]  ∘  T_target⁻¹ )  (6-vector),
  # or a pinhole reprojection of oMf origins through (K, T_cam) vs 2D targets.
  problem = Problem(vars=vars, residuals=(ResidualItem("reproj", CamResidual(...), ...),))
  values = {"q": q0, "T_cam": T_cam0}
  values, state = LMSolver(...).run(values, problem)   # M2b, or Adam (T2c.1)
  ```

- **Assert:** converges to a fixed threshold on seeded synthetic data;
  gradients flow to **both** blocks (perturb the ground-truth `T_cam` and `q`
  and recover both); the SE3 block stays a valid unit quaternion after
  retraction; the solve touches no BR-internal symbol (import only the public
  block API + a solver).

**T2c.6(b) — ICP workarounds as config** (`tests/optim/test_accept_icp_config.py`):
Reproduce the *structure* of BHF's `fit_mesh_to_depth` inner solve at toy scale
(synthetic points, no consumer imports), as a BR `Problem` + solver **config** —
proving each documented workaround is a knob, not a subclass. Read
`BHF tools/geometry/icp.py` and `_icp_problem.py` and map each concretely:

| BHF workaround | Concrete evidence | BR config knob (no subclass) |
|---|---|---|
| **Trust-region clip** | `_icp_problem.py:_clip_dv` (`51-74`): caps `‖δρ‖≤0.20`, `‖δω‖≤0.50`, `|δσ|≤0.30`, applied inside `_retract` (`77-86`) because "BetterRobot's GN/LM solvers don't bound step magnitude" | A **tangent-space step-bound hyperparameter** on the solver (per-block max step norm), applied inside `update`'s retraction, branch-free clamp. If M2b's chosen bounded algorithm (T2b.5) is a reflective trust region, use its radius; **if the landed solver has no per-block step cap, add one as a frozen hyperparameter** (this test is its second concrete caller — standing rule 5 — so it is justified; add to the existing solver dataclass, do **not** fork/subclass it) |
| **Relative damping** | `icp.py:326-330`: `eps_abs = icp_gn_relative_eps · max(diag(JᵀJ))` recomputed per outer iter, passed as GN `eps`; absolute eps "either over- or under-damps depending on M" | **Already the default** in M2b's LM: `mu0 = damping_parameter · max(diag(JᵀJ_scaled))` (T2b.4 step 2). BHF's `icp_gn_relative_eps=1e-3` → `damping_parameter=1e-3`. Just set the knob |
| **External stopping** | `icp.py:340`: outer loop breaks on `|Δrmse| < icp_tol`; GN called with `tol=1e-30` (`:330`) to disable its internal `‖Jᵀr‖` test, which "scales with data magnitude and is hard to tune" | **Consumer owns the outer loop**: call `solver.update` in the test's own re-association loop with its own `|Δcost|` stop (the jaxopt "projects keep their loop; BR owns the step" pattern, M2b T2b.1). Relative-cost-decrease is also a built-in per-element convergence criterion now (M2b T2b.2 step 6) — either path is config |
| (also) duck-typed `_nv` | `_icp_problem.py:99,142` provide `_nv` because GN reads `problem._nv` (`gauss_newton.py:45`) | Gone: the block `Problem`/`VarSpec` expose tangent dims properly (M2a) — no private-attribute hack |

- Model the Sim3 as a block problem: a `VarSpec` for the rigid pose (`SE3()`)
  plus a Euclidean `log_s` block, or a single Sim3-style manifold if M2a defined
  one; a custom point-to-plane residual `r_i = n_i · (v_t,i − p_i)` reading that
  block (the `_icp_problem.py:119-130` math, ported as a block residual).
- **Assert:** the toy ICP converges on synthetic point-to-plane data with the
  three knobs set as config; **no subclass** of `Problem`/solver/state exists in
  the test (the residual is test-local consumer code per the M2a author guide);
  the trust-region step cap actually bounds the first step (a poorly-initialized
  start does not propose a >1 m translation — the exact failure `_clip_dv`
  guards).

**Pitfalls / do-not-forget:**
- Synthetic data only; **do not import from the consumer repos** and **do not
  migrate BHF** — the actual BHF ICP migration is M4. T2c.6(b) proves the
  mapping *exists* as config; M4 executes it.
- If T2c.6(b) surfaces a missing solver knob (the step cap), that is a real M2c
  finding — add the config knob (frozen hyperparameter), don't subclass, and
  note it back to M2b's solver docstring.
- The residuals here are **test-local** consumer code (like the M2a slice), not
  new entries in `residuals/` — vision/SDF/chamfer library residuals are M4.

## Milestone acceptance checklist

- [ ] M2a and M2b acceptance checklists pass; block `Problem`/`VarSpec`/
      manifolds/providers and the M2b `init_state/update/run` solvers import.
- [ ] **Matrix-free batched Adam:** gradient via `Problem.gradient` (tangent-
      space, per block); J never materialized (monkeypatch-J-raises test
      passes); manifold-correct at θ=0 (quaternions stay unit); batched +
      warm-startable; `update` sync-free (hot-path lint extended). (T2c.1)
- [ ] **Batched LBFGS explicitly deferred** — not half-ported; named as a later
      item; honest error or Adam-alias where it could be selected. (T2c.1)
- [ ] **Phase engine:** every BVR `tools/optim.py` feature mapped (lazy shared
      state → M2a providers; per-phase weight column → weight/active overrides;
      per-DOF grad mask → per-phase mask-elimination override; on_start; fresh
      solver state per phase; per-phase optimizer; sync-free logging);
      snapshot/restore incl. raise-path tested. (T2c.2)
- [ ] **`solve_ik` re-based** as a thin preset over
      `Problem(vars=(q,), residuals=pose+limits+rest)`; built-in kinematic
      residuals ported to `reads`/blocks with analytic-Jacobian parity;
      `LeastSquaresProblem` no longer in the `solve_ik` path (full class
      deletion sequenced to M4 — legacy `GaussNewton` shim for BHF ICP). (T2c.3)
- [ ] All existing IK tests + M0 dtype-preservation stay green; M2b batched-IK
      probes P3/P6 run through the re-based `solve_ik`. (T2c.3)
- [ ] **`solve_trajopt` + B-spline:** three breakages reproduced as tests;
      fix-vs-drop evidence note written; **owner signed off**; the chosen path
      implemented; **no silent wrong-quaternion path remains** (standing rule 2).
      (T2c.4)
- [ ] **`costs/` shim retired with a migration note:** `CostStack` canonical
      home under `optim/` with a `costs.stack` re-export shim; `costs.factory`
      stub deleted; the migration note enumerates the full BHF legacy surface,
      states each replacement, and sequences deletion (some → M4) per standing
      rule 1; both consumer repos still import cleanly. (T2c.5)
- [ ] **Acceptance (a):** q + camera-extrinsics toy (two blocks, SE3 extrinsics)
      solves via the public block API only, gradients to both blocks. (T2c.6a)
- [ ] **Acceptance (b):** BHF's ICP trust-region clip, relative damping, and
      external stopping reproduce as configuration, not subclassing — as a BR
      test on synthetic data. (T2c.6b)
- [ ] Full suite green (`uv run pytest tests/ -v`; 897 baseline plus M0/M1/M2a/
      M2b/M2c deltas); pinocchio-parity suite untouched and green; both consumer
      repos import cleanly (standing rule 1 grep).

## Out of scope

- **Real batched LBFGS** — a dedicated later item; per-element histories, line
  search, and curvature/history-reset logic are real design work (T2c.1 names
  it, does not build it).
- **Sparse/banded trajectory assembly, Schur elimination, spline-on-manifold
  trajopt** → `m5_sparse_trajectory_structure.md`. T2c.4 closes the B-spline
  *correctness lie*; it does not build a manifold-aware trajectory optimizer.
- **Actual BHF/BVR consumer migration and shim deletion, `LeastSquaresProblem`/
  legacy `GaussNewton` class deletion, `Data.oMi` removal** →
  `m4_consumer_packs_and_migration.md`. T2c.5 writes the migration note and
  installs shims; T2c.6 proves the mappings exist. Deletions land in M4 after
  BHF's migration commit (standing rule 1).
- **Library vision/scene residuals** (projection, chamfer, SDF trio,
  Geman-McClure kernel) and the inverse contact-force task → M4. T2c.6's
  residuals stay test-local consumer code.
- **Implicit differentiation of `run`, CUDA-graph capture, Warp kernels,
  `torch.compile`** → later milestones. M2c is torch-eager optim/tasks work.
- **New manifolds beyond M2a's four**, redesigning `Problem`/`VarSpec`/solvers —
  M2a/M2b own those. T2c.6(b) may *add one frozen solver hyperparameter* (the
  step cap) — that is config, not a redesign.
- Do not delete any surface with a live consumer import; do not modify the
  consumer repos.

## References

- `plan/04_roadmap.md` — the M2c paragraph (source of the two verbatim
  done-when criteria) and the standing rules; the §Optimizer stack "first-order
  methods: matrix-free only" and "phases" bullets.
- `plan/03_architecture.md §4` — first-order matrix-free rule, batched-LBFGS-is-
  its-own-milestone, phases = MultiStage snapshot/restore generalized to blocks;
  §3 — the `Problem`/`Values`/`ResidualItem` this builds on and "`solve_ik`
  becomes a thin preset / `LeastSquaresProblem` dies / `CostStack` becomes the
  residuals tuple"; §6 change 2 — the real autodiff fallback (jacrev over the
  tangent perturbation; FD only as `strategy="fd"` debug); §1 — `costs/` merges
  into `optim/`.
- `plan/01_assessment.md §2.2` — single-flat-variable blocker + dense-J
  Adam/LBFGS (verified `adam.py:79`, `lbfgs.py:81`); §2.5 — speculative/dead
  code (factory stub, MultiStage) + the BHF migration caveat (`costs.stack`,
  `Data.oMi`); §1.2/§1.4 — batched solving + bounded-LM (M2b's fixes that M2c
  surfaces through the facade).
- `plan/research/audit_consumer_gaps.md` — §3.1 (BVR `tools/optim.py` = the
  requirements spec, feature-mapped in T2c.2), §4.2 (dense-J Adam/LBFGS unusable
  at scale), §4.5 (BHF ICP documented workarounds — the T2c.6b evidence), §2.4
  (the full BHF import inventory), §6 (keep list: kernels, `Trajectory`,
  `CostStack.gradient`).
- `plan/research/audit_optim_stack.md` — §2.7 (matrix-free machinery unused; the
  "never written" Adam test), §2.8 (ornamental options), §2.11 (the three
  B-spline breakages — T2c.4 evidence), §2.13 (`solve_ik` float32 cast + dead
  config), §4 (keep list: MultiStage snapshot/restore is careful, correct code
  to generalize).
- Consumer evidence (read-only):
  `BVR tools/optim.py` (the 188-line engine — read in full for T2c.2);
  `BHF tools/geometry/icp.py` (relative damping `:326-330`, external stopping
  `:340`) + `_icp_problem.py` (trust-region `_clip_dv` `:51-74`, duck-typed
  `_nv` `:99,142`) for T2c.6b;
  `BHF scripts/motion/optimize_motion.py:210-216` (`CostStack` + residual
  imports), `tools/object_align/sdf_fit.py:40-41` (kernel imports),
  `tools/robot_motion/playback.py` (`Data.oMi`) for T2c.5.
- Sibling instruction files: `m2a_variable_blocks_and_slice.md` (the block API,
  providers, manifolds, tangent-autograd helper, author guide — the upstream
  M2c consumes and must not contradict); `m2b_batched_second_order_solvers.md`
  (the `init_state/update/run` pattern, LM/GN presets, `damping_parameter` =
  relative damping, statuses, linear-solver contract, the preserved legacy
  `minimize` path for BHF ICP); `m4_consumer_packs_and_migration.md` (owns the
  migration table + shim deletions T2c.5 sequences into it);
  `m5_sparse_trajectory_structure.md` (owns spline-on-manifold trajopt T2c.4
  defers to); `m0_truth_and_correctness.md` (the θ=0 fix, dtype fix, enum-lie
  deletion, batched guard T2c.3 removes).
