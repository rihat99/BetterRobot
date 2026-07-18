# 02 — Rebuild the optimizer core

> **Implementation log (2026-07-18):** T1–T7 are implemented; the flat
> optimizer is 3,779 formatted Python lines and routes LM only through dense or banded
> systems. Important findings: `increase_factor`, `gain_ratio`, and
> `relative_decrease` remain algorithm/test load-bearing; the matrix-free warm
> start and pure linear-solve diagnostics were removable. One contract update
> outside the stated allowlist was required to remove the root
> `JacobianStrategy` export; details and final gates are in `02_results.md`.
> **Completed:** full gate `1415 passed, 2 skipped, 16 deselected`; contracts
> `312 passed`; Pinocchio parity `136 passed`; HTML and doctest builds green.

**Goal:** one lean, torch-extension-flavored optimization package. The target
API and package layout are in `plan/02_architecture.md` §1 — read it first;
this order adds the mechanics and the evidence. Everything here reshapes the
surviving stack (`optim/blocks/` + shared solvers/kernels) after phase 1's
deletions.

**Size budget:** `optim/` ≤ 3,800 lines when done (~6,300 post-phase-1). The
itemized cuts alone reach ~4,700; the rest comes from rewriting `problem.py`
(1,097), `solver_lm.py` (1,235), and `variables.py` (359) under the new
validation policy — if the rewrite dividend does not materialize, stop and
report per standing rule 4.

**Behavior contract:** `solve_ik`, `solve_trajopt`, `solve_contact_forces`
keep their *solution-quality* regressions (same solutions within existing
tolerances). Solver-*state* semantics change deliberately where this order
says so (Adam, removed options) — the tests pinning removed semantics are
removed with them, listed per task.

**Contract-test authorization** (standing rule 2): this order authorizes
updating `tests/contract/test_hot_path_lint.py` (hard-coded optimizer paths),
`test_submodule_public_imports.py`, `test_docstrings.py`, and
`test_pluggable_protocols.py`/`test_protocols.py` where they name reshaped
modules. No other contract file changes.

Work in the listed order; each step ends green.

---

## T1 — Flatten the package

Move `optim/blocks/*` up into `optim/` per the architecture layout
(`manifolds.py`, `problem.py`, `lm.py` from `solver_lm.py`, `temporal.py`,
`implicit.py`, `first_order.py`); merge the six `optim/kernels/*` files into
one `optim/kernels.py` and `optim/solvers/*` into one `optim/solvers.py`.

**Deliberate API removals** (reachable options, not dead code — each gets a
ledger row and takes its tests along):

- `solvers/lstsq.py` and the `linear_solver="lstsq"` option
  (`tasks/ik.py:32,80,144`; test `tests/optim/test_config_wiring.py:157` and
  the LSTSQ cases in `test_linear_solvers.py`).
- The rank-deficient fallback in `Cholesky.solve` (`cholesky.py:23-41`,
  self-described backcompat; its case in `test_linear_solvers.py:77`).

Public import paths: everything the user needs importable from
`better_robot.optim` directly.

## T2 — Builder surface, one residual protocol

- `Problem()` grows `add_variable(name, *, shape=None, manifold=Euclidean(),
  bounds=None, mask=None, scale=None, time_axis=None)` and
  `add_residual(residual, *, weight=1.0, kernel=None, name=None, dim=None)`.
  `VarSpec`/`ResidualItem` remain as the records these methods create.
  Solvers freeze/validate at `init_state`.
- Add `RobotConfig.joint_bounds()` returning the model's joint-limit `Bounds`
  with unit-norm coordinates handled — it replaces the hand-masking ritual at
  `tasks/ik.py:278-310`. (On `RobotConfig`, *not* on `Model`: `Bounds` is an
  optimizer-layer type and the layer DAG forbids `data_model` importing up.)
- Residual name defaults from `residual.name` or `__name__` — never required
  twice. A plain function plus `dim=` is a valid residual (wrap internally).
- `reads` stays as the Jacobian-structure declaration, defaulting to the sole
  variable when the problem has exactly one. Remove the runtime
  undeclared-access policing (`providers.py:110-113`).
- Delete the scalar-objective subsystem: `ObjectiveItem`
  (`problem.py:111-127`), its evaluation branches (`problem.py:632-656`),
  `require_least_squares` (`problem.py:1088-1097`), the `ObjectiveTerm`
  protocol — zero production callers (verified, including by the adversarial
  review).
- Collapse the dual residual protocol: with the legacy stack gone, delete
  `ResidualState` handling in `residuals/base.py` (`_as_residual_state`,
  `_residual_model_q`, the `residual_jacobian` FD engine it feeds — confined
  to the legacy path, verified), the per-residual legacy `.jacobian()`/
  transpose-apply overrides across `residuals/*.py`,
  `kinematics/jacobian_strategy.py`, and the enum-to-string mappers
  (`tasks/trajopt.py:246+`, ik equivalent). Jacobian strategy becomes a plain
  `Literal["auto", "analytic", "jacrev", "jacfwd", "finite_difference"]`.
  Remove `JacobianStrategy` from the root `__all__` (contract change +
  ledger). Keep: analytic `jacobian_blocks`, temporal declarations, AD
  strategies, the explicit finite-difference debug strategy.

## T3 — Providers become declared helpers on a memo

**Three** providers exist in production — `RobotStateProvider`,
`SceneSDFProvider` (`residuals/scene_sdf.py:29`), and the contact-dynamics
provider (`tasks/contact_forces.py:85,331`) — so this is a reshape, not a
deletion. Keep the provider concept as "declared outputs + reads + a compute
function". Replace the graph engine (explicit cycle detection at
`providers.py:188-211`, topological sort at `problem.py:312-325`) with
recursive memoization on the evaluation context (`ctx.cached(key, fn)`-style;
a resolving-set turns provider cycles into a plain error in a few lines).
Preserve two load-bearing behaviors: (a) reads→variable dependency
propagation through providers — Jacobian block structure needs it
(`problem.py:274`, pinned by `tests/optim/test_providers.py:319`) — as a
simple union computed at problem freeze; (b) the one-evaluation cache
lifetime (graph-bearing outputs never survive an evaluation). Migrate all
three providers; their behavior tests stay green.

## T4 — One evaluation path (kill the shadow API)

Sixteen public/`_prevalidated` twins exist across `problem.py` (8),
`variables.py` (4), `manifolds.py` (4). Under the new validation policy
(`plan/02_architecture.md` §3): **structural** validation (names, batch
shape, dtype/device, scale compatibility — see `_validate_values`,
`problem.py:393`) runs **once** at `init_state`/public entry; **value
policing** (`isfinite` scans of user tensors) is dropped entirely. With
validation hoisted and value scans gone, one internal evaluation path
remains — delete the twins.

Consequences to handle honestly:

- Tests that pin exact non-finite-rejection errors change or go:
  `test_problem_blocks.py:228` (non-finite scale), `test_manifolds.py:312`
  (bounds NaN), and kin. Keep `Bounds` endpoint sanity if it is
  construction-time (one-shot) — the policy targets per-call scanning.
- Non-finite behavior statement for docs/state: an initially non-finite model
  is `FAILED` (`solver_lm.py:520`); a non-finite *trial* is rejected and may
  legitimately end `MAXITER` (`solver_lm.py:936,984,1150`). Do not promise
  more than the code does.
- Remove the constructor host-sync at `variables.py:110` and shrink
  `VarSpec.__post_init__` (66 lines → ~20) and `Problem.__init__`'s 140-line
  wall — keep the checks a `Literal`/type annotation cannot enforce, with
  good messages.

## T5 — First-order via torch.optim

Delete `optim/blocks/solver_adam.py` (335 lines) and add `first_order.py`
(~120): the persistent-tangent-delta loop from the architecture doc — any
`torch.optim` optimizer steps the delta, retraction applies it, rebasing
keeps the chart current, bounds project after retraction, a per-element
converged/step readout replaces `AdamState`/`AdamStatus`.

**This is an authorized behavior change.** The custom Adam's per-element step
counters and bias correction, atomic non-finite trial rollback
(`solver_adam.py:209-220`), and warm-start moment validation
(`solver_adam.py:252`) are the over-engineering being removed — delete
`tests/optim/test_solver_adam_matrix_free.py` cases that pin them
(`:191,:237,:300`) rather than reimplementing them. What must hold:
`solve_ik`'s `"adam"` and `"lm_then_adam"` solution-quality regressions.

`"lm_then_adam"` becomes two sequential solver calls inside `solve_ik`,
rebuilding the refinement problem with the `refine_disabled_items` weight
overrides the phase engine applied (`ik.py:87,384`, `phase.py:79`) — with the
T2 builder that is a few lines. Then delete `optim/blocks/phase.py` (154
lines; this preset was its only production caller). Ledger rows: `Adam`,
`AdamState`, `AdamStatus`, `Phase`, `PhaseResult`, `run_phases`.

## T6 — LM/GN trim

- `LMState`: drop pure-diagnostic fields not read by any task or surviving
  test (candidates: `linear_solve_relative_residual`, `previous_linear_step`,
  `relative_decrease`, `gain_ratio`, `increase_factor` — verify each, keep
  warm-start-essential state).
- **Deliberate API removal** of the matrix-free route:
  `linearization="matrix_free"`, `NormalOperator` consumption in the solver
  (`solver_lm.py:596,644`), `optim/solvers/normal_cg.py`, and the operator
  branches in `temporal.py`/`structure.py`. It is reachable via configuration
  and `solve_trajopt`'s route reporting (`trajopt.py:269,352,385`) — so also
  update `TrajOptResult`'s route/reason fields and docs, shrink
  `LinearizationReason` to surviving values, and update/delete the
  matrix-free cases in `test_solver_lm_routing.py:110,145` and
  `test_linear_operators.py`. `auto` chooses dense or banded. Ledger rows.
  (`for_future.md` records the resurrection point.)
- Deduplicate: `_broadcast_weight`/`_is_inactive`/`_validate_runtime_weight`
  (verbatim in `problem.py:137-173` and `temporal.py:216-245`),
  `_state_coordinates` (`solver_lm.py:306`, `implicit.py:291`), robust
  grouping logic (three sites). One home each.
- Fix while touching: `_objective_with_context` iterates inactive residuals
  and adds `kernel.rho(0)` for them (`problem.py:619-628`) while
  `_residual_with_context:566` skips them — make objective skip too.
- Fold `structure.py` survivors (`BlockBandedMatrix`, `LinearizationDecision`)
  into `temporal.py`/`lm.py`.

## T7 — Slim implicit differentiation

`implicit.py` (822 lines) → ≤ 500, keeping every guard that prevents
*silently wrong* gradients — they are tested correctness, not bloat: the
quaternion π-cut check (`implicit.py:351,632`; tests
`test_implicit_diff.py:223,251,282`), the Huber-kink scan
(`implicit.py:440`; test `:466`), the dense-cap/routing gates (test `:557`),
per-element convergence and active-set stability. What shrinks: the
nine-knob `ImplicitDiffConfig` (collapse knobs that no test distinguishes),
duplicated state-layout/eligibility validation (share with `lm.py` after T6's
dedup), and the error-formatting apparatus. Keep
`solve(differentiate="implicit")` as the entry (phase 3 wires it into
`solve_ik`). `tests/optim/test_implicit_diff.py` stays green throughout.

## Acceptance

- `optim/` ≤ 3,800 lines; `find src/better_robot/optim -name '*.py' | xargs wc -l` reported in results.
- The line-fit example from `plan/02_architecture.md` §1 runs exactly as
  written there; add it as a doc-tested example.
- Direct IK per the architecture sketch works: `RobotConfig(model)` +
  `joint_bounds()` + two `add_*` calls + `run` — no `VarSpec`, no provider
  wiring, no hand-masked bounds.
- `tests/optim/test_implicit_diff.py` green unmodified except knob-collapse
  mechanical updates; solution-quality regressions for all three tasks green.
- Full gate green; parity and (authorized-only) contract updates green;
  results file per standing rule 8, including every deleted test and why.
