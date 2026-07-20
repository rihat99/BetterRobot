# Order 01 — Simplify: delete speculative surface, flatten layers, warn on fallback

> **Implementation log (2026-07-20):** Complete on `dev`; source is
> 21,464 → 20,684 lines, CPU/CUDA/parity/contracts/docs are green. Review
> corrected the bounds fast-path flag at state init. Deviations: `implicit.py`
> is 439 rather than ~360 lines and `lie/` is 1,064 rather than ~1,000.

Read `plan/README.md` and `DESIGN_RULES.md` first. Everything below is backed
by the 2026-07-20 audit; every "zero callers" claim was grep-verified, but
re-verify before deleting (rules: deletion needs evidence, docs move with the
code, MIGRATION.md row per removed public symbol).

**Budget (hard):** `src/**/*.py` ends ≤ 20,700 physical lines (from 21,464).
Tests deleted here are those attached to deleted features; the general test
prune is Order 02.

## T1 — Fallback warnings (the only feature addition in this round)

1. Define `AutodiffFallbackWarning(RuntimeWarning)` in `optim` and export it.
2. In `problem.py` `jacobian_blocks` (`problem.py:506-516`): when
   `strategy == "auto"` and a residual's `jacobian()` returned `None`, warn
   once per `(Problem, residual name)` — track a `set` on the Problem, reset
   at freeze — naming the residual class, the chosen `torch.func` transform,
   and how to silence (provide `jacobian()` or pass an explicit strategy).
   Explicit `jacrev`/`jacfwd`/`finite_difference` never warn. Warn at first
   linearization, not per block, so a 50-iteration solve warns exactly once.
3. Same one-shot warning when a temporal-declared residual is routed dense
   because it lacks temporal blocks (`MISSING_TEMPORAL_BLOCKS` path,
   `temporal.py:263-275`).
4. Tests: a warning-fires test (chamfer or a jacobian-less custom residual
   under `auto`), a warns-once test, and a no-warn test for the analytic IK
   graph and for explicit strategies.
5. Document in `docs/concepts/residuals_costs_and_solvers.md` (strategy
   section) and `optim/CLAUDE.md`.

## T2 — optim: delete zero-caller surface

All greps re-run before deletion; MIGRATION.md rows for each public removal.

1. **`block_step_limits`**: remove `_limit_block_step_norms`,
   `_validate_block_step_limits` (`lm.py:131-172`), the constructor param and
   its threading through `_project_step`/`_init_state` (`lm.py:452,473,478,
   567-576`). Delete `tests/optim/test_solver_lm_step_limits.py`; rework
   `test_accept_icp_config.py` to drop its step-limit assertions only.
2. **`LU`** (`solvers.py:181-242`): delete class, export, its five tests in
   `test_linear_solvers.py`, and protocol-test references.
3. **`mask` / `scale` on `Variable`**: delete the params, `_free_indices`
   non-trivial branch, `gather_tangent`/`expand_tangent`
   (`variables.py:115-119,211-226`), the mask-conditional temporal properties'
   dead branches, and the corresponding `movable`/free-dim plumbing in
   `lm.py` (`_static_layout`) that exists only for masks. Migrate or delete
   mask/scale tests (`test_tangent_autograd.py` masked cases,
   retraction-mask tests, `test_variable_scale_sets_scaled_mu...`). Update
   `optim/CLAUDE.md` (it documents masks today) and any docs mention.
   This is the riskiest T2 item: run the full optim suite after each step.
4. **`manifolds.py`**: move `Bounds` to the top of `variables.py`, delete the
   file, fix the two test imports, update `optim/CLAUDE.md:70`.
5. Rename `_solver_common.py` → `utils.py` (convention decision 6 in the
   README).

## T3 — optim: readability and the unconstrained fast path

1. **`implicit.py` restructure** (target ~360 lines from 445, correctness
   untouched — the existing implicit test suite is the oracle):
   - Delete the `_ImplicitTerminalState` Protocol / `_StateSnapshot`
     duplication (`implicit.py:68-92`); one source of truth for the 9 fields.
   - Group into four titled sections: config & errors; eligibility & guards;
     KKT system; autograd bridge.
   - Split `backward` (`implicit.py:320-391`) into a terminal-validation
     helper and a static-gradients helper.
   - Rename for return contracts (`_eligibility` →
     `_eligible_and_stable_masks`, `_exact_system` →
     `_optimality_and_hessian`) and give each helper a one-line docstring
     naming what it returns.
2. **LM projected-gradient fallback** (`lm.py:678-698`): make it conditional
   on a `bool` computed once at state init from whether the problem declares
   any bounds (static Python bool — no per-iteration host sync, fixed shapes
   preserved). Unconstrained solves skip the two extra JVPs. Guard with the
   existing LM pattern tests plus one new test asserting bounded problems
   still take the pg path.
3. Section the flat helper lists in `lm.py` with titled comment blocks
   (state; layout; robust/IRLS; linearize/route; drivers).

## T4 — lie: flatten the facade

1. Inline `_impl.py`'s `so3_*` bodies into `so3.py` and `se3_*` into
   `se3.py`; shared primitives (`_hat3`, `_quat_mul`, `_quat_to_matrix`,
   `_matrix_to_quat`, `_taylor_theta2`) live in `so3.py`; `se3.py` and
   `tangents.py:21` import them from `so3`. Delete `_impl.py`.
2. `types.py`: replace the ~18 per-method lazy `from . import so3` with one
   module-level import (no cycle exists — verified).
3. Public `better_robot.lie` exports unchanged. Existing lie tests +
   Pinocchio parity are the oracle. Target: `lie/` ~1,000 lines (from 1,265).

## T5 — dynamics and tasks: prune dead modules

1. Delete `dynamics/derivatives.py` (AD sugar, no analytic content) and its
   three exports (`dynamics/__init__.py:19-23,47-49`). Inline a direct
   `torch.autograd.functional.jacobian` call into
   `tests/test_pinocchio/test_dynamics_derivatives.py` to keep its
   ∂τ/∂a-vs-CRBA identity checks. Drop the paragraph at
   `docs/concepts/dynamics.md:132-133` and the generated API page.
2. Delete `dynamics/integrators.py`, its export, and
   `tests/dynamics/test_integrators.py` (`model.integrate(q, dt*v)` is the
   documented spelling).
3. Delete `tasks/parameterization.py`, the `parameterization` parameter of
   `solve_trajopt` (`trajopt.py:172,187-190`), and
   `tests/tasks/test_trajopt_param.py`; update the B-spline rejection tests
   in `test_trajopt_named_blocks.py` to the new signature and the roadmap
   wording (B-splines remain a deferred direction).
4. `tasks/ik.py`: remove the dead `lbfgs`/`lm_then_lbfgs` spellings from the
   `Literal`/frozenset (`ik.py:232-236`) and trim the combination-policing
   validation (`_reject_unused` and friends, `ik.py:47-136`) to plain
   "unknown name" errors. Presets ignore inapplicable fields.
5. Create `tasks/utils.py` holding one hemisphere-align helper (merging
   `smoothing.py:13` and `trajopt.py:81`) and one `_public_diagnostics`
   (merging `ik.py:181` and `trajopt.py:144`).

## T6 — residuals: Node trim and rename

1. `nodes.py`: delete the write-only `_epoch` field and the `epoch` parameter
   threading (`nodes.py:22,36` and the six `problem.py` call sites); demote
   `_evaluation_depth` to a bool; drop the redundant model component of
   `RobotState.merge_key` (`nodes.py:85`). Node tests are the oracle;
   standalone-freshness and nested-scope semantics must not change.
2. Rename `residuals/_variables.py` → `residuals/utils.py`; collapse its two
   single-use protocols into their parents. `_temporal_jacobian.py` keeps its
   name.

## Acceptance

- Full non-bench/non-CUDA gate, Pinocchio parity, contracts, doc tests green.
- `uv run pytest tests/ -q -m cuda` green (run it yourself; pin a GPU).
- Greps show zero hits for: `block_step_limits`, `class LU`, `mask=`/`scale=`
  on variables in src, `manifolds`, `_impl`, `derivatives`, `integrators`,
  `parameterization` (src, non-generated docs).
- The fallback warning fires in its test and does not fire for the built-in
  IK graph.
- Line accounting table in `01_results.md`; src total ≤ 20,700.
- Contract files you may touch: `test_submodule_public_imports.py`,
  `test_public_api.py`, `test_protocols.py`, `test_pluggable_protocols.py`,
  `test_hot_path_lint.py` (module-name watch lists), docstring contracts.
