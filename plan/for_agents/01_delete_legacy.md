# 01 — Delete the legacy optimization stack

> **Implementation log (2026-07-18):** Caller recheck confirmed no surviving
> runtime dependency beyond trajopt. The plan overlooked the example's
> `CostStack.total_dim()` call and located the legacy-only solver-state suite
> under `tests/contract/`; both are handled under T1/T4 and reported as plan
> deviations.
> **Completed:** full gate `1467 passed, 2 skipped, 16 deselected`; contracts
> `324 passed`; Pinocchio parity `136 passed`; Python source net `-1,821` lines.

**Goal:** remove every part of the old flat-vector optimization stack and the
caller-less experiments. After this phase there is one optimization surface in
the library (the block `Problem` under `optim/blocks/` — phase 2 reshapes it).

**Why it's safe (caller-graph evidence, verified 2026-07-18):** no production
module imports `optim.optimizers.*`, `optim.problem.LeastSquaresProblem`,
`optim.state.SolverState`, or `optim.strategies.*` — they are reachable only
through `better_robot/__init__` re-exports and tests. `better_robot.optim.
{LevenbergMarquardt, GaussNewton, Adam}` already resolve to `optim/blocks/*`.
The one production tie is `solve_trajopt`, which takes a `CostStack` argument
but uses it purely as a container: `tasks/trajopt.py:207-243` reads only
`cost_stack.items.values()` and each item's `.active/.kind/.name/.residual/
.weight`; the `CostStack.residual/jacobian/gradient/slice_map/total_dim`
methods have zero callers anywhere. `optim/_graph_executor.py` is imported
only by `tests/optim/test_graph_capture.py`.

**Size budget:** net `src/` delta ≤ −1,700 lines. New code: ≤ 40 lines
(the trajopt input change).

**Contract-test authorization** (standing rule 2 — this order authorizes
updating exactly these, which reference deleted surfaces):
`tests/contract/test_cost_stack_location.py` (delete — its subject dies),
`test_docstrings.py:40` (imports `SolverState`),
`test_pluggable_protocols.py:21` / `test_protocols.py:26` (import deleted
strategies/optimizers), `test_submodule_public_imports.py:76` (expects
deleted symbols), `test_layer_dependencies.py:159` (expects the `costs`
package), `test_roadmap_stub_inventory.py:57` (names `optim/state.py`), and
`test_public_api.py` (required-symbol list). No other contract file changes.

---

## T1 — `solve_trajopt` takes residual items directly

Replace the `cost_stack: CostStack` parameter with
`residuals: Sequence[ResidualItem]` (the same `ResidualItem` used by
`Problem`; import from the block layer). In `_active_soft_residuals`
(`tasks/trajopt.py:207-243`):

- Drop the `isinstance(cost_stack, CostStack)` check and the `.active` /
  `.kind` gating — an item a caller doesn't want is an item they don't pass.
  This deliberately retires the `active`/`kind` semantics; the non-soft-kind
  rejection test (`tests/tasks/test_trajopt_named_blocks.py:218`) is updated
  or removed with them — an authorized behavior change, noted in the results
  file. Keep the "at least one residual" error and the per-item preparation
  (`_prepare_cost_residual`) exactly as is.
- The wrapping into `ResidualItem` collapses: callers now hand us items;
  re-wrap only to attach the prepared/adapted residual.
- Keep the `optimizer=` parameter behavior (named-block LM instance,
  rejection of legacy objects at `trajopt.py:314-320` becomes unnecessary
  once the legacy classes are gone — delete that branch too).

Update `examples/05_panda_trajopt.py` (`build_cost_stack` becomes a plain
list of `ResidualItem`s) and every test that builds a `CostStack` for
`solve_trajopt`. Behavior parity: the trajopt regression tests must pass with
unchanged solutions/tolerances after mechanical input conversion.

## T2 — Delete the legacy modules

Remove files (and their `__init__` wiring inside `optim/`):

| Path | Lines |
|---|---:|
| `optim/optimizers/` (LM, GN, Adam, LBFGS, MultiStage, LMThenLBFGS, base) | 791 |
| `optim/strategies/` (Constant, Adaptive, base) | 88 |
| `optim/problem.py` (`LeastSquaresProblem`) | 176 |
| `optim/state.py` (`SolverState`) | 105 |
| `optim/cost_stack.py` (`CostStack`, `CostItem`, `CostKind`) | 165 |
| `costs/` (forwarding shim package) | 22 |
| `optim/_graph_executor.py` (experimental, test-only) | 429 |

`optim/solvers/`, `optim/kernels/`, `optim/structure.py`, and all of
`optim/blocks/` **stay** — they are the live stack. (Phase 2 reshapes them;
do not pre-empt it.)

## T3 — Public surface and contracts

- Remove `CostStack` and `LeastSquaresProblem` from `better_robot/__init__`
  (imports at lines 42-43, `__all__` entries) and everything else that dies
  with T2 from `optim/__init__.py` (`CostItem`, `CostKind`,
  `OptimizationResult`, `Optimizer`, …).
- Update `tests/contract/test_public_api.py` deliberately (the required-symbol
  list shrinks). This is an authorized contract change.
- Append one ledger row per removed public symbol to
  `plan/migration_ledger.md` (the "to be removed" table there already names
  most of them — move rows up and fill in exact replacements).

## T4 — Tests

Delete tests whose only subject was deleted; port nothing. Known candidates
(verify each actually targets the legacy stack before deleting):
`tests/optim/test_multi_stage.py`, `test_bounded_lm_weakness.py`,
`test_matrix_free.py` (verify: legacy-Adam matrix-free, not the block
solver), `test_robust_kernels.py` (legacy row-wise IRLS — keep any parts that
pin the shared `optim/kernels/` rho/weight math;
`test_kernel_rho_weight_consistency.py` stays), `test_solver_state.py`,
`test_graph_capture.py`, and `tests/tasks/test_lm_then_lbfgs.py`.
**Not** wholesale candidates: `tests/optim/test_config_wiring.py` tests the
*live* `solve_ik` configuration surface — only its legacy-specific cases go
(phase 2 handles its `lstsq` case). A test that covers still-alive shared
code moves, not dies.

## T5 — Truth sweep (minimal)

Grep `src/ docs/ CLAUDE.md src/better_robot/*/CLAUDE.md examples/` for
`CostStack`, `LeastSquaresProblem`, `MultiStage`, `LBFGS`, `DampingStrategy`,
`optim.solve`, `GraphExecutor`. Fix every statement your deletions made false
with the **smallest truthful edit** — phase 5 owns the real docs rewrite; do
not restructure docs here. `optim/CLAUDE.md` loses its legacy half. The
front-page docs example and `docs/index.md` capability list must not name
deleted symbols.

## Acceptance

- `grep -rn "CostStack\|LeastSquaresProblem\|optim.optimizers\|SolverState" src/ examples/` → no hits.
- The deleted paths in T2 no longer exist; `better_robot.costs` no longer imports.
- Full gate green; pinocchio parity and (updated) contracts green; scoped ruff
  clean; Sphinx builds.
- Net `src/` delta reported in `01_results.md`, with exact test counts and
  every ledger row added.
