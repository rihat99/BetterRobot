# 01 Results — Delete the legacy optimization stack

**Status:** complete on `dev` (2026-07-18).

## Delivered

- `solve_trajopt` now accepts `Sequence[ResidualItem]`. Preparation uses
  `dataclasses.replace`, so weights, kernels, and robust `group_size` values
  survive adaptation. Active/kind flags and the deleted-optimizer rejection
  branch are gone; callers select a solve by passing only the wanted items.
- The Panda example and every trajectory test now construct residual items
  directly. Static-horizon residuals are sized before item construction.
- Removed `optim/optimizers/`, `optim/strategies/`, `optim/problem.py`,
  `optim/state.py`, `optim/cost_stack.py`, `optim/_graph_executor.py`, and the
  `better_robot.costs` shim package, including ignored bytecode directories.
- Removed all root and `better_robot.optim` exports backed by those modules.
  The live block `Adam`, `GaussNewton`, and `LevenbergMarquardt` exports remain.
- Deleted tests whose only subject disappeared. Preserved live coverage by
  moving the Warp layout/capture test to `tests/warp/test_fk_cuda.py` and the
  `lm_then_adam` diagnostic test to `tests/tasks/test_ik_regression.py`.
- Updated authorized contracts, current documentation, CLAUDE files, the
  trajectory example, generated API pages, and the roadmap inventory. Current
  public documentation no longer presents the removed API as available.

## Public migration rows

Added 18 semantic rows to `plan/migration_ledger.md`, grouping alias import
paths in each row:

1. `CostStack`
2. `CostItem`
3. `CostKind`
4. `LeastSquaresProblem`
5. `SolverState`
6. `SolverStatus`
7. `Optimizer`
8. `OptimizationResult`
9. deep-package `LevenbergMarquardt`
10. deep-package `GaussNewton`
11. deep-package `Adam`
12. `LBFGS`
13. `LMThenLBFGS`
14. `MultiStageOptimizer`
15. `OptimizerStage`
16. `DampingStrategy`
17. `Constant`
18. `Adaptive`

The pending deep-kernel and `ResidualState` rows remain for later work orders.

## Verification

| Check | Result |
|---|---|
| Baseline full gate | `1543 passed, 2 skipped, 28 deselected` |
| Final full gate | `1467 passed, 2 skipped, 16 deselected` |
| Pinocchio parity | `136 passed` |
| Contract suite | `324 passed` |
| Trajectory conversion scope | `14 passed` |
| Ruff check / format | clean across 26 changed Python files |
| Sphinx HTML | succeeded with 0 warnings |
| Required source/example grep | 0 hits |
| Deleted paths / `better_robot.costs` | absent / not importable |

The first post-deletion full gate had one failure and otherwise reported
`1466 passed, 2 skipped, 16 deselected`: the roadmap still listed the deleted
`src/better_robot/optim/state.py`. Removing that stale inventory row fixed the
failure; no failures remain.

## Line accounting

- Python under `src/`: 26,881 → 25,060 lines.
- Diff: +53 / -1,874, net **-1,821** (budget: at most -1,700).
- `tasks/trajopt.py`: +19 / -42, net -23 (new-code budget: at most 40).

## Deviations and findings

1. The caller evidence said all `CostStack` helper methods had zero callers.
   `examples/05_panda_trajopt.py` called `total_dim()`, while the deleted
   `LeastSquaresProblem` called `residual()`, `jacobian()`, and `gradient()`
   inside the same deletion closure. The example was mechanically converted;
   no surviving runtime caller was found.
2. The exact contract allowlist omitted `tests/contract/test_solver_state.py`
   and referred only to `test_solver_state.py` in T4. That file was entirely
   about the deleted type, so T4's basename was treated as authorization to
   delete it. This is the only contract-file authorization deviation.
3. Two wholesale test candidates contained live, independent coverage. Those
   tests were moved rather than deleted, as described above.
4. `test_roadmap_stub_inventory.py` did not itself name `optim/state.py`; it
   computes the inventory dynamically. The truthful change was therefore in
   `docs/reference/roadmap.md`, not that authorized contract file.
5. `test_matrix_free.py` covered deleted `LeastSquaresProblem` gradient helpers,
   not legacy Adam as the work order suggested. Its subject still died, so the
   disposition did not change.
6. Direct `ResidualItem` construction enforces matching item/residual names
   and a positive static dimension. The example therefore supplies names and
   `horizon` before wrapping acceleration/time-indexed residuals.
7. Two inert historical references remain outside the runtime/public-doc
   surface: `test_hot_path_lint.py` still watches the now-absent
   `optim/optimizers/` directory, and an M6 benchmark evidence JSON mentions
   `GraphExecutor`. The contract file was outside the exact edit allowlist;
   the benchmark statement is preserved evidence rather than an API claim.

CUDA-marked tests were not run, per the standing owner-run rule. No work from
order 02 or later was started.
