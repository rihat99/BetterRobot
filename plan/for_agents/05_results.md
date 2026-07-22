# Order 05 results — optimizer phases, docs, and vertical slice v2

Status: Order 05 scope complete on `dev` (2026-07-22). Repository-wide gates
are temporarily blocked only by concurrent Warp FK/RNEA work outside this
order.

## Delivered

- `TorchOptimizer` now keys state refresh to the actual trainable layout:
  variable names/free dimensions, batch shape, dtype, and device. Same-layout
  static or target updates preserve tangent buffers, Adam moments, and the
  optional scheduler; real layout changes rebuild them.
- The shared `Optimizer` lifecycle exposes `resume()`. First-order resume keeps
  values, buffers, moments, scheduler state, and cumulative counts. LM resume
  keeps compatible values/counts while rebuilding damping and acceptance state
  for the current objective.
- `TorchOptimizer(..., scheduler=factory)` owns a standard PyTorch
  `LRScheduler`, advances it exactly once after an outer update (including
  L-BFGS), and never advances it on terminal or converged-before-update paths.
- The staged-fit how-to uses a separate frozen-layout warm-up, one Problem and
  scheduled Adam instance across two plain-data phases, static input updates,
  `enabled`/weight changes, `resume()`, `term_costs()`, and a new L-BFGS polish
  adapter. It explains why BetterRobot has no Phase object.
- Vertical slice v2 fits a synthetic free-flyer-plus-hinge body through nested
  counted Nodes, scene attraction, masked Chamfer, point projection,
  `mean_active`, outer weights, a `ScalarCost` prior, a frozen-root warm-up, and
  two resumed Adam phases. It reaches the numeric target with exactly 211 calls
  per shared node; its phase-driving test is 90 lines.
- Optimizer contracts, concepts, migration guidance, changelog, generated API,
  and executable docs were synchronized. The roadmap and residual package
  contract already contained Order 04's final deferrals and needed no edit.

## Verification

| Gate | Result |
|---|---|
| Focused optimizer lifecycle + vertical slice | 14 passed |
| Complete optimizer CPU suite | 344 passed, 5 deselected |
| Optimizer CUDA suite, GPU 2 | 5 passed, 344 deselected |
| Documentation snippet/content contracts | 27 passed |
| Warnings-as-errors Sphinx HTML (`-E -W --keep-going`) | Passed |
| Sphinx doctest (`-W --keep-going`) | 31 passed, 0 failed |
| Ruff, Ruff format, and `git diff --check` | Passed |
| Full CPU tree | 1,661 passed, 2 skipped, 61 deselected; 4 unrelated Warp FK failures |
| Full CUDA tree, GPU 2 | 51 passed, 1,668 deselected; 9 unrelated Warp FK/RNEA failures |

By the required `wc -l` accounting, the two changed Python source files grew
from 1,183 to 1,238 lines: net **+55**, below the Order 05 cap of +150. The two
vertical-slice files are 427 lines total and their replacement is net **-92**
lines (330 additions, 422 deletions).

## Wishlist disposition

| # | Wishlist request | Disposition | In-tree pointer |
|---|---|---|---|
| 1 | Nodes compose | **Shipped.** Nodes form recursively scoped DAGs, and shipped mesh residuals consume computed point Nodes. | `residuals/nodes.py`; vertical slice v2 |
| 2 | Plain term multiplier and mean reduction | **Shipped.** Outer `weight`, independent `row_weight`, and `sum`/`mean`/`mean_active` share one objective path. | `optim/problem.py`; Order 01 tests |
| 3 | Per-row gates and active-row mean | **Shipped.** Detached `active_groups()` and scene/projection gates drive `mean_active`. | `residuals/scene_sdf.py`; Order 03 tests |
| 4 | Mutable residual inputs | **Shipped via the accepted mechanism.** Named static Variables update atomically through `Problem.update()`; same-layout first-order state now survives. | `optim/problem.py`; optimizer lifecycle tests |
| 5 | Phase runner, LR schedule, and on/off terms | **Partially shipped / partly rejected.** Scheduler, `enabled`, layout-aware refresh, and `resume()` ship. A Phase/curriculum object is intentionally rejected in favor of plain application data. | `guides/staged_fit.md`; `optim/optimizers.py` |
| 6 | Scalar objective term | **Partially shipped.** `ScalarCost` contributes exactly `weight * f` through the common residual path. A second Problem scalar pathway and variable-tracking kernel scale are intentionally rejected. | `residuals/base.py`; Order 02 tests |
| 7 | Per-block robot freeze and weighting | **Shipped with an immutable layout.** Named tangent groups, `frozen_groups`, and `tangent_weight()` cover the generic need; another frozen set requires another Variable/Problem. | `optim/variables.py`; Order 04 tests |
| 8 | Higher-order block-weighted smoothness | **Partially shipped.** Orders 2--4 and coordinate weights ship. Automatic short-trajectory skipping is rejected; state-dependent manifold blocks remain deferred. | `residuals/smoothness.py`; roadmap |
| 9 | Per-term logging | **Shipped at the optimizer boundary.** `Problem.term_costs()` reports the objective contributions. A broader application metrics framework is intentionally out of scope. | `optim/problem.py`; staged-fit guide |

## Deviations and findings

- The old vertical slice's guide-fence execution and batched-parity cases were
  replaced by the accepted BVR-shaped consumer. Documentation contracts and the
  residual batching/Jacobian suites already own those behaviors, so this slice
  stays focused on composition across Orders 01--05.
- The vertical fit intentionally exhausts fixed phase budgets and reports
  MAXITER while meeting strict numerical convergence thresholds; lifecycle
  convergence after `resume()` is separately pinned by focused tests.
- Metric-driven schedulers such as `ReduceLROnPlateau` are outside the accepted
  no-argument `LRScheduler.step()` contract. No BetterRobot schedule DSL or
  milestone type was added.
- A clean nitpicky Sphinx audit still reports the repository's existing
  unresolved public-reexport/private-type references. The established
  warnings-as-errors HTML gate and every executable documentation example are
  green; this order adds no private scheduler alias to that debt.
- The required full-tree CPU/CUDA runs are not green on the shared worktree:
  their 4 CPU and 9 CUDA failures are confined to concurrent Warp FK/RNEA
  changes. Per the user's scope correction, Order 05 did not inspect, modify,
  stage, or weaken those implementations/tests.

No other deviations from the accepted Order 05 semantics remain.
