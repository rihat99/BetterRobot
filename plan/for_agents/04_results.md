# Order 04 results — tangent groups, freezing, and smoothness

Status: complete on `dev` (2026-07-22).

## Delivered

- `RobotVariable` exposes topology-derived joint, `root`, `root_lin`, `root_ang`, and `joints` tangent groups. `tangent_weight()` builds documented square-root-information multipliers and rejects unknown or ambiguous names.
- Construction-time `frozen_groups` produces immutable knot-major free indices. Dense, bounded LM, TorchOptimizer, temporal/banded, and implicit paths operate only on free coordinates, while public robot differences remain full tangent values and frozen ambient coordinates remain exact.
- Two-phase root/joint optimization, bounds, analytic-column reduction, dense/banded agreement, full-tangent `Difference`, and CPU/CUDA implicit gradients have end-to-end coverage.
- `SmoothnessResidual` implements orders two through four from forward differences of manifold first differences, with exact `dt**-order` scaling and live `(nv,)` coordinate weights. `AccelerationResidual` was removed with migration guidance; `VelocityResidual` keeps its central-difference contract.
- Constant analytic and direct-banded blocks are retained only for all-scalar topologies. Non-scalar models emit one actionable warning and route to dense autodiff. Documentation, examples, generated API, roadmap, changelog, and benchmark metadata were updated.

## Verification

| Gate | Result |
|---|---|
| Full CPU, `pytest tests/ -q -m "not bench and not cuda"` | 1,660 passed, 2 skipped, 61 deselected |
| CUDA, GPU 2, `pytest tests/ -q -m cuda` | 59 passed, 1,663 deselected |
| Trajectory benchmark smoke suite | 4 passed |
| Focused smoothness audit | 32 passed, including finite CUDA order-four errors, Jacobians, and gradients |
| Documentation content/snippet rerun | 12 passed |
| Strict Sphinx HTML (`-W --keep-going -E`) | Passed |
| Sphinx doctest (`-W --keep-going -E`) | 30 passed, 0 failed |
| Ruff and Ruff format, changed Python files | Passed |
| Public import contract and `git diff --check` | Passed |

The Order 04 `src/` diff is 378 additions and 82 deletions, net **+296** lines; Python alone is net **+284**. This meets the requested 250--310 target and the plan's `+350` cap.

## Deviations and findings

- The required pre-build audit confirmed a released defect on a six-knot spherical trajectory: the former constant velocity block differed from AD by `1.27060699` (72 entries above `1e-3`), and the former acceleration/order-two block differed by `57.68396` (108 entries above `1e-3`). The accepted dense fallback fixes routing without introducing an unreviewed state-dependent log Jacobian.
- The canonical sparse benchmark uses free-flyer and spherical joints, so the honest topology gate makes its old structured lane invalid. That lane was retired instead of reporting a misleading comparison; the benchmark reports dense data and marks structured acceptance inconclusive until a reviewed all-scalar fixture is added. Separate tests still prove frozen dense/banded equivalence.
- `ReferenceTrajectoryResidual` appears to share the same constant-block risk on manifold models. Correcting it requires its own contract and is explicitly deferred in the roadmap rather than expanding this order.
- The deleted acceleration residual accepted arbitrary time-varying `row_weight`; the dedicated replacement intentionally narrows that surface to the planned `(nv,)` `coordinate_weight`. `MIGRATION.md` records the conversion and advises a general residual for other patterns.
- CUDA exposed that implicit backward could not read least-squares rank metadata. The implementation now validates `matrix_rank` and `solve_ex` status/residual for the square adjoint system; this was a necessary in-scope repair for the required CUDA frozen-coordinate guard.
- Reserved derived group names now raise on unequal joint-name collisions rather than silently replacing topology groups. A conventional free-flyer joint named `root` remains valid when its indices exactly match the derived root group.

No other deviations from the accepted Order 04 semantics remain.
