# Order 01 results — objective algebra

Status: complete on `dev` (2026-07-22).

## Delivered

- `Residual.weight` is a non-negative outer coefficient; `row_weight` owns square-root-information whitening. `sum`, `mean`, detached `mean_active`, per-group activity, and layout-stable `enabled` state are implemented.
- One node-free evaluation bundle captures rows, activity, effective coefficients, per-group costs, per-term costs, and the total inside one evaluation scope. `Problem`, dense and structured LM, robust decrease, and implicit reconstruction consume it.
- `Problem.error()` and task result residuals expose whitened rows; `Problem.term_costs()` exposes named objective contributions.
- LM uses fixed-active-set uncorrected IRLS with row scale `sqrt(active * weight * norm * kernel_weight)`. Eligible implicit differentiation handles outer weights and `mean`; `mean_active` and custom activity hooks fail actionably.
- Huber is gradient-safe at exact-zero and masked rows. Float32 zero-row gradchecks cover L2, Huber, Cauchy, Tukey, and Geman–McClure.
- All built-in residual constructors expose `row_weight`; the decorator, `Difference`, and `TimeIndexedResidual` forward the full algebraic state. Documentation, generated API pages, examples, task result contracts, and `MIGRATION.md` were updated.

## Verification

| Gate | Result |
|---|---|
| Full CPU, `pytest tests/ -q -m "not bench and not cuda"` | 1,602 passed, 2 skipped, 50 deselected |
| CUDA, GPU 2, `pytest tests/ -q -m cuda` | 49 passed, 1,605 deselected |
| Focused post-format objective/implicit/problem/temporal suite | 102 passed |
| Documentation contracts and snippets | 26 passed |
| Strict Sphinx HTML (`-W --keep-going -E`) | Passed |
| Sphinx doctest (`-W --keep-going -E`) | 30 passed, 0 failed |
| Ruff and Ruff format, all 34 changed Python files | Passed |
| `git diff --check` | Passed |

Source accounting by the plan's `wc -l` method: **22,337 → 22,622** lines. The source diff is 428 additions and 143 deletions, net **+285**, within the `+300` hard cap and the requested 250–310 target.

## Full weight migration

| Call site | Old | New | Intent / outcome |
|---|---|---|---|
| `src/better_robot/tasks/ik.py`, pose term | `weight=pose_weight` | `weight=pose_weight**2` | Preserve prior L2 row-scale tuning. |
| `src/better_robot/tasks/ik.py`, limit term | `weight=limit_weight` | `weight=limit_weight**2` | Preserve prior L2 row-scale tuning. |
| `src/better_robot/tasks/ik.py`, rest term | `weight=rest_weight` | `weight=rest_weight**2` | Preserve prior L2 row-scale tuning. |
| `src/better_robot/tasks/ik.py`, refinement disable/restore | Save, zero, and restore `weight` | Save, clear, and restore `enabled` | Preserve configured importance and optimizer layout/state. |
| `src/better_robot/tasks/contact_forces.py`, base wrench | `weight=sqrt(base_wrench)` | `weight=base_wrench` | The public value was already an objective coefficient. |
| Same, force magnitude | `weight=sqrt(force_magnitude)` | `weight=force_magnitude` | Same. |
| Same, force smoothness | `weight=sqrt(force_smooth)` | `weight=force_smooth` | Same. |
| Same, torque smoothness | `weight=sqrt(torque_smooth)` | `weight=torque_smooth` | Same; the square-root helper was deleted. |
| `examples/05_panda_trajopt.py`, start pose | `1000` | `1_000_000` | Preserve L2 contribution. |
| Same, goal pose | `100` | `10_000` | Preserve L2 contribution. |
| Same, acceleration | `0.1` | `0.01` | Preserve L2 contribution. |
| Same, joint limits | `10` | `100` | Preserve L2 contribution. |
| `tests/bench/bench_trajopt_sparse.py`, velocity | `0.05` | `0.0025` | Preserve L2 benchmark objective. |
| Same, acceleration | `0.005` | `0.000025` | Preserve L2 benchmark objective. |
| Same, reference | `0.01` | `0.0001` | Preserve L2 benchmark objective. |
| Same, envelope | `0.10` | `0.01` | Preserve L2 benchmark objective. |
| `tests/optim/solver_quality_support.py`, limits | `0.1` | `0.01` | Preserve L2 solve behavior. |
| Same, rest | `0.01` | `0.0001` | Preserve L2 solve behavior. |
| `tests/optim/slice_support.py`, root penetration | `0.25` | `0.0625` | Preserve L2 solve behavior. |
| Same, root scale prior | `sqrt(0.6)` | `0.6` | Preserve the intended objective coefficient. |
| Same, full penetration | `0.4` | `0.16` | Preserve L2 solve behavior. |
| Same, full clearance | `0.15` | `0.0225` | Preserve L2 solve behavior. |
| Same, full scale prior | `0.5` | `0.25` | Preserve L2 solve behavior. |
| `tests/optim/test_implicit_diff.py`, identity-only prior | `0.5` | `0.25` | Preserve L2 objective. |
| Same, robust-optimality prior | `0.5` | `0.25` | Preserve the prior's L2 contribution. |
| `tests/optim/test_problem.py`, analytic residual | `weight=2.0` | `row_weight=2.0` | Keep the whitening/Jacobian-scaling assertion. |
| Same, graph-visible zero | `ScaleWeight(tensor(0))` | outer `weight=tensor(0)` | Exercise tensor-zero evaluation without a host skip. |
| `tests/optim/test_residual_base.py`, scale object | `weight=ScaleWeight([2,3])` | `row_weight=ScaleWeight([2,3])` | Weight objects remain whitening operators. |
| Same, diagonal object | `weight=DiagonalWeight([2,4])` | `row_weight=DiagonalWeight([2,4])` | Same. |
| Same, raw tensor and mutation validation | `weight=...` | `row_weight=...` | Keep row-shape and dtype/device contracts. |
| `tests/optim/test_temporal_structure.py`, difference residual | `weight=1.7` | `row_weight=1.7` | Preserve dense/structured row and Jacobian scale. |
| `tests/residuals/test_pose_limits.py`, pose/position-limit/velocity-limit | `weight=0.25 / 0.4 / 0.2` | same values via `row_weight` | Preserve whitening assertions. |
| `tests/residuals/test_smoothness.py`, velocity | `weight=0.25` | `row_weight=0.25` | Preserve whitening behavior. |
| `tests/residuals/test_temporal_structure.py`, velocity/acceleration/reference/reference/rest/contact/contact | `weight=0.7 / 0.4 / 0.3 / 0.3 / 0.5 / 0.3 / 0.6` | same values via `row_weight` | Preserve analytic, finite-difference, and temporal parity. |
| `tests/tasks/test_ik_block_rebase.py`, position/orientation/limits/rest | `weight=0.6 / 0.8 / 0.4 / 0.2` | same values via `row_weight` | Preserve the row-level object protocol test. |
| `tests/optim/test_ad_strategies.py`, batched tensor | Tensor row multiplier | Same tensor as outer `weight` | Deliberate new semantic test: the public Jacobian is no longer importance-scaled. |
| Same, finite-difference graph-release tensor | `weight=tensor(1.5)` | unchanged outer `weight` | Only graph-release behavior is under test; outer weight remains graph-visible. |
| `tests/optim/test_problem_robust_gradient.py` | `1.7` inside the kernel argument | `1.7` outside the kernel | Deliberate robust-algebra test. |
| Unit and Python-zero declarations in `src/`, `tests/`, and examples | `1` / `0` | unchanged | Unit terms are numerically identical; Python zero remains a static skip. |
| Domain multipliers: pose position/orientation, per-joint, per-frame, contact masks, projection observations, Chamfer vertices, scene confidence | Domain/row multipliers | unchanged | These are not `Residual.weight`; they continue to act inside residual construction or whitening. |

## Deviations and findings

- **Required robust semantic change:** exact old non-L2 behavior cannot coexist with the accepted decoupling. Old `weight=a` produced `rho(a² s)`; migrated importance produces `a² rho(s)`. IK preserves its L2 tuning, while robust kernels now keep an independent outlier scale as the plan requires.
- `ContactForceResult.residual` also forwards `Problem.error()`. The plan explicitly named IK and trajectory results; contact-force result documentation was updated too so all public task results consistently promise whitened rows.
- Built-in residual constructors explicitly gained `row_weight`. `reduce` and `enabled` remain writable inherited `Residual` properties unless a constructor already forwards the full base surface; this avoids repetitive forwarding and stays within the source budget.
- Repository-wide Ruff currently reports 112 violations, all in unchanged files. The 34 changed Python files pass both Ruff checks. Optional nitpicky Sphinx mode likewise exposes 58 pre-existing unresolved cross-references; the project's strict non-nitpicky HTML build and doctest build pass with warnings treated as errors.

No other implementation deviations from Order 01 were found in the final adversarial audit.
