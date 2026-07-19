# 01 — Optimization API v2

> **Implementation log (2026-07-19):** Complete: `1574 passed, 2 skipped,
> 15 deselected`; parity `135 passed`; contracts `352 passed`. Final size is
> `optim/` 3,777 lines, `residuals/` -1 and `tasks/` -139 versus baseline;
> `optimizers.py` is 251 lines. All 93 changed Python files pass Ruff/format;
> the repository retains 140 lint findings and 55 unformatted untouched files.

**Goal:** rebuild `better_robot.optim` around the object model in
`plan/01_target_api.md`: standalone `Variable` objects, `Residual` classes
holding variable references, a `Weight` hierarchy applied identically to
error and Jacobian, `Optimizer` classes with `step()`/`optimize()`, and the
optimizer/solver vocabulary split. Rewire the residual library and the three
task facades. Read the target-API document first — it is the specification;
this order adds mechanics, evidence, and sequencing.

**Why now (verified 2026-07-19, adversarially re-checked):** residuals read
variables through 26 dict-keyed `ctx[...]` sites; `Values` is a bare
`dict[str, Tensor]` alias threaded beside a 22-field `LMState`;
`Problem.add_*` re-runs the whole constructor per call (`problem.py:222-231`);
weight application is duplicated across five sites; two near-identical
callable adapters exist (`problem.py:65-77`, `trajopt.py:43-71`); the public
`autograd.py` wrappers and `Problem.external_parameters` have test-only
callers (the private `_tangent_value_and_grad` machinery stays).

**Size budget:** `optim/` ≤ 3,800 lines (hard — today it is 3,742, so there
are 58 lines of headroom and the new classes must be paid for by real
deletions). **First deliverable of this order is a per-module before/after
line budget** (current 13 modules → target modules), written into the results
file before implementation starts. Much of the "deleted" list relocates
rather than disappears (manifold math → variable classes, first-order
mechanics → `TorchOptimizer`, providers → nodes); budget honestly. If the
table cannot reach ≤ 3,800, stop and report. `residuals/` and `tasks/` net
≤ +0 each.

**Behavior contract:** `solve_ik`, `solve_trajopt`, `solve_contact_forces`
keep their solution-quality regressions (same solutions, same tolerances).
LM's numerical behavior (damping schedule and gain-ratio escalation,
acceptance, step limits, bounds KKT, robust grouped IRLS, banded routing,
implicit guards, full-graph compile property) must not change: the pure
per-iteration step is being made private and re-hosted, not re-derived.
Tests pinning the removed *public surface* (`LMState` field access,
`run/update/finalize` signatures, `run_first_order`, provider `reads`
semantics) are removed or rewritten — but **only the public-field assertions
may be deleted outright**; numerical coverage those tests carry (gain-ratio
damping `test_solver_lm_damping.py:166`, step/decrease termination `:265`,
step limits `test_solver_lm_step_limits.py:98`, compile
`test_solver_lm_pattern.py:199`) is rewritten against the private step or
observable outputs, never dropped. List every removed/rewritten test in the
results file.

**Contract-test authorization:** `tests/contract/test_public_api.py`,
`test_submodule_public_imports.py`, `test_docstrings.py`,
`test_protocols.py`, `test_pluggable_protocols.py`, `test_hot_path_lint.py`
(module paths), `test_boundary_validation_count.py` (entry-point names). No
parity changes. **No layer-dependency changes** — see T2 for how the
`Residual` ABC respects the existing ranking.

Work in the listed order; each task ends green (the gate may run with the
previous task's API until T5 completes the rewire — sequence commits so the
tree is never red at a task boundary).

---

## T1 — Variables

Rewrite `optim/variables.py` per target-API §2: `Variable` base (Euclidean;
owns `tensor`/`name`/`trainable`/`bounds`/`mask`/`scale`/`batch_ndim`/
`time_axis`), `SO3Variable`, `SE3Variable`, `RobotVariable`. Fold the
`Manifold` classes' math into the subclasses (`manifolds.py` keeps `Bounds`
and shared private helpers; delete the public protocol and classes, ledger
rows). The `_JointCoordinateLayout` machinery inside today's `RobotConfig`
moves into `RobotVariable` unchanged — it is correct and tested.

Decisions fixed here so the code has one answer:

- `time_axis=0` keeps today's `VarSpec.time_axis` semantics exactly
  (knot-major event `(T, width)`, separable masks, temporal tangent width —
  `variables.py:98-128` relocates). A trajectory is
  `RobotVariable(model, q_traj, time_axis=0)` with event `(T, nq)`.
- Event shape: typed variables know their trailing feature width (`4`/`7`/
  `nq`), extended by the time axis when declared; plain `Variable` treats its
  constructor tensor's full shape as the event unless `batch_ndim` declares
  leading batch axes.
- Names: auto-generate `f"{type(self).__name__.lower()}_{counter}"` when
  omitted; `Problem` rejects duplicate names at freeze with a helpful error.
- `trainable=False` variables are readable and updatable by name, contribute
  no tangent columns, and are the declared gradient route for implicit
  differentiation (T4). Bare tensors passed to residual constructors stay
  bare tensors (no implicit wrapping).
- Validation lives in `__init__` (structural: tensor-ness, event shape,
  bounds/mask/scale shapes and dtypes — the checks `VarSpec.__post_init__`
  does today), not per evaluation. `Problem.update` re-validates fed tensors
  (public boundary) and applies the invalidation semantics of target-API §1.

## T2 — Residual base, weights, adapter

The `Residual` ABC and `Weight` hierarchy live in **`residuals/base.py`** —
the layer contract ranks `residuals` below `optim`
(`test_layer_dependencies.py:23,107`) and stays untouched. `base.py` imports
no optimizer types: variables are held through a small local protocol
(`.tensor`/`.name`/`.trainable`), and `kernel` is stored uninterpreted
(`None` means L2; `optim` owns kernel semantics). `better_robot.optim`
re-exports `Residual`, `Weight`, `ScaleWeight`, `DiagonalWeight`, and the
`residual` adapter so user code imports from one place.

- `Residual.__init__(*variables, dim, weight=1.0, kernel=None, group_size=1,
  name=None)` registers variable references in block order. `group_size`
  keeps `ResidualItem`'s validated grouped-IRLS semantics — production
  callers need groups of 6 and 3 (`contact_forces.py:334`) — with the same
  positive-divisor validation. `error()` abstract; `jacobian()` returns
  `None` by default (autodiff via the jacrev-over-tangent machinery,
  relocated from `problem.py`); analytic overrides return one block per
  trainable variable in reduced tangent coordinates — the contract
  `jacobian_blocks` has today, renamed.
- `Weight` / `ScaleWeight` / `DiagonalWeight` with `apply(error)` and
  `apply_jacobian(blocks)`; floats and tensors auto-wrap by shape. Replace
  all five `_broadcast_weight` sites (`problem.py:369,486,530,580`,
  `temporal.py:407`) and `_robustify`'s separate row-scale path so weighting
  has exactly one implementation. Keep the grouped robust-kernel semantics
  bit-identical (the kernel consumes the *weighted* squared norm exactly as
  today — verify against `test_kernel_rho_weight_consistency`).
- Trial evaluation: implement the swap-and-restore on `Problem` with the
  epoch rule of target-API §3 — **an epoch is one exact tensor assignment**;
  node memos invalidate on every assignment (every LM candidate, every AD
  closure call, every finite-difference sign — today each gets a fresh
  context, `problem.py:481,517`, and the rewrite must not weaken that);
  restoration on every exit path including exceptions. Add focused tests:
  jacrev, jacfwd, finite-difference, exception restore, graph release, and
  keep the compile property green.
- The `@residual(*variables, dim=)` decorator/adapter replaces
  `_CallableResidual` and `_ResidualAdapter`. Keep attribute pass-through
  only for the hooks that still exist (temporal declarations).
- `Difference(var, target, *, weight=1.0, kernel=None)` generic residual
  (manifold-aware via `var.difference`).

## T3 — Nodes replace providers

Per target-API §3: `RobotState(q)` computes exactly what
`RobotStateProvider` computes today — FK with frames (`providers.py:89`);
Jacobian-bearing quantities stay lazy in the residuals that need them
(`pose.py:150`) so cost-only evaluation does not get more expensive.
Identity-merging is automatic **only for `RobotState`** (variable + model
identity); `SceneSDFState` and the contact-dynamics node
(`scene_sdf.py:29`, `contact_forces.py:78,324`) have too many inputs for an
honest key and share only by passing the same node object. Residual
constructors accept a node or a bare `RobotVariable` (bare → private node,
merged at freeze). Memoization is per evaluation epoch (T2 rule);
graph-bearing outputs never survive an epoch. Delete `providers.py`,
`EvaluationContext`, `reads`, and the auto-provider block
(`problem.py:127-144`). Variable dependency for Jacobian structure = union
of direct variable refs and refs reachable through nodes, computed at freeze
(replaces reads propagation — port the transitive-dependency test
`test_providers.py:298`, the inactive-weight laziness test `:126`, and the
graph-lifetime test `:262` to node equivalents).

## T4 — Optimizers and solvers

- `optimizers.py`: `Optimizer` ABC (`step`/`optimize`/`reset`),
  `OptimizerInfo` (per-element `status`, `iterations`, `cost`, derived
  `converged` property — nothing stored that nothing reads), and
  `TorchOptimizer` (subsumes `first_order.py`; the persistent tangent-buffer
  + retract + rebase mechanics and per-element convergence mask move intact —
  ~70 lines relocate, budget accordingly). Closure-free torch optimizers are
  the supported family. `torch.optim.LBFGS` needs a closure evaluated
  multiple times per step with one global history (`lbfgs.py:325,350,379`):
  implement the closure path over the summed objective with the documented
  batch-coupling caveat, add a solution-quality test — and if that test
  cannot meet the IK regression bar, keep today's honest error for the
  `"lbfgs"`/`"lm_then_lbfgs"` spellings and record the evidence.
- `lm.py`: `LevenbergMarquardt(problem, *, solver="auto", max_iterations,
  tolerance, ...)`. The existing `update` becomes the private pure step; the
  22-field `LMState` splits into (a) slim public `OptimizerInfo`, (b) a
  private per-iteration state carrying **everything the step reads or
  carries forward** (`mu`, `increase_factor`, `residual`, `robust_weights`,
  `cost`, per-element status/iterations, the carry-forward diagnostics at
  `lm.py:764-775`, and the implicit-diff fields `gradient`/`active_mask`/
  `projected_grad_norm`/`implicit_valid` that `implicit.py:158-171`
  consumes), and (c) a frozen static-layout context built once at freeze
  (`scale`, `bound_state_index`, `bound_lower/upper`, `bounded_mask`). What
  dies is the *public exposure* of diagnostic fields — the numerical
  behavior they participate in is pinned by the rewritten tests (see
  behavior contract).
- Implicit differentiation: entry moves to
  `optimize(differentiate="implicit")`. The custom-backward inputs become the
  harvested graph-carrying static variables (target-API §2) — today's
  explicit-parameter registration (`ik.py:271,306` → `implicit.py:442,462`)
  maps onto `Variable(trainable=False)` wrapping in `solve_ik`. All 23
  implicit-diff tests stay green (mechanical renames plus the parameter-route
  change only).
- `verbose=True` prints one line per iteration from `optimize()` only.
- `solvers.py`: add dense `LU` via `torch.linalg.lu_factor`/`lu_solve` with
  the same `LinearSolveResult` reporting; `LinearSolver` protocol is the
  parent. Wire `solver=` acceptance (`"auto"`, instance) and keep the auto
  dense/banded routing decision logic as is.

## T5 — Rewire the library, the tasks, and the temporal plumbing

- `residuals/`: every residual class takes its variables/nodes at
  construction and subclasses `Residual` (26 `ctx[...]` sites across
  `pose.py`, `limits.py`, `contact.py`, `regularization.py`, `scene_sdf.py`,
  `chamfer.py`, `temporal.py`, `projection.py`, `base.py`).
- **Temporal is an interface rewrite, not a rename** — only the banded
  matrix/solver mathematics is unchanged. Re-target: temporal analysis and
  structured assembly off `problem.vars`/`VarSpec.time_axis`/item reads
  (`temporal.py:195,334`) onto variables and the freeze-computed dependency
  map; `TimeIndexedResidual`'s context wrapping, `Data` slicing, and reduced
  index rewriting (`residuals/temporal.py:26,73`) onto nodes; route/reason
  metadata reporting stays.
- `tasks/ik.py`: `solve_ik` keeps its six optimizer spellings (`lm`, `gn`,
  `adam`, `lbfgs`, `lm_then_adam`, `lm_then_lbfgs`, `ik.py:38`) — the torch
  spellings construct `TorchOptimizer` per T4's evidence rule. Config
  validation (rejecting unused knobs and invalid combinations, `ik.py:110`)
  survives. `lm_then_adam` becomes two sequential optimizers with
  **snapshot/set/restore weights in `try/finally`** — final diagnostics
  evaluate at the original full weights (`ik.py:209,389` semantics), so
  mutate-and-forget is wrong. Targets and `q_rest` wrap as static variables
  (T4). The `_state_iterations` hasattr ladder (`ik.py:181-186`) dies.
- `tasks/trajopt.py`: the facade (or the caller) constructs the trajectory
  `RobotVariable(model, initial_q_traj, time_axis=0)`; residuals reference
  it; horizon derives from it. The `optimizer=` parameter becomes an
  optimizer **factory** (`Callable[[Problem], Optimizer]`, default
  `LevenbergMarquardt`) — a pre-built optimizer cannot precede the problem.
  Status and route/reason result metadata (`trajopt.py:324`) survive.
- `tasks/contact_forces.py`: preserve the objective-coefficient →
  sqrt-row-multiplier conversion (`contact_forces.py:176`), the explicit
  re-evaluation of dynamics at the solved forces (`:376`), and the gradient
  flow that re-evaluation provides (pinned at `test_contact_forces.py:92`).
- Migrate the unscheduled in-repo consumers the gate will catch:
  `examples/05_panda_trajopt.py` (imports `ResidualItem`, builds LM without a
  problem — `05_panda_trajopt.py:27,53,149`), the normal-suite benchmark
  smoke `tests/bench/test_trajopt_sparse_smoke.py:20` and its harness
  `bench_trajopt_sparse.py` (uses `VarSpec`/`ResidualItem`/provider/lifecycle
  APIs at `:183,355,365`), and `tests/optim/slice_support.py:13,50,221`
  (imports the old surface and exact-executes the custom-residual guide —
  update the guide and the extractor together).

## T6 — Truth sweep and accounting

- Update every falsified statement: `optim/CLAUDE.md`, **root `CLAUDE.md`**
  (`:63` describes manifolds/providers/`run_first_order`),
  `docs/concepts/residuals_costs_and_solvers.md`,
  `docs/conventions/extension.md:52`, `docs/conventions/performance.md:74`,
  `docs/guides/custom_residual.md` (+ its exact-extraction test),
  `own_your_optimization_loop.md`, `differentiate_through_kinematics.md`,
  tutorial 03, the front-page example (its extractor accepts only a literal
  ` ```python ` fence — `tests/docs/test_front_page.py:9` — update them
  together), snippet-count expectations
  (`tests/docs/test_tutorial_snippets.py:17`), and the generated API pages.
  Order 03 restyles; this order makes pages *true*.
- Rename `tests/optim/test_solver_adam_matrix_free.py` →
  `test_torch_optimizer.py` (its content is already generic factories).
- MIGRATION.md rows for every target-API §6 removal.

## Acceptance

- The line-fit, IK, and trajectory examples in `plan/01_target_api.md` §1
  run exactly as written (add them as doctests or snippet-harness pages).
- `grep -rn "ctx\[" src/better_robot/residuals/` → 0;
  `grep -rn "VarSpec\|EvaluationContext\|ResidualItem\|LMState\|run_first_order" src/ examples/ tests/bench/` → 0.
- The per-module budget table exists in the results file; `optim/` line count
  reported; ≤ 3,800.
- Full gate, parity, contracts (authorized updates only), ruff, Sphinx HTML +
  doctest all green; solution-quality regressions for the three tasks
  unchanged; implicit-diff suite green.
- Results file per standing rules, including the deleted/rewritten-test
  disposition.
