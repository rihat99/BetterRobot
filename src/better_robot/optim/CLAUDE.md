# `optim/` — Nonlinear Least Squares

This package has one graph representation: object-owned variables referenced
by fixed-width residuals and harvested into a `Problem`. LM/GN and
`TorchOptimizer` consume that same graph; task helpers assemble it rather than
defining another optimizer protocol.

## Variables and geometry

`Variable` owns its current tensor, stable name, trainable/static role, event
shape, optional `Bounds`, batch declaration, and optional knot-major
`time_axis=0`. `SO3Variable`, `SE3Variable`, and
`RobotVariable` own their corresponding retract/difference geometry. There is
no separate public variable specification or manifold object.

Trainable Variables use float32/float64. Static Variables may also hold bool or
integer masks and labels; they never enter tangent or retraction paths. All
Variables in a Problem share one device, while only floating Variables must
share a dtype. Replace graph inputs through `Problem.update()`. Bare tensors
accepted by residual or node constructors are construction-time constants,
not harvested Variables.

Leading axes are independent execution batches. Every trainable tangent
coordinate participates in derivatives and linear systems unless a
`RobotVariable` excludes topology-derived `frozen_groups` at construction.
Its public difference remains full-width; only optimizer-facing gather and
expand hooks use free coordinates. A trajectory repeats one immutable group
mask at every knot. Retraction projects supported bounds in state space;
bounds are not tangent step limits, and bounds on frozen coordinates are not
active columns. Validate structural compatibility at graph freeze or solve
entry, then let numerical non-finites produce honest optimizer status.

## Residuals, nodes, and problems

A `Residual` holds ordered references to every variable it reads, a positive
static `dim`, `name`, outer `weight`, square-root-information `row_weight`,
`reduce`, `enabled`, robust `kernel`, and `group_size`. `error()` returns raw
`(..., dim)` rows. Optional `jacobian()` blocks are ordered like the trainable
dependencies and use each variable's full tangent coordinates; `Problem`
centrally gathers construction-time free columns. Strategies are `auto`,
`analytic`, `jacrev`, `jacfwd`, and explicit debug-only
`finite_difference`; a malformed advertised analytic block is an error.

One evaluation bundle captures whitened rows, detached activity, effective
coefficients, and costs while node memos are valid. Every objective consumer
uses the same formula:

```text
rows = row_weight.apply(error())
cost = Σ_k active_k · w_k · ρ(‖rows_k‖²) · norm
```

`weight` is the non-negative outer coefficient `w_k`, independent of kernel
scale. `norm` is `1` for `sum`, `1 / n_groups` for `mean`, and
`1 / clamp(Σ_k active_k, 1)` for `mean_active`; the activity mask and active
count are detached. L2 uses `ρ(s) = 0.5 · s`, preserving the exact
`0.5 · Σ_k active_k · w_k · ‖rows_k‖² · norm` convention.

Evaluation-scoped `Node` objects own shared graph-bearing work such as
`RobotState`. A node may read Variables and child Nodes; its `nodes` tuple is
direct children and its `variables` tuple is the stable, deduplicated set of
transitive leaves. Residuals list direct nodes in `nodes`. `Problem` walks the
acyclic graph, merges compatible nodes at every depth, and scopes every node.
Never retain a node result across candidate values.

Under the `auto` strategy, a residual without analytic blocks emits
`AutodiffFallbackWarning` once per `Problem` and residual before using the
selected `torch.func` transform. Explicit Jacobian strategies do not warn.

`Problem` freezes on first use, validates unique names, and computes row and
tangent-column layouts from references. A Python-zero residual weight skips
its rows; `enabled=False` does the same without changing layout or optimizer
state, while tensor zero remains graph-visible. `active_groups()` is an
authoritative fixed-shape boolean mask. `Problem.error()` returns only
concatenated `row_weight`-whitened rows; use `objective()` for the scalar cost
and `term_costs()` for named contributions. Task-result `residual` fields copy
the same whitened diagnostic rows.

LM/GN use uncorrected IRLS, not a Triggs second-order correction. The group
row scale is
`sqrt(active_k · w_k · norm · kernel.weight(‖rows_k‖²))`, which is
gradient-consistent on a fixed active set. Detached masks and counts make an
activity threshold non-differentiable; never test it as a derivative point.

## Optimizers

`Optimizer` owns one `Problem`. Public control is `step()` for a caller-owned
loop, `optimize()` for the complete eager driver, and `reset()` to clear
optimizer state while retaining variable values. `OptimizerInfo` exposes only
per-element `status`, `iterations`, `cost`, and derived `converged`; solved
values live in the variables.

`LevenbergMarquardt` and `GaussNewton` preserve arbitrary leading batch axes
and per-element damping, acceptance, convergence, and status internally.
Dense routing uses `Cholesky` by default; directly eligible temporal graphs use
`BlockBandedMatrix` and `BandedCholesky`. Automatic routing must report a
stable `LinearizationReason`. A temporal declaration without numeric blocks
also emits `AutodiffFallbackWarning` once when automatic routing chooses dense;
explicit dense or structured routing does not warn.

Keep LM's private per-iteration tensor program input-pure, fixed-shape,
sync-free, and tensor-branching. `TorchOptimizer` owns persistent tangent
buffers, delegates update rules to `torch.optim`, retracts after each step, and
rebases without discarding optimizer state. Non-closure optimizers run one
objective forward per step (LBFGS line searches add their own closure
evaluations): `step()` reports cost from the gradient forward (the entering
iterate), and only `optimize()` refreshes the final cost at the solution.

## Differentiation

`optimize(differentiate="implicit")` attaches a guarded first-order backward
to an eligible detached solution. Differentiable inputs are graph-carrying
static `Variable` objects referenced by residuals or nodes. Keep guards for
convergence, active bounds, robust kinks, quaternion branch cuts, routing and
size limits, non-finite systems, and singular systems.

A residual using `reduce="mean_active"`, overriding `active_groups()`, or
adapting a scalar penalty through `ScalarCost` makes the problem
implicit-ineligible. Reject it actionably by residual name.

Direct public modules are `problem.py`, `variables.py`, `optimizers.py`,
`lm.py`, `implicit.py`, `kernels.py`, `solvers.py`, and `temporal.py`;
`Bounds` lives with variables and shared optimizer helpers live in `utils.py`.
Evaluation nodes live with residuals. Keep new behavior on this surface.
