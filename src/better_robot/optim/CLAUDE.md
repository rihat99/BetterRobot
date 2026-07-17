# optim/ — Legacy Solvers and Named-Block Optimization

Two optimization surfaces coexist. Keep their types and capabilities separate.

## Status at a Glance

| Surface | Problem type | What is implemented |
|---|---|---|
| Legacy solver stack | `optim.problem.LeastSquaresProblem` | Flat-variable `CostStack`; LM/GN/Adam/LBFGS/MultiStage; `optim.solve`; trajopt integration |
| Named-block layer | `optim.blocks.problem.Problem` | Manifold blocks/providers/evaluation, matrix-free batched Adam, batched LM/GN, robust groups, and projected active-set bounds |

Named-block residual-vector problems use
`better_robot.optim.LevenbergMarquardt` or `GaussNewton`. Scalar
`ObjectiveItem`s remain for consumer-owned first-order/manual loops and are
strictly rejected by the second-order entry points.

## Named Variable Blocks

`VarSpec` describes one entry in the `Values = dict[str, Tensor]` mapping:

- `shape` is the state event shape; any leading axes in the value are
  independent evaluation batch axes.
- `manifold` is `Euclidean`, `SO3Manifold`, `SE3Manifold`, or
  `RobotConfig(model)`.
  Retractions use the repository's local/right perturbation convention.
- `mask` lives in flattened tangent space. Nonzero entries are free; fixed
  entries are **eliminated** from steps, gradients, Jacobian columns, and
  normal systems. They are not retained as zero columns.
- `scale` is positive finite tangent-coordinate metadata, gathered through the
  same mask and consumed by named-block LM/GN when forming scaled normal
  systems.

There are two different kinds of bounds:

- `optim.blocks.manifolds.Bounds` is a **state/configuration-space** box.
  Initial values are validated and rejected when infeasible; retracted values
  are projected back to feasibility. Quaternion/unit-circle coordinates are
  never clamped, and global boxes on `SO3Manifold`/`SE3Manifold` blocks are
  rejected.
- Trust regions, step clamps, and other **tangent-space** limits belong to a
  solver. The shipped active-set solver consumes state-space `Bounds`; it does
  not reinterpret them as tangent boxes.

The legacy `LeastSquaresProblem.lower/upper` pair is separate, older
projection-only solver behavior; it is not the named-block `Bounds` contract.

## Residuals, Scalar Objectives, and Dense Assembly

A block residual declares `name`, positive static `dim`, `reads`, and
`__call__(ctx) -> Tensor` with shape `(B..., dim)`. `ResidualItem.weight` is
applied during M2a evaluation. Optional `jacobian_blocks(ctx)` entries are keyed
by variable name and must already use mask-reduced tangent columns.

`ResidualItem.kernel` and `ResidualItem.group_size` are validated by `Problem`.
`Problem.objective()` sums grouped `rho`, and `Problem.gradient()`
differentiates that same robust objective under `weight(s) = 2*rho'(s)`.
Named-block LM/GN uses the matching group-wise IRLS rows. Direct
`Problem.residual()` remains the weighted raw residual, not an IRLS view.

`ObjectiveItem` represents a scalar term with output shape `(B...)`. Scalar
terms participate in `Problem.objective` and tangent-space `Problem.gradient`,
so they work in a consumer-owned first-order/manual loop. They are not silently
converted into least-squares residuals. `Problem.require_least_squares(...)`
rejects a problem containing scalar objectives with an exact error for
Gauss-Newton/Levenberg-Marquardt entry points.

Assembly is intentionally dense:

- variable order defines column offsets;
- residual-item order defines row offsets;
- an absent `(residual_name, variable_name)` block is structurally zero;
- `dense_jacobian` preallocates and fills the full reduced tangent matrix;
- a trajectory stored as one block remains one large dense block.

Symbolic temporal sparsity, banded storage/solvers, and Schur elimination are
M5 scope. Legacy `ResidualSpec` remains importable for compatibility but has no
production sparse solver consumer; do not attach it to new block residuals.

## AD Strategies

The block layer differentiates `f(values ⊕ delta)` at `delta = 0`:

- an optional analytic reduced block is preferred by `strategy="auto"`;
- otherwise auto selects `jacrev` when residual dimension is no larger than
  the free block dimension, and `jacfwd` in the opposite regime;
- a raising analytic implementation is an error, never a silent fallback;
- central finite differences are reachable only through the explicit
  `strategy="finite_difference"` debug path;
- `create_graph=True` preserves higher-order Torch graphs where supported.

## Providers and Evaluation Lifetime

Providers declare `inputs` and `outputs` and form a construction-time-validated
DAG. They are lazy: only outputs requested by an active item's declared
`reads` are computed. The item context is read-only and rejects undeclared
access.

Provider results are cached only inside one `EvaluationContext`. A residual,
objective, gradient, or analytic-Jacobian path shares one such context. Each
missing AD-generated `(residual, variable)` block gets a fresh transform-local
context, so its providers may run again. No cache is stored on `Problem` or
survives into the next iteration; graph-bearing FK/NN outputs therefore cannot
leak across evaluations. Accepted artifacts retained by a consumer must be
detached explicitly.

## Batching and Solver Boundary

Named-block residual, objective, gradient, Jacobian-block, and dense-assembly
methods accept arbitrary common leading batch axes and preserve them. Named-
block LM/GN preserves the same axes in per-element cost, damping, acceptance,
factorization, convergence, status, and iteration tensors. Elements that are
terminal or failed ride along without moving while their valid neighbors
continue.

Named-block `Adam` preserves those axes in its reduced per-block moments and
per-element step/cost/gradient/status leaves. Its update calls only the
prevalidated tangent objective VJP plus feasible retraction; Jacobian assembly
is forbidden. `run` is detached and warm starts retain moments/step counts only
across an exactly compatible named reduced layout. Batched named-block LBFGS is
deferred; do not port the scalar dense-J legacy history.

The legacy optimizers and legacy trajopt remain single-problem solvers.
`solve_ik` uses the named-block stack. Do not pass a block `Problem` to a
legacy optimizer merely because its evaluation methods are batched.

## Named-Block Solver Lifecycle

Use `init_state(values, problem)`, pure `update(values, state, problem)`, and
`finalize(values, state, problem)` for an external loop, or `run(...)` for the
detached eager driver. `finalize` ensures residuals, robust weights, active
masks, and KKT diagnostics all describe the returned point. A warm-start state
retains damping but refreshes target-dependent evaluation artifacts.

For the explicit small-problem unrolled oracle, pass `create_graph=True` to
`init_state`, each `update`, and `finalize`. The default calls do not retain
the Jacobian graph; `run` has no graph-preserving mode and always returns
detached values/state. This oracle is a differentiation regression aid, not a
stable implicit solver backward. M6 owns active-set validity and implicit
differentiation through a complete solve.

Warm-start damping is reusable only when batch shape, dtype, and device match
exactly. `mu_min` is strictly positive, and a factorization that keeps failing
gets one real attempt at `mu_max` before `FAILED`. Accepted-step `xtol`/`ftol`
termination is unbounded-only; bounded success always requires projected KKT.
Tolerance-only unbounded convergence stays `implicit_valid=False` unless the
final-point KKT evaluation also passes.

`LMState` is a fixed-structure tensor-only `NamedTuple`; solver
hyperparameters are frozen Python configuration. `update` must stay
fixed-shape, sync-free, input-pure, and tensor-branching only. Public/static
validation belongs to `init_state`; the eager `run` boundary may perform one
all-terminal host check per iteration. The module is capture-ready by the M2b
structural checklist, but only M6's actual CUDA capture/replay parity harness
may call it capture-certified. Custom residuals/providers that use dynamic
shapes, host syncs, or value-keyed Python caches remain eager-only.

Finite Euclidean/configuration bounds use a restricted active-set normal
system plus projected-gradient KKT termination. Finite world-axis boxes on
free-flyer translation are rejected because the solver tangent is right-local.
Do not weaken that rejection without a constraint-normal representation.

## Legacy Solver Stack

The existing stack remains:

```text
LeastSquaresProblem  ->  Optimizer  ->  LinearSolver
                         |              |
                         v              v
                    DampingStrategy   RobustKernel
```

- Optimizers: `LevenbergMarquardt`, `GaussNewton`, `Adam`, `LBFGS`,
  `MultiStageOptimizer`, and `LMThenLBFGS`.
- Linear solvers: dense batched `Cholesky` and `LSTSQ`, both with the
  `solve(A, b, ridge=None)` contract. Iterative and structured solvers wait
  for M5.
- Legacy damping: `Constant` and `Adaptive`; no placeholder strategies are
  exported.
- Kernels: `L2`, `Huber`, `Cauchy`, and `Tukey` with the legacy row-wise IRLS
  convention.

Legacy LM projects trial points to `lower/upper`, does not clamp the initial
point, and has no active-set/KKT treatment. It can finish as `maxiter` at an
active bound.

## Top-Level `solve()`

`better_robot.optim.solve(problem, ...)` is a convenience wrapper for the
legacy `LeastSquaresProblem` and defaults to legacy `LevenbergMarquardt`. It
does **not** dispatch `optim.blocks.Problem`, scalar block objectives, or
batched block problems. Named-block consumers call
`better_robot.optim.LevenbergMarquardt().run(values, problem)` (or
`GaussNewton`) directly.
