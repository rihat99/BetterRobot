# optim/ — Legacy Solvers and Named-Block Evaluation

Two optimization surfaces coexist. Keep their types and capabilities separate.

## Status at a Glance

| Surface | Problem type | What is implemented |
|---|---|---|
| Legacy solver stack | `optim.problem.LeastSquaresProblem` | Flat-variable `CostStack`; LM/GN/Adam/LBFGS/MultiStage; `optim.solve`; IK/trajopt task integration |
| Named-block layer | `optim.blocks.problem.Problem` | Manifold variable blocks, evaluation-local providers, residual/scalar evaluation, tangent gradients, analytic/`jacrev`/`jacfwd` blocks, deterministic dense assembly |

The named-block layer has no optimizer driver in M2a. A consumer keeps its
manual first-order loop and calls `Problem.objective`, `Problem.gradient`, and
`Problem.retract`. Batched second-order solving and per-element solver state are
M2b work.

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
  same mask. M2a exposes it for future damping/preconditioning but does not run
  a solver that consumes it.

There are two different kinds of bounds:

- `optim.blocks.manifolds.Bounds` is a **state/configuration-space** box.
  Initial values are validated and rejected when infeasible; retracted values
  are projected back to feasibility. Quaternion/unit-circle coordinates are
  never clamped, and global boxes on `SO3Manifold`/`SE3Manifold` blocks are
  rejected.
- Trust regions, step clamps, and other **tangent-space** bounds belong to a
  solver and are M2b scope. Never put them in `VarSpec` or `Bounds`.

The legacy `LeastSquaresProblem.lower/upper` pair is separate, older
projection-only solver behavior; it is not the named-block `Bounds` contract.

## Residuals, Scalar Objectives, and Dense Assembly

A block residual declares `name`, positive static `dim`, `reads`, and
`__call__(ctx) -> Tensor` with shape `(B..., dim)`. `ResidualItem.weight` is
applied during M2a evaluation. Optional `jacobian_blocks(ctx)` entries are keyed
by variable name and must already use mask-reduced tangent columns.

`ResidualItem.kernel` and `ResidualItem.group_size` are recorded and validated
for the M2b robust-normal-equation implementation. **M2a does not apply the
kernel or group-wise IRLS weights.** Do not describe a grouped kernel as active
until the M2b solver path exists.

`ObjectiveItem` represents a scalar term with output shape `(B...)`. Scalar
terms participate in `Problem.objective` and tangent-space `Problem.gradient`,
so they work in a consumer-owned first-order/manual loop. They are not silently
converted into least-squares residuals. `Problem.require_least_squares(...)`
rejects a problem containing scalar objectives with an exact error for
Gauss-Newton/Levenberg-Marquardt entry points.

Assembly is intentionally dense in M2a:

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

## Batching Boundary

Named-block residual, objective, gradient, Jacobian-block, and dense-assembly
methods accept arbitrary common leading batch axes and preserve them. This is
**batched evaluation only**. M2a does not implement batched iterations,
per-element damping, accept/reject decisions, convergence statuses, or a phase
engine.

The legacy optimizers and `solve_ik` remain single-problem solvers. Do not pass
a block `Problem` to them merely because its evaluation methods are batched.

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
- Linear solvers: `Cholesky` and `LSTSQ`. `CG` and `SparseCholesky` are
  importable stubs that raise `NotImplementedError`.
- Damping: `Constant` and `Adaptive`. `TrustRegion` is an importable stub.
- Kernels: `L2`, `Huber`, `Cauchy`, and `Tukey` with the legacy row-wise IRLS
  convention.

Legacy LM projects trial points to `lower/upper`, does not clamp the initial
point, and has no active-set/KKT treatment. It can finish as `maxiter` at an
active bound.

## Top-Level `solve()`

`better_robot.optim.solve(problem, ...)` is a convenience wrapper for the
legacy `LeastSquaresProblem` and defaults to legacy `LevenbergMarquardt`. It
does **not** dispatch `optim.blocks.Problem`, scalar block objectives, or
batched block problems. Named-block consumers keep a manual loop until the
solver migration milestones land.
