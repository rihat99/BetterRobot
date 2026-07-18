# optim/ — Named-block optimization

One flat optimization package is supported: named variables and least-squares
residuals in a `Problem`, consumed by LM/GN or the `torch.optim` adapter.

## Surface

| Entry | Role |
|---|---|
| `Problem.add_variable` / `add_residual` | Simple builder surface |
| `VarSpec` / `ResidualItem` | Underlying static records |
| `LevenbergMarquardt` / `GaussNewton` | Dense or block-banded second-order solve |
| `run_first_order` | Persistent tangent buffers updated by any compatible `torch.optim.Optimizer` |
| `Cholesky` / `BandedCholesky` | Strict dense SPD and block-banded solves |

The package is flat: implementation modules are `problem.py`, `variables.py`,
`manifolds.py`, `providers.py`, `lm.py`, `first_order.py`, `temporal.py`,
`implicit.py`, `kernels.py`, and `solvers.py`.

## Named variable blocks

`VarSpec` describes one entry in `Values = dict[str, Tensor]`:

- `shape` is the event shape; leading value axes are independent batches.
- `manifold` is `Euclidean`, `SO3Manifold`, `SE3Manifold`, or
  `RobotConfig(model)` and uses local/right perturbations.
- `mask` is flattened tangent-space structure. Fixed entries are eliminated
  from steps, gradients, Jacobian columns, and normal systems.
- `scale` is reduced tangent-coordinate metadata used by LM/GN.
- `time_axis=0` marks knot-major time structure without changing value or
  dense-column order.

`Bounds` is a state/configuration-space box. Retraction projects back to it;
it is never reinterpreted as a tangent-space trust region. Global boxes on
`SO3Manifold` and `SE3Manifold` are rejected. `RobotConfig.joint_bounds()`
constructs model joint bounds while excluding unit-norm coordinates.

Structural validation (names, event/batch shape, dtype/device, compatible
scale and bounds) belongs at the public solve boundary. Do not reintroduce
per-call `_prevalidated` twins or `isfinite` scans of user values. Numerical
non-finites propagate to honest solver status: an invalid current model is
`FAILED`; an invalid trial is rejected and may finish `MAXITER`.

## Residuals and linearization

A residual declares `name`, positive static `dim`, optional `reads`, and
`__call__(ctx) -> Tensor` shaped `(B..., dim)`. A plain function registered
with `dim=` is valid. If the problem has exactly one variable, omitted `reads`
defaults to that variable. `reads` declares Jacobian structure; it is not
runtime access policing.

Optional `jacobian_blocks(ctx)` entries are keyed by variable name and already
use mask-reduced tangent columns. AD strategy is the literal `"auto"`,
`"analytic"`, `"jacrev"`, `"jacfwd"`, or
`"finite_difference"`. A raising analytic block is an error, never a silent
fallback. Finite differences are an explicit debug path.

`ResidualItem.kernel` and `group_size` define robust groups.
`Problem.objective()` sums grouped `rho`; LM/GN applies the matching
`weight(s) = 2*rho'(s)` IRLS rows. `Problem.residual()` stays the weighted raw
vector. There is no scalar-objective subsystem: every optimized term is a
residual.

Temporal residuals may provide `TemporalPattern` and exact reduced numeric
blocks. The supported LM routes are:

- `"dense"`: dense normal plus `Cholesky`;
- `"structured"`: requires direct temporal eligibility and uses
  `BlockBandedMatrix` plus `BandedCholesky`;
- `"auto"`: banded only when directly eligible, otherwise dense with stable
  `LinearizationReason` and detail.

There is no operator/matrix-free LM route, numerical-zero sparsity inference,
or mixed band-plus-dense container. Temporal plus shared optimized variables
remain dense until a reviewed Schur design has a production caller.

## Providers and evaluation lifetime

Providers declare `reads` and `outputs`, then return the declared mapping.
`EvaluationContext` resolves them recursively and memoizes outputs for one
evaluation only. A resolving set makes dependency cycles direct errors.
Provider reads propagate to variable dependency structure. Context values are
read-only, but undeclared access is not policed.

Shared graph-bearing FK/nearest-neighbor outputs never survive an evaluation.
AD-generated missing blocks get fresh transform-local contexts. A single
`RobotConfig` variable receives an automatic `RobotStateProvider` when a
consumer reads `"data"` and no explicit provider supplies it.

## Solver lifecycle and batching

LM/GN preserve arbitrary leading batch axes and per-element cost, damping,
acceptance, factorization, convergence, status, and iteration state. Use
`init_state`, fixed-shape `update`, and `finalize` for an external loop, or
detached eager `run`. Warm starts retain damping only across an exactly
compatible batch shape, dtype, and device.

`LMState` is tensor-only. `increase_factor` is warm-start/algorithmic state;
do not turn removed pure diagnostics back into public state. `mu_min` is
strictly positive, and a factorization that keeps failing gets one attempt at
`mu_max` before `FAILED`. Bounded success requires projected-gradient KKT.

`run_first_order(values, problem, optimizer_factory, ...)` owns persistent
tangent parameters, retracts after each Torch optimizer step, and rebases the
buffers without replacing them. `FirstOrderResult` contains per-element
`step`, `converged`, and `cost`. Do not recreate the deleted custom Adam warm
state, per-element bias correction, atomic rollback, or phase engine.

## Implicit differentiation

`solve(..., differentiate="implicit")` attaches a first-order backward to a
detached converged solution. Preserve every guard that prevents silently
wrong gradients: convergence, stable active set, robust kinks, absolute-pi
quaternion representatives, dense-size/routing limits, non-finite systems,
and singular systems. A small banded forward requires explicit capped dense
backward opt-in; true banded backward is deferred.

Only named external parameters receive gradients. Initialization, warm state,
bounds, masks, and solver configuration do not. Never infer parameter binding
from tensor identity, and never relax a correctness guard merely to make a
case return a gradient.

## Invariants

- `update` stays fixed-shape, sync-free, input-pure, and tensor-branching only.
- Eager `run` may perform one all-terminal host check per iteration.
- Finite world-axis boxes on free-flyer translation remain rejected because
  the solver tangent is right-local.
- Residual/provider code with dynamic shapes, host synchronization, or
  value-keyed Python caches remains eager-only.
