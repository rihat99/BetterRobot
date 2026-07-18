# `optim/` — Nonlinear Least Squares

This package has one problem representation: named tensor variables and
fixed-width residuals in a `Problem`. LM/GN and `run_first_order` consume that
same representation; task helpers build it rather than defining another
solver protocol.

## Variables and manifolds

`Values` maps names to tensors. `VarSpec` defines each tensor's trailing event
shape, manifold (`Euclidean`, `SO3Manifold`, `SE3Manifold`, or
`RobotConfig`), optional state-space `Bounds`, tangent mask and scale, and an
optional knot-major `time_axis=0`.

Leading axes are independent execution batches. Masks eliminate tangent
columns from derivatives and linear systems. Retraction projects supported
bounds in state space; bounds are not tangent step limits. Validate structural
compatibility at solve entry, then let numerical non-finites produce honest
solver status.

## Residuals and providers

A residual declares a positive static `dim`, optional `reads`, and returns
`(..., dim)`. Optional `jacobian_blocks` are already in reduced tangent
coordinates. Strategies are `auto`, `analytic`, `jacrev`, `jacfwd`, and the
explicit debug-only `finite_difference`; a failing analytic block is an error.

`ResidualItem` owns weight, robust kernel, and group size. The objective and
LM/GN row weights must implement the same grouped robust loss. A Python zero
weight skips the item, including any constant kernel offset.

Providers declare `reads` and `outputs`. `EvaluationContext` resolves them
recursively and caches results for one evaluation only. Never persist
graph-bearing provider results across candidate values. A single
`RobotConfig` variable gets `RobotStateProvider` automatically when `data` is
needed and no explicit provider supplies it.

## Solvers

`LevenbergMarquardt` and `GaussNewton` preserve arbitrary leading batch axes
and per-element cost, damping, acceptance, convergence, status, and iteration
state. Dense routing uses `Cholesky`; directly eligible temporal problems may
use `BlockBandedMatrix` and `BandedCholesky`. `auto` must report a stable
`LinearizationReason` when it chooses dense execution.

Use `init_state` / `update` / `finalize` for an application-owned loop, or
detached eager `run`. `update` stays fixed-shape, input-pure, sync-free, and
tensor-branching. Eager `run` may perform one all-terminal host check per
iteration. Keep `LMState` tensor-only and require projected-gradient KKT for
bounded success.

`run_first_order` owns persistent tangent parameters, retracts after each
Torch optimizer step, and rebases without replacing those parameters.

## Differentiation

`solve(..., differentiate="implicit")` attaches a guarded first-order backward
to a detached converged solution. Keep guards for convergence, active bounds,
robust kinks, quaternion branch cuts, routing/size limits, non-finite systems,
and singular systems. Only declared external parameters receive gradients;
never infer roles from tensor identity or `requires_grad`.

Direct public modules are `problem.py`, `variables.py`, `manifolds.py`,
`providers.py`, `lm.py`, `first_order.py`, `implicit.py`, `kernels.py`,
`solvers.py`, and `temporal.py`. Keep new behavior on this surface.
