# `optim/` — Nonlinear Least Squares

This package has one graph representation: object-owned variables referenced
by fixed-width residuals and harvested into a `Problem`. LM/GN and
`TorchOptimizer` consume that same graph; task helpers assemble it rather than
defining another optimizer protocol.

## Variables and geometry

`Variable` owns its current tensor, stable name, trainable/static role, event
shape, optional `Bounds`, tangent mask and scale, batch declaration, and
optional knot-major `time_axis=0`. `SO3Variable`, `SE3Variable`, and
`RobotVariable` own their corresponding retract/difference geometry. There is
no separate public variable specification or manifold object.

Leading axes are independent execution batches. Masks eliminate tangent
columns from derivatives and linear systems. Retraction projects supported
bounds in state space; bounds are not tangent step limits. Validate structural
compatibility at graph freeze or solve entry, then let numerical non-finites
produce honest optimizer status.

## Residuals, nodes, and problems

A `Residual` holds ordered references to every variable it reads, a positive
static `dim`, `name`, `weight`, robust `kernel`, and `group_size`. `error()`
returns `(..., dim)`. Optional `jacobian()` blocks are ordered like the
trainable dependencies and already use reduced tangent coordinates. Strategies
are `auto`, `analytic`, `jacrev`, `jacfwd`, and explicit debug-only
`finite_difference`; a malformed advertised analytic block is an error.

Evaluation-scoped `Node` objects own shared graph-bearing work such as
`RobotState`. Residuals list nodes in `nodes`; `Problem` merges compatible
nodes, harvests their variable dependencies, and invalidates memos at every
evaluation boundary. Never retain a node result across candidate values.

`Problem` freezes on first use, validates unique names, and computes row and
tangent-column layouts from references. A Python-zero residual weight skips
its rows; tensor zero remains graph-visible. Grouped robust objective and IRLS
row scaling must stay mathematically consistent.

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
stable `LinearizationReason`.

Keep LM's private per-iteration tensor program input-pure, fixed-shape,
sync-free, and tensor-branching. `TorchOptimizer` owns persistent tangent
buffers, delegates update rules to `torch.optim`, retracts after each step, and
rebases without discarding optimizer state.

## Differentiation

`optimize(differentiate="implicit")` attaches a guarded first-order backward
to an eligible detached solution. Differentiable inputs are graph-carrying
static `Variable` objects referenced by residuals or nodes. Keep guards for
convergence, active bounds, robust kinks, quaternion branch cuts, routing and
size limits, non-finite systems, and singular systems.

Direct public modules are `problem.py`, `variables.py`, `optimizers.py`,
`lm.py`, `implicit.py`, `kernels.py`, `solvers.py`, and `temporal.py`;
`manifolds.py` retains only bounds and private shared helpers. Evaluation nodes
live with residuals. Keep new behavior on this surface.
