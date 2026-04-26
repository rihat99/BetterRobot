# optim/ — Least-Squares Optimization

## Architecture

Three-layer pluggable design:

```
LeastSquaresProblem  →  Optimizer  →  LinearSolver
                         ↓              ↓
                    DampingStrategy   RobustKernel
```

## LeastSquaresProblem

Connects the cost stack to the solver:
- `cost_stack: CostStack` — weighted residual composition
- `state_factory: Callable[[Tensor], ResidualState]` — wraps flat `x` into ResidualState (runs FK for IK)
- `x0: Tensor` — initial guess
- `lower/upper: Tensor | None` — box constraints
- `retract: Callable` — manifold retraction (handles SE3 for free-flyer)
- `nv: int | None` — tangent-space dimension; defaults to `x0.shape[-1]`
- `.residual(x)`, `.jacobian(x)`, `.step(x, delta_v)`
- `.gradient(x)` — matrix-free `Σ wᵢ² Jᵢᵀ rᵢ` via per-residual `apply_jac_transpose` (skips dense `J` for banded/sparse residuals)
- `.jacobian_blocks(x)` — per-item Jacobian dict for block-sparse trajopt solvers

## Pluggable Components

**Optimizers** (`optimizers/`):
- Single-stage: `LevenbergMarquardt`, `GaussNewton`, `Adam`, `LBFGS`
- Multi-stage: `MultiStageOptimizer` + `OptimizerStage` (snapshots `cost_stack` state on entry; restores in a `try / finally` so a stage that raises does not leak `active`/`weight` changes)
- `LMThenLBFGS` is a thin wrapper around `MultiStageOptimizer((LM, LBFGS))` with optional stage-2 disabled items.

**Linear solvers** (`solvers/`): `Cholesky`, `LSTSQ`, `CG`, `SparseCholesky`
**Damping strategies** (`strategies/`): `Constant`, `Adaptive`, `TrustRegion`
**Robust kernels** (`kernels/`): `L2`, `Huber`, `Cauchy`, `Tukey`

All four pluggable layers are runtime-checkable Protocols (`base.py` in each subpackage); they're enforced by `tests/contract/test_pluggable_protocols.py`.

## LM Details

- IRLS reweighting: a non-`L2` kernel produces per-row `w_i = kernel.weight(r_i²)`; the residual and Jacobian rows are scaled by `sqrt(w_i)` before forming the normal equations.
- Adaptive damping (default): starts at `lam0=1e-4`, doubles on reject, halves on accept.
- After each accepted step: clamps `x_new` to `[lower, upper]` (no-op for ±∞ bounds).
- Initial `x0` is NOT clamped — caller must provide a feasible start.
- Convergence: `||J^T r|| < tol`.
- Gain ratio (`actual / predicted` cost decrease) is recorded on every accept.

## ResidualSpec

Optional metadata on a residual for sparse-aware solvers:
```python
ResidualSpec(
    dim=int,
    output_dim=int | None,
    tangent_dim=int | None,
    structure="dense" | "diagonal" | "block" | "banded",
    time_coupling="single" | "5-point" | "custom",
    affected_knots=tuple[int, ...],
    affected_joints=tuple[int, ...],
    affected_frames=tuple[int, ...],
    dynamic_dim=bool,
)
```

`Residual.spec` is opt-in — solvers that ignore it continue to work.

## Top-Level `solve()`

`optim.solve(problem, optimizer=None, ...)` is a one-shot convenience that defaults to `LevenbergMarquardt()`. Used by `tasks/ik.py`.
