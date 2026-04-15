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
- `.residual(x)`, `.jacobian(x)`, `.step(x, delta_v)`

## Pluggable Components

**Optimizers** (`optimizers/`): `LevenbergMarquardt`, `GaussNewton`, `Adam`, `LBFGS`
**Linear solvers** (`solvers/`): `Cholesky`, `LSTSQ`, `CG`
**Damping strategies** (`strategies/`): `Adaptive`, `Constant`, `TrustRegion`
**Robust kernels** (`kernels/`): `L2`, `Huber`, `Cauchy`, `Tukey`

## LM Details

- Adaptive damping: starts at `lam0=1e-4`, doubles on reject, halves on accept
- After each accepted step: clamps `x_new` to `[lower, upper]`
- Initial `x0` is NOT clamped — caller must provide feasible start
- Convergence: `||J^T r|| < tol`

## Top-Level `solve()`

One-shot convenience wrapper that builds an LM optimizer with sensible defaults. Used by `tasks/ik.py`.
