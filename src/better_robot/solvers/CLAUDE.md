# solvers/ — Optimization Backends

## Abstractions (`_base.py`)

```python
@dataclass
class CostTerm:
    residual_fn: Callable[[Tensor], Tensor]  # (x,) -> residual vector
    weight: float = 1.0
    kind: Literal["soft", "constraint_leq_zero"] = "soft"

@dataclass
class Problem:
    variables: Tensor          # initial value / warm start
    costs: list[CostTerm]
    lower_bounds: Tensor | None  # joint lower limits (for projection)
    upper_bounds: Tensor | None  # joint upper limits (for projection)
    jacobian_fn: Callable[[Tensor], Tensor] | None = None
    # None  → our LM computes J via torch.func.jacrev(total_residual)(x)
    # not None → our LM calls jacobian_fn(x) directly → (m, n) Tensor
```

`Problem.total_residual(x)` concatenates all `soft` cost residuals (weighted). The LM solver minimizes `||total_residual(x)||^2`.

## LM Solver — OUR IMPLEMENTATION (`_lm.py`) — DEFAULT

`SOLVER_REGISTRY["lm"]` → `LevenbergMarquardt` from `_lm.py`.

```python
LevenbergMarquardt(damping=1e-4, factor=2.0, reject=16).solve(problem, max_iter=100)
```

- `jacobian_fn=None` → uses `torch.func.jacrev(problem.total_residual)(x)` (autodiff)
- `jacobian_fn=fn` → calls `fn(x)` directly (analytic or custom)
- Adaptive damping: accept step if `||r_new|| <= ||r||`, else multiply λ × factor (up to `reject` retries)
- Bounds enforced via `.clamp()` after each accepted step

## LM Solver — PYPOSE (`_levenberg_marquardt.py`) — COMPARISON

`SOLVER_REGISTRY["lm_pypose"]` → PyPose-based LM. Always uses autograd for J. Ignores `jacobian_fn`.

Use this only for comparison/benchmarking. The new default is `_lm.py`.

### Key gotcha: `vectorize=True` vs `vectorize=False`

The PyPose LM uses `vectorize=True` (torch.vmap for batched Jacobian). This works because FK only branches on fixed joint-type data. If FK branching ever depends on tensor values, `vectorize=True` may break.

## Registry

```python
SOLVER_REGISTRY = {
    "lm":        LevenbergMarquardt,           # our LM (default)
    "lm_pypose": PyposeLevenbergMarquardt,     # PyPose LM (comparison)
    "gn":        GaussNewton,                  # stub
    "adam":      AdamSolver,                   # stub
    "lbfgs":     LBFGSSolver,                  # stub
}
```

## Other Solvers — STUBS

`_gauss_newton.py`, `_adam.py`, `_lbfgs.py` raise `NotImplementedError`. When implementing:
- GN: same structure as LM but without damping (`lambda=0`)
- Adam: use `torch.optim.Adam([x], lr=...)` with scalar loss `0.5 * ||r||^2`
- LBFGS: use `torch.optim.LBFGS([x], ...)` with closure

All must respect `problem.lower_bounds/upper_bounds` via clamping.
