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
```

`Problem.total_residual(x)` concatenates all `soft` cost residuals (weighted). The LM solver minimizes `||total_residual(x)||^2`.

## LM Solver (`_levenberg_marquardt.py`) — IMPLEMENTED

Uses `pypose.optim.LevenbergMarquardt` with:
- `strategy=Adaptive(damping=1e-4)` — adjusts lambda based on cost reduction quality
- `vectorize=True` — ~3.4x faster than `vectorize=False`; works with our FK
- `reject=16` (PyPose default) — adaptive damping with up to 16 retries per step

After each `optimizer.step()`, variables are projected onto `[lower_bounds, upper_bounds]`. This is hard joint limit enforcement — do not remove it.

The `_ProblemModule` wraps `Problem` as `nn.Module`. The `nn.Parameter` `self.x` holds the joint config. PyPose LM computes `J = d(total_residual(x))/d(x)` via autograd and solves the normal equations.

### Key gotcha: `vectorize=True` vs `vectorize=False`

`vectorize=True` uses `torch.vmap` for batched Jacobian computation. This works because the FK, despite having Python-level branching, only branches on fixed joint-type data (not on tensor values). If you restructure FK so the branching depends on tensor data, `vectorize=True` may break.

## Other Solvers — STUBS

`_gauss_newton.py`, `_adam.py`, `_lbfgs.py` raise `NotImplementedError`. When implementing:
- GN: same structure as LM but without damping (`lambda=0`)
- Adam: use `torch.optim.Adam([x], lr=...)` with scalar loss `0.5 * ||r||^2`
- LBFGS: use `torch.optim.LBFGS([x], ...)` with closure

All must respect `problem.lower_bounds/upper_bounds` via clamping.
