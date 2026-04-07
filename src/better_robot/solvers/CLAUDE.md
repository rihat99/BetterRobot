# solvers/ — Optimization Backends

Registry-driven solver layer. Depends on `costs/` (for `CostTerm`).

## Public API

```python
from better_robot.solvers import (
    SOLVERS, Registry,
    Problem, Solver,
    LevenbergMarquardt, PyposeLevenbergMarquardt,
    GaussNewton, AdamSolver, LBFGSSolver,
)
```

## Core Abstractions

### `CostTerm` (lives in `costs/cost_term.py`)

```python
@dataclass
class CostTerm:
    residual_fn: Callable[[Tensor], Tensor]   # (x,) → residual vector
    weight: float = 1.0
    kind: Literal["soft", "constraint_leq_zero"] = "soft"
```

### `Problem` (`problem.py`)

```python
@dataclass
class Problem:
    variables: Tensor             # initial value / warm start
    costs: list[CostTerm]
    lower_bounds: Tensor | None   # joint limits for hard projection
    upper_bounds: Tensor | None
    jacobian_fn: Callable | None = None
    # None  → LM/GN uses torch.func.jacrev(total_residual)(x)
    # not None → solver calls jacobian_fn(x) → (m, n) Tensor
```

`problem.total_residual(x)` — concatenates all weighted `soft` residuals.
`problem.constraint_residual(x)` — concatenates all `constraint_leq_zero` residuals.

### `Solver` ABC (`base.py`)

```python
class Solver(ABC):
    @abstractmethod
    def solve(self, problem: Problem, max_iter: int = 100, **kwargs) -> Tensor: ...
```

## Registry (`registry.py`)

```python
SOLVERS: Registry   # global instance

@SOLVERS.register("lm")
class LevenbergMarquardt(Solver): ...

# Usage
solver = SOLVERS.get("lm")(damping=1e-3)    # instantiate with custom params
solver = SOLVERS.get("lm")()                 # default params
solvers_available = SOLVERS.list()           # ["lm", "lm_pypose", "gn", "adam", "lbfgs"]
```

To add a new solver: create a file, decorate the class, and add the import to `__init__.py`.

## LM Solver (`levenberg_marquardt.py`) — DEFAULT `"lm"`

```python
LevenbergMarquardt(damping=1e-4, factor=2.0, reject=16).solve(problem, max_iter=100)
```

- `jacobian_fn=None` → `torch.func.jacrev` (autodiff)
- `jacobian_fn=fn` → calls `fn(x)` (analytic or custom)
- Adaptive damping: accept step if `‖r_new‖ ≤ ‖r‖`, else multiply λ×factor (up to `reject` retries)
- Bounds enforced via `.clamp()` after each accepted step

## PyPose LM (`levenberg_marquardt_pypose.py`) — `"lm_pypose"`

Always uses PyPose autograd for J; ignores `jacobian_fn`. Kept for benchmarking.

**Key gotcha:** uses `vectorize=True` (torch.vmap). Works because FK only branches on fixed joint-type data. If FK branching ever depends on tensor values, `vectorize=True` may break.

## GaussNewton (`gauss_newton.py`) — `"gn"`

```python
GaussNewton().solve(problem, max_iter=100)
```

- Same structure as LM with no adaptive damping (`lam=0`)
- Tiny regularization `1e-8 * I` to handle rank deficiency
- Supports `jacobian_fn` (analytic or autodiff via `jacrev`)
- Bounds clamped after each step

## AdamSolver (`adam.py`) — `"adam"`

```python
AdamSolver(lr=1e-2).solve(problem, max_iter=100)
```

- `torch.optim.Adam([x], lr=lr)` with scalar loss `0.5 * ‖r‖²`
- Bounds enforced via `x.data.clamp_()` after each optimizer step
- Does not use `jacobian_fn`; always uses autograd

## LBFGSSolver (`lbfgs.py`) — `"lbfgs"`

```python
LBFGSSolver(lr=1.0).solve(problem, max_iter=100)
```

- `torch.optim.LBFGS` with `strong_wolfe` line search and `max_iter=20` inner iterations per outer step
- Outer loop runs `max(1, max_iter // 20)` steps
- Bounds enforced via `x.data.clamp_()` after each outer step
- Does not use `jacobian_fn`; always uses autograd
