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
    # None  → LM uses torch.func.jacrev(total_residual)(x)
    # not None → LM calls jacobian_fn(x) → (m, n) Tensor
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

## Stub Solvers

`gauss_newton.py`, `adam.py`, `lbfgs.py` — registered as `"gn"`, `"adam"`, `"lbfgs"`. All raise `NotImplementedError`.

When implementing:
- **GN**: same structure as LM, set `lam=0`
- **Adam**: `torch.optim.Adam([x], lr=...)` with scalar loss `0.5 * ‖r‖²`
- **LBFGS**: `torch.optim.LBFGS([x], ...)` with closure

All must respect `problem.lower_bounds / upper_bounds` via clamping.
