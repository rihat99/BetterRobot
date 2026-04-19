# 07 · Residuals, Costs, Solvers

This document defines the optimization substrate: **pure residual functions**,
a **named+weighted cost stack**, and **pluggable least-squares solvers**.
It generalises the current `CostTerm`/`Problem`/`SOLVERS` trio into
something the IK, trajopt, and (future) optimal-control layers all sit on.

> **Extension points.** `Residual`, `Optimizer`, `LinearSolver`,
> `RobustKernel`, `DampingStrategy`, and `StopScheduler` are all
> `typing.Protocol`s — user extensions need only implement the shape.
> See [15_EXTENSION.md](15_EXTENSION.md) for step-by-step recipes.

> **Jacobian strategy.** The `JacobianStrategy` enum is defined in
> [05_KINEMATICS.md §3](05_KINEMATICS.md) and is the single source of
> truth for analytic/autodiff/functional dispatch. Do not redefine it here.

## 1. The three layers

```
residuals/          pure functions: state → residual vector
costs/              named, weighted compositions of residuals
optim/              Gauss-Newton / LM / Adam / ... over a LeastSquaresProblem
```

Every layer is a plain Python module full of small classes. No custom
metaclass, no global registry except the residual decorator.

## 2. Residuals

### Signature

```python
# src/better_robot/residuals/base.py
from typing import Protocol
import torch

from ..data_model.model import Model
from ..data_model.data  import Data

class ResidualState:
    """Thin struct passed to every residual.

    Residuals do NOT take free-standing kwargs. All configuration
    (target_pose, weights, link indices, …) is captured as attributes of the
    concrete residual object, constructed once.

    Attributes
    ----------
    model     : the immutable Model
    data      : a Data object whose joint_pose_world / frame_pose_world are up-to-date
    variables : the flat optimisation variable tensor x; residuals
                reach back to model-via-Data or straight into x depending
                on how the residual was built.
    """
    model: Model
    data:  Data
    variables: torch.Tensor      # (B..., nx) flat variable tensor

class Residual(Protocol):
    name: str
    dim: int

    def __call__(self, state: ResidualState) -> torch.Tensor:
        """(B..., dim)."""

    def jacobian(self, state: ResidualState) -> torch.Tensor | None:
        """(B..., dim, nx) or None if analytic is not available."""
```

### The decorator

```python
# src/better_robot/residuals/registry.py
_REGISTRY: dict[str, type[Residual]] = {}

def register_residual(name: str):
    def _inner(cls):
        if name in _REGISTRY:
            raise ValueError(f"residual '{name}' already registered")
        _REGISTRY[name] = cls
        cls.name = name
        return cls
    return _inner

def get_residual(name: str) -> type[Residual]:
    return _REGISTRY[name]
```

Third-party users add their own residuals by decorating a class. The solver
has no idea the registry exists — residuals are composed into a `CostStack`,
and the stack is passed to the solver.

### Built-in residual catalog

The first round lives under `residuals/`:

| File | Residual class | dim | Analytic |
|------|----------------|-----|----------|
| `pose.py` | `PoseResidual`, `PositionResidual`, `OrientationResidual` | 6 / 3 / 3 | ✔ |
| `limits.py` | `JointPositionLimit`, `JointVelocityLimit`, `JointAccelLimit` | 2 * nv | ✔ (diagonal) |
| `regularization.py` | `RestResidual`, `NullspaceResidual` | nv | ✔ (identity) |
| `smoothness.py` | `Velocity5pt`, `Accel5pt`, `Jerk5pt` | nv * (T - k) | ✔ |
| `manipulability.py` | `YoshikawaResidual` | 1 | ✘ (autodiff) |
| `collision.py` | `SelfCollisionResidual`, `WorldCollisionResidual` | variable | partial |

A residual with no analytic implementation simply returns `None` from
`jacobian()` — the solver handles the fallback.

### Example: `RestResidual`

```python
# src/better_robot/residuals/regularization.py
import torch
from .base import Residual, ResidualState
from .registry import register_residual

@register_residual("rest")
class RestResidual(Residual):
    dim: int
    def __init__(self, q_rest: torch.Tensor, *, weight: float = 1.0) -> None:
        self.q_rest = q_rest
        self.weight = weight
        self.dim = q_rest.shape[-1]

    def __call__(self, state: ResidualState) -> torch.Tensor:
        q = state.variables                         # (B..., nv) for this problem
        return (q - self.q_rest) * self.weight

    def jacobian(self, state: ResidualState) -> torch.Tensor:
        I = torch.eye(self.dim, device=state.variables.device,
                      dtype=state.variables.dtype)
        return I.expand(*state.variables.shape[:-1], self.dim, self.dim) * self.weight
```

### Example: `PoseResidual` — analytic via `get_frame_jacobian`

See [05_KINEMATICS.md §5](05_KINEMATICS.md#5-pose-residual-analytic-jacobian--the-elegant-version).
The residual stores a `frame_id` and a target SE3; its `.jacobian()` composes
`Jr_inv(log_err)` with `get_frame_jacobian(..., reference='local')`. Works
for **any** root joint — free-flyer or fixed.

## 3. Costs — `CostStack`

```python
# src/better_robot/costs/stack.py

@dataclass
class CostItem:
    name: str
    residual: Residual
    weight: float = 1.0
    active: bool   = True
    kind:   Literal["soft", "constraint_leq_zero"] = "soft"

class CostStack:
    """Named, weighted, individually activatable stack of residuals.

    Mirrors Crocoddyl's `CostModelSum` — a dict keyed by name, with scalar
    weights and per-item on/off flags. The stack concatenates the weighted
    residuals of all active items into a single vector and provides
    slice-maps so solvers can compute per-item Jacobians in place.

    Usage:
        stack = CostStack()
        stack.add("pose_rh", PoseResidual(frame_id=..., target=...), weight=1.0)
        stack.add("pose_lh", PoseResidual(frame_id=..., target=...), weight=1.0)
        stack.add("limits",  JointPositionLimit(),        weight=0.1)
        stack.add("rest",    RestResidual(q_rest),        weight=0.01)

        r = stack.residual(state)             # (B..., total_dim)
        J = stack.jacobian(state)             # (B..., total_dim, nx)
    """

    items: dict[str, CostItem]

    def add(self, name, residual, *, weight=1.0, kind="soft"): ...
    def remove(self, name): ...
    def set_active(self, name: str, active: bool): ...
    def set_weight(self, name: str, weight: float): ...
    def total_dim(self) -> int: ...
    def slice_map(self) -> dict[str, slice]: ...
    def residual(self, state: ResidualState) -> torch.Tensor: ...
    def jacobian(self, state: ResidualState, *, strategy=JacobianStrategy.AUTO) -> torch.Tensor: ...
```

`stack.residual()` evaluates every active residual and concatenates the
results along the last dim. `stack.jacobian()` does the same for Jacobians,
dispatching to analytic or autodiff per-residual via the strategy flag.

Crocoddyl-style memory layout:

- Allocate one flat residual buffer of shape `(B..., total_dim)` inside
  `CostStack`.
- Each residual writes into its pre-computed slice `items[name].slice`.
- No Python-level `torch.cat` in the hot path — only a pre-allocated tensor
  plus `index_put_`-style writes.

### Sparsity-aware assembly (cuRobo pattern)

For residuals whose Jacobian is structurally sparse (self-collision, joint
limits, 5-point finite differences across a trajectory), the residual
exposes a `.spec: ResidualSpec` (§7) describing which columns of `x` it
touches. `CostStack.jacobian(...)` propagates the sparsity into a
block-CSR layout, and the linear solver (see
[14_PERFORMANCE.md §2.8](14_PERFORMANCE.md)) assembles only the non-zero
blocks of `JᵀJ`. On the G1 humanoid this yields ~6× speed-up on the
collision-Jacobian step.

Residuals without `.spec` are assumed dense — nothing regresses.

## 4. `LeastSquaresProblem`

```python
# src/better_robot/optim/problem.py

@dataclass
class LeastSquaresProblem:
    """A least-squares problem over a flat optimisation variable x ∈ (nx,).

    - The cost stack supplies residuals r(x) and (optionally) J(x).
    - Equality/inequality constraints are exposed as cost items with
      kind="constraint_leq_zero" (crocoddyl-like).
    - Variable bounds are hard — enforced by projection at every step.

    The problem is intentionally minimal; the solvers in `optim/optimizers/`
    own the iteration strategy.
    """
    cost_stack: CostStack
    state_factory: Callable[[Tensor], ResidualState]    # wrap x into ResidualState
    x0: Tensor
    lower: Tensor | None = None
    upper: Tensor | None = None
    jacobian_strategy: JacobianStrategy = JacobianStrategy.AUTO

    def residual(self, x: Tensor) -> Tensor: ...
    def jacobian(self, x: Tensor) -> Tensor: ...
```

The `state_factory` callable is how a problem hooks into the data model:
for an IK problem, it turns the flat `x` into a `Data` object via
`model.create_data(q=x)` and a cached FK pass. This keeps `LeastSquaresProblem`
agnostic — it does not know whether it is solving IK, trajopt, or
pose-graph SLAM.

## 5. Solvers — pluggable components

```
optim/optimizers/
├── base.py                   # Optimizer ABC
├── gauss_newton.py
├── levenberg_marquardt.py
├── adam.py
└── lbfgs.py
```

Every optimiser implements the `Optimizer` Protocol:

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class Optimizer(Protocol):
    def minimize(
        self,
        problem: LeastSquaresProblem,
        *,
        max_iter: int,
        linear_solver: "LinearSolver",
        kernel: "RobustKernel",
        strategy: "DampingStrategy",
        scheduler: "StopScheduler" | None = None,
    ) -> "OptimizationResult":
        ...
```

Because this is a `Protocol`, any class with a matching `minimize` is an
`Optimizer` — **no inheritance required**. User-provided optimizers
(DDP, iLQR, ADMM, IPOPT) plug in without touching the library (see
[15_EXTENSION.md §3](15_EXTENSION.md)).

### SolverState (shared between Optimizer, DampingStrategy, LinearSolver)

All solvers share a common per-iteration state object:

```python
@dataclass
class SolverState:
    x:             Tensor              # (B..., nx) current iterate
    residual:      Tensor              # (B..., total_dim) r(x)
    residual_norm: Tensor              # (B...,) ||r(x)||
    iters:         int
    damping:       float               # λ for LM / trust-region radius for TR
    gain_ratio:    float | None = None
    status:        Literal["running", "converged", "stalled", "maxiter"] = "running"

    @classmethod
    def from_problem(cls, problem: LeastSquaresProblem) -> "SolverState": ...
    def converged(self, tol: float) -> bool: ...
```

This is the "one struct passes through every component" pattern cuRobo
uses — it removes the ad-hoc tuple returns in the current codebase.

### Why pluggable

PyPose's `LevenbergMarquardt` is the blueprint:

```python
optimizer = LevenbergMarquardt(
    kernel=Huber(delta=0.1),
    linear_solver=Cholesky(),
    strategy=TrustRegion(),
    scheduler=StopOnPlateau(tol=1e-8, patience=5),
)
```

Each of `kernel`, `linear_solver`, `strategy`, `scheduler` is a small class.
Replacing one swaps one knob without touching the optimisation loop.

### Linear solvers (`optim/solvers/`)

```python
class LinearSolver(Protocol):
    def solve(self, A: Tensor, b: Tensor) -> Tensor: ...

class Cholesky(LinearSolver): ...     # dense, SPD
class LSTSQ(LinearSolver): ...        # rank-deficient safe
class CG(LinearSolver): ...           # conjugate gradients for large sparse
class SparseCholesky(LinearSolver): ... # scipy/torch_sparse fallback
```

### Robust kernels (`optim/kernels/`)

```python
class RobustKernel(Protocol):
    def weight(self, squared_norm: Tensor) -> Tensor: ...

class L2(RobustKernel):    ...   # trivial identity
class Huber(RobustKernel): ...
class Cauchy(RobustKernel): ...
class Tukey(RobustKernel): ...
```

### Damping strategies (`optim/strategies/`)

```python
class DampingStrategy(Protocol):
    def init(self, problem) -> float: ...
    def accept(self, lam: float) -> float: ...
    def reject(self, lam: float) -> float: ...

class Constant(DampingStrategy): ...
class Adaptive(DampingStrategy): ...                # current adaptive LM
class TrustRegion(DampingStrategy): ...
```

### `LevenbergMarquardt.minimize` sketch

```python
def minimize(self, problem, *, max_iter, linear_solver, kernel, strategy, scheduler=None):
    x = problem.x0.clone()
    lam = strategy.init(problem)
    for step in range(max_iter):
        r = problem.residual(x)
        J = problem.jacobian(x)
        r = kernel.weight(r.pow(2).sum(-1, keepdim=True)).sqrt() * r
        JtJ = J.mT @ J
        Jtr = J.mT @ r

        while True:
            A = JtJ + lam * torch.eye(x.shape[-1], dtype=x.dtype, device=x.device)
            delta = linear_solver.solve(A, -Jtr)
            x_new = _project(x + delta, problem.lower, problem.upper)
            if problem.residual(x_new).norm() <= r.norm():
                x, lam = x_new, strategy.accept(lam)
                break
            lam = strategy.reject(lam)
            if lam > MAX_LAM: break

        if scheduler and scheduler.should_stop(step, r, x):
            break
    return OptimizationResult(x=x, residual=r, iters=step + 1)
```

This is one loop. The current four implementations of LM
(our LM, PyPose LM, floating-base autodiff LM, floating-base analytic LM)
all collapse into this.

## 6. The Jacobian strategy flag lives on the problem

```python
problem = LeastSquaresProblem(
    cost_stack=stack,
    state_factory=lambda x: ResidualState(model=model,
                                          data=model.forward_kinematics(x, compute_frames=True),
                                          variables=x),
    x0=q0,
    lower=model.lower_pos_limit,
    upper=model.upper_pos_limit,
    jacobian_strategy=JacobianStrategy.AUTO,
)
```

In `AUTO`, the cost stack asks each residual for an analytic Jacobian. If a
residual returns `None`, that residual's block of the Jacobian is filled by
`torch.func.jacrev` over the residual alone — not over the full stack. This
gives analytic speed for residuals that support it and autodiff correctness
for the rest, with no "switch the whole problem to autodiff because one
residual has no analytic" footgun.

Central finite differences are the final fallback (eps = 1e-3 for fp32,
1e-7 for fp64). Used only when the residual is a black box and autodiff
is broken (e.g. the historical PyPose `Log` gradient bug). See
[17_CONTRACTS.md §5](17_CONTRACTS.md).

## 7. Sparsity hints

```python
# src/better_robot/optim/jacobian_spec.py

@dataclass
class ResidualSpec:
    """Optional structural metadata attached to a residual.

    Lets solvers pre-build sparse Jacobian masks without running
    torch.func.vmap over each column.
    """
    dim: int
    input_indices: tuple[int, ...] | None = None    # which cols of x this residual touches
    is_diagonal: bool = False                       # fast path for diagonal residuals
    time_coupling: Literal["single", "5-point", "custom"] = "single"
```

Residuals can optionally expose `.spec` for solvers that want to exploit
sparsity (trajopt in particular). Solvers that don't look at `.spec` still
work; it's an opt-in.

## 8. Stop schedulers

```python
class StopScheduler(Protocol):
    def should_stop(self, step: int, residual: Tensor, x: Tensor) -> bool: ...

class MaxIterations(StopScheduler): ...
class StopOnPlateau(StopScheduler): ...         # relative improvement
class EarlyStopOnGradient(StopScheduler): ...
```

## 9. Results

```python
@dataclass
class OptimizationResult:
    x: Tensor
    residual: Tensor
    iters: int
    converged: bool
    history: list[dict]              # per-iter {step, loss, lam} — optional
```

## 10. Mapping current code

| Current file | Fate |
|--------------|------|
| `src/better_robot/costs/cost_term.py` (`CostTerm`) | Replaced by `CostItem` in `costs/stack.py`. |
| `src/better_robot/costs/pose.py` | Split into `residuals/pose.py` as `PoseResidual` / `PositionResidual` / `OrientationResidual`. |
| `src/better_robot/costs/limits.py` | Moved to `residuals/limits.py`. |
| `src/better_robot/costs/regularization.py` | Moved to `residuals/regularization.py` + `smoothness.py`. |
| `src/better_robot/costs/collision.py` | Moved to `residuals/collision.py`; closest-point + SDF math stays in `collision/`. |
| `src/better_robot/costs/manipulability.py` | Moved to `residuals/manipulability.py`. |
| `src/better_robot/solvers/problem.py` (`Problem`) | Replaced by `optim/problem.py` (`LeastSquaresProblem`). |
| `src/better_robot/solvers/registry.py` (`SOLVERS`) | Removed — there is no runtime solver registry; optimisers are chosen by import. |
| `src/better_robot/solvers/levenberg_marquardt.py` | Rewritten at `optim/optimizers/levenberg_marquardt.py` against the new shapes. |
| `src/better_robot/solvers/levenberg_marquardt_pypose.py` | **Deleted.** |
| `src/better_robot/solvers/gauss_newton.py`, `adam.py`, `lbfgs.py` | Moved and rewritten in `optim/optimizers/`. |
| `src/better_robot/solvers/base.py` (`Solver` ABC) | Replaced by the `Optimizer` protocol. |

## 11. What this lets users do

```python
import better_robot as br
from better_robot.residuals.pose import PoseResidual
from better_robot.residuals.limits import JointPositionLimit
from better_robot.residuals.regularization import RestResidual
from better_robot.costs import CostStack
from better_robot.optim import LeastSquaresProblem, solve
from better_robot.optim.optimizers import LevenbergMarquardt
from better_robot.optim.strategies import Adaptive
from better_robot.optim.solvers import Cholesky
from better_robot.optim.kernels import Huber

model = br.load("panda.urdf")
hand_id = model.frame_id("panda_hand")

stack = CostStack()
stack.add("pose", PoseResidual(frame_id=hand_id, target=target_pose))
stack.add("limits", JointPositionLimit(model), weight=0.1)
stack.add("rest",   RestResidual(model.q_neutral), weight=0.01)

problem = LeastSquaresProblem(
    cost_stack=stack,
    state_factory=lambda x: br.residuals.ResidualState(
        model=model, data=br.forward_kinematics(model, x, compute_frames=True), variables=x),
    x0=model.q_neutral,
    lower=model.lower_pos_limit,
    upper=model.upper_pos_limit,
)

result = LevenbergMarquardt().minimize(
    problem,
    max_iter=50,
    linear_solver=Cholesky(),
    kernel=Huber(delta=0.1),
    strategy=Adaptive(),
)
```

No fixed-vs-floating special case. No `solver_params` dict. No `jacobian_fn`
argument to `Problem`. A single path from residuals to `result.x`.
