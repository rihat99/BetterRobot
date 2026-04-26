# 07 · Residuals, Costs, Solvers

This document defines the optimization substrate: **pure residual functions**,
a **named+weighted cost stack**, and **pluggable least-squares solvers**.
It is the substrate the IK, trajopt, and (future) optimal-control layers
all sit on; the pre-skeleton `CostTerm`/`Problem`/`SOLVERS` trio it
replaced is gone.

> **Extension points.** `Residual`, `Optimizer`, `LinearSolver`,
> `RobustKernel`, `DampingStrategy`, `StopScheduler`, and
> `TrajectoryParameterization` are all `typing.Protocol`s — user extensions
> need only implement the shape. See
> [15_EXTENSION.md](../conventions/15_EXTENSION.md) for step-by-step recipes.

> **Jacobian strategy.** The `JacobianStrategy` enum is defined in
> [05_KINEMATICS.md §3](05_KINEMATICS.md) and is the single source of
> truth for analytic/autodiff/functional/finite-diff dispatch. Do not redefine here.

> **Three concepts that look similar but are not.** *Active* (a `CostStack`
> structural inclusion flag), *weight* (a scalar multiplier on the
> residual), and *robust kernel* (a per-iter reweighting in the normal
> equations) are **independent**. `weight = 0` is **not** equivalent to
> `active = False`: a zero-weight item still occupies a slot in the
> preallocated residual / Jacobian; only the active flag structurally
> removes it. Sparse trajopt depends on this distinction.

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

    def spec(self, state: ResidualState) -> "ResidualSpec":
        """Structural metadata. Default impl returns dense, output_dim=dim.
        Override to advertise sparse / banded / matrix-free structure."""

    def apply_jac_transpose(
        self,
        state: ResidualState,
        vec: torch.Tensor,
    ) -> torch.Tensor:
        """Compute J(x)^T @ vec without materialising J.
        Default: J = self.jacobian(state); return J.mT @ vec.
        Override on temporal residuals where J is banded over T —
        long-horizon trajopt depends on this for memory."""
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
[14_PERFORMANCE.md §2.8](../conventions/14_PERFORMANCE.md)) assembles only the non-zero
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

    def gradient(self, x: Tensor) -> Tensor:
        """J(x)^T @ r(x), matrix-free.

        Iterates the active CostStack items and accumulates each item's
        ``apply_jac_transpose(state, r_item)`` contribution. Used by
        Adam, L-BFGS, and any optimiser that does not need the dense
        Jacobian. For dense residuals the default ``apply_jac_transpose``
        falls back to ``J.T @ r``; for banded/temporal residuals a
        per-knot kernel keeps the per-iter memory at ``O(T·nv)``."""

    def jacobian_blocks(self, x: Tensor) -> dict["BlockKey", Tensor]:
        """Block-sparse Jacobian per ResidualSpec.

        Used by the block_cholesky linear solver for trajopt. Residuals
        whose ``spec.structure`` is ``"dense"`` contribute a single block;
        ``"block"`` / ``"banded"`` items contribute their declared blocks;
        ``"matrix_free"`` items raise — they should be solved with a
        gradient-based optimiser, not assembly-based LM."""
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
[15_EXTENSION.md §3](../conventions/15_EXTENSION.md)).

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
uses — it replaces the ad-hoc tuple returns the pre-skeleton solver
loops carried.

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
[17_CONTRACTS.md §5](../conventions/17_CONTRACTS.md).

## 7. Sparsity hints — `ResidualSpec`

```python
# src/better_robot/optim/jacobian_spec.py
from dataclasses import dataclass
from typing import Literal

@dataclass(frozen=True)
class ResidualSpec:
    """Structural metadata about a residual.

    Returned by ``Residual.spec(state)``. Used by solvers to plan
    Jacobian assembly, sparse-block storage, and active-set updates.
    """
    output_dim: int
    tangent_dim: int

    structure: Literal[
        "dense",          # full (output_dim, tangent_dim) Jacobian.
        "block",          # one or more (out, in) blocks at named indices.
        "banded",         # banded along a temporal axis.
        "matrix_free",    # only J^T r is available; J is never built.
    ] = "dense"

    # Time coupling for trajopt residuals.
    time_coupling: Literal["single", "5-point", "custom"] | None = None
    affected_knots: tuple[int, ...] | None = None     # for "block" trajopt residuals

    # Spatial / kinematic locality.
    affected_joints: tuple[int, ...] | None = None
    affected_frames: tuple[int, ...] | None = None

    # Whether output_dim depends on the input (e.g. active-pair collision).
    # If True, a stable upper-bound dimension still exists; the *active*
    # subset varies. Solvers preallocate to the upper bound and zero out
    # inactive rows — see §10 (collision residuals).
    dynamic_dim: bool = False
```

Default `Residual.spec()` returns a dense spec with `output_dim = self.dim`
— existing residuals work without modification. Trajopt residuals
override to advertise their banded / 5-point time coupling so the linear
solver builds a block-Cholesky factorisation; collision residuals
override to set `dynamic_dim=True` so LM preallocates correctly.

## 8. Optimizer config — every knob is wired

`OptimizerConfig` lives in `tasks.ik` (
[08_TASKS.md §1](08_TASKS.md)) and is the user-facing dial.
Every declared knob is honoured — no decorative fields:

```python
@dataclass(frozen=True)
class OptimizerConfig:
    optimizer: Literal["lm", "gn", "adam", "lbfgs",
                       "lm_then_lbfgs", "multi_stage"] = "lm"
    max_iter: int = 100
    jacobian_strategy: JacobianStrategy = JacobianStrategy.AUTO

    linear_solver: Literal["cholesky", "qr", "lsqr", "cg",
                           "block_cholesky"] = "cholesky"
    kernel: Literal["identity", "huber", "cauchy", "tukey"] = "identity"
    huber_delta: float | None = None
    tukey_c: float | None = None
    damping: float | Literal["constant", "adaptive", "trust_region"] = "adaptive"
    tol: float = 1e-7

    # Two-stage refinement.
    refine_max_iter: int = 30
    refine_cost_mask: tuple[str, ...] = (
        "pose_*", "limits", "rest", "self_collision", "world_collision",
    )

    # Multi-stage (generalises lm_then_lbfgs)
    stages: tuple["OptimizerStage", ...] | None = None
```

`solve_ik` builds the optimizer, linear solver, robust kernel, and damping
strategy explicitly:

```python
optimizer = _make_optimizer(optimizer_cfg)
linear    = _make_linear_solver(optimizer_cfg)
kernel    = _make_robust_kernel(optimizer_cfg)
damping   = _make_damping_strategy(optimizer_cfg)
state = optimizer.run(problem,
                      linear_solver=linear,
                      robust_kernel=kernel,
                      damping=damping,
                      max_iter=optimizer_cfg.max_iter)
```

If a knob is set but ignored by the chosen optimiser (e.g. Adam doesn't
take a linear solver), the build step warns rather than silently
swallowing it. A contract test asserts `OptimizerConfig(linear_solver="qr")`
produces a *different* numerical trajectory than the default.

## 9. Multi-stage solvers — `MultiStageOptimizer`

The historical `LMThenLBFGS` is the special case of a more general
construct:

```python
@dataclass(frozen=True)
class OptimizerStage:
    optimizer: Optimizer
    max_iter: int
    active_items: tuple[str, ...] | None = None       # if not None, replace active set
    disabled_items: tuple[str, ...] = ()
    weight_overrides: dict[str, float] | None = None  # e.g. {"smooth": 0.0}
    tol: float | None = None

class MultiStageOptimizer(Optimizer):
    """Run a fixed sequence of solver stages. Each stage may toggle
    cost-stack active flags or override item weights. ``LMThenLBFGS``
    is implemented as ``MultiStageOptimizer(stages=(LM, LBFGS))``.

    The multi-stage optimizer **must restore active-flag and
    weight-override state** even if a stage raises. A try/finally around
    ``cost_stack.snapshot()`` / ``.restore()`` keeps the user's CostStack
    intact regardless of outcome."""
```

Tested explicitly: stage-wise weight overrides correctly restore the
original `CostStack` weights after the run, including in error paths.

## 10. Collision residuals — stable dim, dynamic active set

The collision residual exposes a per-pair output but mixes "candidate
pairs" with "active pairs above the margin". To keep LM line-search and
damping stable, the contract is:

- `dim = number_of_candidate_pairs` — **stable across iterations**.
- Pairs outside the safety margin contribute zero — but the slot exists,
  so the Jacobian has a corresponding row of zeros.
- Active-pair compaction is a *kernel-internal* optimisation; collision
  SDFs and matrix-free `J^T r` paths can avoid the work on inactive rows
  without changing the public residual dim.
- `ResidualSpec.dynamic_dim = True` declares the slot reservation so LM
  preallocates once.

This is what makes `solve_ik` with `SelfCollisionResidual` work without
re-allocating per iteration.

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
