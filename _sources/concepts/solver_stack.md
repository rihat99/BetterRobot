# The Solver Stack

A `CostStack` knows how to compute residuals; it does not know how to
minimise them. That job belongs to `LeastSquaresProblem` (which packs
the cost stack, the initial guess, and the manifold retraction into
a single self-describing problem) and to an `Optimizer` (which
iterates on it). The optimiser does *not* fuse the linear-solver
choice, the robust kernel, the damping schedule, or the
stop-condition into its own loop; those are independently pluggable
axes that compose at construction.

That separation is the whole reason "swap LM for Adam" or "switch
the linear solver to CG for a sparse trajopt" is one Protocol swap
instead of a rewrite. Every optimiser implements the same
`Optimizer` Protocol; every linear solver implements the same
`LinearSolver` Protocol; every robust kernel and damping strategy
likewise. The solver loop sketch at the bottom of this chapter is
deliberately short — it is the *only* solver loop in the library.
The four stages of the LM-then-LBFGS multi-stage solver, the
collision-aware refinement pass, and the trajopt block-Cholesky path
all reduce to the same shape with different components.

This chapter walks through `LeastSquaresProblem` and the four
pluggable axes (Optimizer, LinearSolver, RobustKernel,
DampingStrategy) and ends with the loop sketch.

## `LeastSquaresProblem`

```python
@dataclass
class LeastSquaresProblem:
    """A least-squares problem over a flat optimisation variable x ∈ (nx,).

    - The cost stack supplies residuals r(x) and (optionally) J(x).
    - Equality / inequality constraints are exposed as cost items with
      kind="constraint_leq_zero" (Crocoddyl-like).
    - Variable bounds are hard — enforced by projection at every step.

    The problem is intentionally minimal; the solvers in
    optim/optimizers/ own the iteration strategy.
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
        Jacobian.
        """

    def jacobian_blocks(self, x: Tensor) -> dict["BlockKey", Tensor]:
        """Block-sparse Jacobian per ResidualSpec.

        Used by the block-Cholesky linear solver for trajopt.
        """
```

Source: `src/better_robot/optim/problem.py`.

The `state_factory` callable is how a problem hooks into the data
model: for an IK problem, it turns the flat `x` into a `Data` object
via `model.create_data(q=x)` and a cached FK pass. This keeps
`LeastSquaresProblem` agnostic — it does not know whether it is
solving IK, trajopt, or pose-graph SLAM.

The two extras worth highlighting:

- **`gradient(x)` is matrix-free.** Adam, L-BFGS, and any optimiser
  that does not need the dense Jacobian read this instead of
  `jacobian(x)`. For dense residuals the default
  `apply_jac_transpose` falls back to `J.T @ r`; for banded /
  temporal residuals a per-knot kernel keeps the per-iteration
  memory at `O(T·nv)`. This is what makes a 200-knot G1 trajopt fit
  inside the 200 MiB CUDA peak watermark from {doc}`/conventions/performance` §1.3.
- **`jacobian_blocks(x)` exposes structure.** The block-Cholesky
  linear solver reads it to skip zero blocks. Residuals whose
  `spec.structure` is `"dense"` contribute one block;
  `"block"` / `"banded"` items contribute their declared blocks;
  `"matrix_free"` items raise — those should be solved with a
  gradient-based optimiser, not assembly-based LM.

## The `Optimizer` Protocol

```python
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

Source: `src/better_robot/optim/optimizers/base.py`.

Because this is a `Protocol`, any class with a matching `minimize` is
an `Optimizer` — no inheritance required. User-provided optimisers
(DDP, iLQR, ADMM, IPOPT) plug in without touching the library; see
{doc}`/conventions/extension` §3.

Built-in optimisers:

| File | Class | Notes |
|------|-------|-------|
| `levenberg_marquardt.py` | `LevenbergMarquardt` | Default; analytic Jacobian + adaptive damping |
| `gauss_newton.py` | `GaussNewton` | Pure GN; no damping |
| `adam.py` | `Adam` | Reads `problem.gradient(x)`; never materialises J |
| `lbfgs.py` | `LBFGS` | Same |
| `multi_stage.py` | `MultiStageOptimizer` | Sequence of stages with weight overrides |
| `lm_then_lbfgs.py` | `LMThenLBFGS` | Backward-compat wrapper around `MultiStageOptimizer` |

## `SolverState`

All solvers share a common per-iteration state object passed between
the optimiser, the damping strategy, the linear solver, and any
custom diagnostics:

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

Source: `src/better_robot/optim/state.py`. The "one struct passes
through every component" pattern (cuRobo) replaces the ad-hoc tuple
returns the early prototype carried.

## Linear solvers

```python
class LinearSolver(Protocol):
    def solve(self, A: Tensor, b: Tensor) -> Tensor: ...

class Cholesky(LinearSolver): ...        # dense, SPD
class LSTSQ(LinearSolver): ...           # rank-deficient safe
class CG(LinearSolver): ...              # conjugate gradients for large sparse
class SparseCholesky(LinearSolver): ...  # block-sparse for trajopt
```

Source: `src/better_robot/optim/linear_solvers/`.

`Cholesky` is the default for dense IK problems. `SparseCholesky` is
the default for trajopt problems where the Jacobian has banded /
block structure (driven by `ResidualSpec`). `CG` is the choice for
very large problems where assembly-based methods exhaust memory;
`LSTSQ` is for cases where `JᵀJ` may be rank-deficient.

## Robust kernels

```python
class RobustKernel(Protocol):
    def weight(self, squared_norm: Tensor) -> Tensor: ...

class L2(RobustKernel):    ...   # trivial identity
class Huber(RobustKernel): ...
class Cauchy(RobustKernel): ...
class Tukey(RobustKernel): ...
```

Source: `src/better_robot/optim/kernels/`.

Robust kernels reweight `r → ρ(r)` per residual, not per cost stack
— different terms can use different kernels. The IRLS-style
reweighting happens once per outer iteration before the linear
solve.

## Damping strategies

```python
class DampingStrategy(Protocol):
    def init(self, problem) -> float: ...
    def accept(self, lam: float) -> float: ...
    def reject(self, lam: float) -> float: ...

class Constant(DampingStrategy):    ...
class Adaptive(DampingStrategy):    ...   # double on reject, halve on accept
class TrustRegion(DampingStrategy): ...
```

Source: `src/better_robot/optim/damping/`.

`Adaptive` is the default for LM. It starts at `1e-4`, doubles on
reject, halves on accept. `TrustRegion` is the right choice when the
problem has known step-size constraints.

## `OptimizerConfig` — every knob is wired

`OptimizerConfig` is the user-facing dial. Every declared knob is
honoured — there are no decorative fields.

```python
@dataclass(frozen=True)
class OptimizerConfig:
    optimizer: Literal["lm", "gn", "adam", "lbfgs",
                       "lm_then_lbfgs", "multi_stage"] = "lm"
    max_iter: int = 100
    jacobian_strategy: JacobianStrategy = JacobianStrategy.AUTO

    linear_solver: Literal["cholesky", "lstsq", "cg", "sparse_cholesky"] = "cholesky"
    kernel: Literal["l2", "huber", "cauchy", "tukey"] = "l2"
    huber_delta: float | None = None
    tukey_c: float | None = None
    damping: float | Literal["constant", "adaptive", "trust_region"] = "adaptive"
    tol: float = 1e-7

    # Two-stage refinement (cuRobo pattern).
    refine_max_iter: int = 30
    refine_cost_mask: tuple[str, ...] = (
        "pose_*", "limits", "rest", "self_collision", "world_collision",
    )

    # Multi-stage (generalises lm_then_lbfgs).
    stages: tuple["OptimizerStage", ...] | None = None
```

`solve_ik` builds the optimiser, linear solver, robust kernel, and
damping strategy explicitly:

```python
optimizer = _make_optimizer     (cfg)
linear    = _make_linear_solver (cfg)
kernel    = _make_robust_kernel (cfg)
damping   = _make_damping       (cfg)
state = optimizer.minimize(problem,
                           linear_solver=linear,
                           kernel=kernel,
                           strategy=damping,
                           max_iter=cfg.max_iter)
```

If a knob is set but ignored by the chosen optimiser (Adam does not
take a linear solver), the build step warns rather than silently
swallowing it. A contract test asserts
`OptimizerConfig(linear_solver="lstsq")` produces a *different*
numerical trajectory than the default — declared knobs are wired,
never decorative.

## `MultiStageOptimizer`

The historical `LMThenLBFGS` is the special case of a more general
construct:

```python
@dataclass(frozen=True)
class OptimizerStage:
    optimizer: Optimizer
    max_iter: int
    active_items: tuple[str, ...] | None = None
    disabled_items: tuple[str, ...] = ()
    weight_overrides: dict[str, float] | None = None
    tol: float | None = None

class MultiStageOptimizer(Optimizer):
    """Run a fixed sequence of solver stages. Each stage may toggle
    cost-stack active flags or override item weights. ``LMThenLBFGS``
    is implemented as ``MultiStageOptimizer(stages=(LM, LBFGS))``.

    The multi-stage optimiser **must restore active-flag and
    weight-override state** even if a stage raises. A try/finally
    around ``cost_stack.snapshot()`` / ``.restore()`` keeps the
    user's CostStack intact regardless of outcome.
    """
```

Tested explicitly: stage-wise weight overrides correctly restore the
original `CostStack` weights after the run, including in error paths.

## The `LevenbergMarquardt.minimize` sketch

```python
def minimize(self, problem, *, max_iter, linear_solver, kernel, strategy, scheduler=None):
    x   = problem.x0.clone()
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

Source: `src/better_robot/optim/optimizers/levenberg_marquardt.py`.

This is one loop. The four optimisers' worth of code that the early
prototype carried (our LM, PyPose LM, fixed-base autodiff LM,
floating-base analytic LM) all collapse into this with different
components plugged in.

## Stop schedulers

```python
class StopScheduler(Protocol):
    def should_stop(self, step: int, residual: Tensor, x: Tensor) -> bool: ...

class MaxIterations(StopScheduler):     ...
class StopOnPlateau(StopScheduler):     ...   # relative improvement
class EarlyStopOnGradient(StopScheduler): ...
```

## Results

```python
@dataclass
class OptimizationResult:
    x: Tensor
    residual: Tensor
    iters: int
    converged: bool
    history: list[dict]              # per-iter {step, loss, lam} — optional
```

The task layer (`solve_ik`, `solve_trajopt`) wraps this in a
task-specific result type (`IKResult`, `TrajOptResult`) that carries
the model and a frame-pose accessor.

## What this lets users do

```python
import better_robot as br
from better_robot.residuals.pose          import PoseResidual
from better_robot.residuals.limits        import JointPositionLimit
from better_robot.residuals.regularization import RestResidual
from better_robot.costs                    import CostStack
from better_robot.optim                    import LeastSquaresProblem
from better_robot.optim.optimizers         import LevenbergMarquardt
from better_robot.optim.damping            import Adaptive
from better_robot.optim.linear_solvers     import Cholesky
from better_robot.optim.kernels            import Huber

model = br.load("panda.urdf")
hand_id = model.frame_id("panda_hand")

stack = CostStack()
stack.add("pose", PoseResidual(frame_id=hand_id, target=target_pose))
stack.add("limits", JointPositionLimit(model), weight=0.1)
stack.add("rest",   RestResidual(model.q_neutral), weight=0.01)

problem = LeastSquaresProblem(
    cost_stack=stack,
    state_factory=lambda x: br.residuals.ResidualState(
        model=model,
        data=br.forward_kinematics(model, x, compute_frames=True),
        variables=x,
    ),
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

No fixed-vs-floating special case. No `solver_params` dict. No
`jacobian_fn` argument. A single path from residuals to `result.x`.

In normal use you would never write that loop; `solve_ik` does it
internally. The example exists to show that the four pluggable axes
compose cleanly.

## Sharp edges

- **Adam and L-BFGS read `problem.gradient(x)`, not
  `problem.jacobian(x)`.** They never materialise the dense
  Jacobian. Long-horizon trajopt depends on this for memory.
- **The LM solver projects after each accepted step.** Initial `x0`
  is **not** projected — caller must provide a feasible `x0` if
  bounds matter.
- **`MultiStageOptimizer` restores weights / active flags via
  try/finally.** Stage-wise overrides do not leak even if a stage
  raises.
- **`OptimizerConfig` knobs are honoured.** A contract test asserts
  setting `linear_solver="lstsq"` produces a different numerical
  trajectory than the default. Decorative knobs would fail the test.

## Where to look next

- {doc}`tasks` — `solve_ik` and `solve_trajopt`, which assemble a
  `CostStack`, build a `LeastSquaresProblem`, and call an
  `Optimizer`.
- {doc}`/conventions/extension` §3, §4, §5, §6 — recipes for adding
  a custom optimiser, damping strategy, linear solver, or robust
  kernel.
- {doc}`/conventions/performance` §2.7 — matrix-free trajopt and the
  memory wins it brings.
