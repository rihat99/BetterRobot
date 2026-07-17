# Named-Block Evaluation and the Legacy Solver Stack

BetterRobot temporarily exposes two optimization contracts while tasks migrate
to named variable blocks. They coexist deliberately through M2c:

- **Named-block optimization** is the new public construction and batched
  second-order API. A `Problem` owns named `VarSpec` blocks, structural
  residuals and scalar objective terms, and a lazy provider DAG. It evaluates
  residuals, objectives, tangent gradients, and Jacobian blocks;
  `LevenbergMarquardt` and `GaussNewton` solve residual-vector problems with
  independent tensor state for every batch element.
- **The legacy solver stack** still powers `solve_ik`, `solve_trajopt`, and the
  optimizers under `better_robot.optim.optimizers`. It composes `CostStack`,
  `LeastSquaresProblem`, and `Optimizer`. The convenience
  `better_robot.optim.solve` still belongs exclusively to this legacy path.

## Named-block evaluation

The canonical imports live under `better_robot.optim`; the top-level
`better_robot` namespace remains compact:

```python
from better_robot.optim import (
    Bounds,
    Euclidean,
    GaussNewton,
    LevenbergMarquardt,
    LMState,
    LMStatus,
    ObjectiveItem,
    Problem,
    ResidualItem,
    RobotConfig,
    RobotStateProvider,
    SE3Manifold,
    SO3Manifold,
    Values,
    VarSpec,
    detach_values,
)
```

`SO3Manifold` and `SE3Manifold` are intentionally explicit names.
`better_robot.SE3` and `better_robot.lie.SE3` remain the typed Lie-group pose
wrapper, not optimization manifolds.

A `VarSpec` separates full state coordinates from reduced tangent coordinates.
Its `shape` is an event shape; arbitrary leading axes on each value are
independent batch axes. `bounds` constrain state space, while `mask` eliminates
fixed tangent coordinates and `scale` describes the retained tangent
coordinates. `RobotConfig(model)` is the boundary that handles `nq != nv`.

Residual and provider implementations are structural: authors declare names,
dependencies, and output dimensions without inheriting from a BetterRobot base
class. `Problem` validates that static graph once. Residual, objective,
gradient, and analytic-Jacobian paths use a fresh read-only context and run
each requested provider at most once in that context. An AD-generated
Jacobian uses a fresh transform-local context for each missing
`(residual, variable)` block, so a provider may run once per transformed
block. See {doc}`/guides/custom_residuals` for the author contract.

The public evaluation operations are:

- `residual(values)` and `objective(values)`;
- `gradient(values)`, in reduced tangent coordinates per variable;
- `jacobian_blocks(values)` and `dense_jacobian(values)`;
- `normal_matrix(values)` for dense correctness and solver hand-off; and
- `retract(values, steps)`, which applies manifold-aware feasible steps.

`ResidualItem.kernel` and `group_size` define robust-loss groups consumed by
the named-block second-order solvers. Scalar `ObjectiveItem`s still
participate in `objective()` and `gradient()` for a caller-owned first-order
loop, but LM/GN reject them via `Problem.require_least_squares()` instead of
silently changing their mathematical meaning.

## Named-block LM/GN

The named-block solvers expose a jaxopt-style lifecycle:

```python
from better_robot.optim import LevenbergMarquardt

solver = LevenbergMarquardt(max_iter=50, gtol=1e-6)
state = solver.init_state(values, problem)
values, state = solver.update(values, state, problem)  # one pure tensor step
values, state = solver.finalize(values, state, problem)

# Or use the detached eager driver:
values, state = solver.run(values, problem)
```

For a small explicit unrolled-differentiation oracle, pass
`create_graph=True` to `init_state`, every `update`, and `finalize`. The
default path does not retain the Jacobian graph, and `run` is intentionally
always detached. This oracle is not the stable implicit solver backward;
active-set validity and implicit differentiation remain M6 work.

`LMState` is a fixed-structure pytree containing tensors only. Cost, damping,
gain ratio, accept/reject effects, factorization health, KKT measures, status,
and iteration counts retain every leading batch axis; a rejected or failed
element does not move a valid neighbor. `GaussNewton` is a fixed-damping preset
of the same guarded update rather than a second implementation.

The solver scales residual/Jacobian rows by the configured group-wise robust
weights, solves reduced tangent systems, and applies state-space feasibility
through each `VarSpec` manifold. Finite Euclidean/configuration bounds use a
projected active set and projected-gradient KKT termination. World-axis boxes
on a free-flyer translation are rejected because they are not axis-aligned in
the right-local `SE(3)` tangent; express those constraints as residuals until
constraint-normal support lands.

`update` is a pure, fixed-shape, sync-free tensor program. Public validation
and static layout construction happen in `init_state`; `run` is an eager
convenience loop and may perform one host-side all-terminal check per
iteration. This makes the step capture-ready by construction, but does not
certify CUDA graph replay. Certification requires M6's warmup/capture/replay
parity harness. Custom residuals and providers must also satisfy the
fixed-shape, sync-free eligibility rules in
{doc}`/guides/custom_residuals`; non-eligible residuals remain usable eagerly.

The terminal `LMStatus` values are `RUNNING`, `CONVERGED`,
`STALLED_AT_BOUNDS`, `MAXITER`, and `FAILED`. Call `finalize` after a manual
update loop so residuals, robust weights, active masks, and KKT diagnostics all
describe the returned terminal point. `run` does this automatically and
detaches returned artifacts. A supplied prior state is a warm start for
damping, not permission to reuse stale target-dependent linearizations; its
batch shape, dtype, and device must match exactly.

Accepted-step `xtol`/`ftol` termination applies only to unbounded problems.
Bounded problems require projected-gradient KKT for every success status, and
a tolerance-only unbounded `CONVERGED` state is not `implicit_valid` unless
final evaluation also satisfies KKT. Damping has a strictly positive floor;
after escalation reaches `mu_max`, a failing factorization receives one solve
attempt at the cap before the element becomes `FAILED`.

The legacy optimizer classes have the same human-readable LM/GN names but live
under `better_robot.optim.optimizers`; they consume `LeastSquaresProblem` and
offer `minimize`, not this step API.

## Legacy solver stack (task backend through M2c)

A `CostStack` knows how to compute residuals; it does not know how to minimise
them. In the legacy contract, that job belongs to `LeastSquaresProblem` (which
packs the cost stack, the initial guess, and the manifold retraction into a
single self-describing problem) and to an `Optimizer` (which iterates on it).
The optimizer does *not* fuse the linear-solver choice, robust kernel, damping
schedule, or stop condition into its loop; those remain independently
pluggable axes.

That separation is why "swap LM for Adam" or "switch Cholesky to LSTSQ for a
rank-deficient problem" is one Protocol swap within the legacy stack. Every
legacy optimizer implements the same `Optimizer` Protocol; every linear solver
implements `LinearSolver`; every robust kernel and damping strategy likewise.
The LM-then-LBFGS multi-stage solver and its individual stages use that same
legacy optimizer contract with different components.

The rest of this chapter documents `LeastSquaresProblem` and its four
pluggable axes (Optimizer, LinearSolver, RobustKernel, DampingStrategy), then
ends with its loop sketch. These details remain current for the task facades
until their M2c rebase.

## `LeastSquaresProblem`

```python
@dataclass
class LeastSquaresProblem:
    """A least-squares problem over a flat optimisation variable x ∈ (nx,).

    - The cost stack supplies residuals r(x) and (optionally) J(x).
    - Equality / inequality constraints are exposed as cost items with
      kind="constraint_leq_zero" (Crocoddyl-like).
    - Variable bounds preserve feasibility by projecting every trial point
      before evaluation; this is not an active-set bounded solver.

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

        Metadata for future block-sparse trajopt solvers.
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
- **`jacobian_blocks(x)` exposes structure.** It is metadata for the future
  block-sparse trajopt solvers planned for M5; the shipped linear solvers are
  dense. Residuals whose
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
    residual_norm: Tensor              # (B...,) raw 0.5·||r(x)||²
    iters:         int
    damping:       float               # λ for LM; 0.0 for other solvers
    gain_ratio:    float | None = None
    status:        Literal["running", "converged", "stalled", "maxiter"] = "running"

    @classmethod
    def from_problem(cls, problem: LeastSquaresProblem) -> "SolverState": ...
    @property
    def converged(self) -> bool: ...
```

Source: `src/better_robot/optim/state.py`. The "one struct passes
through every component" pattern (cuRobo) replaces the ad-hoc tuple
returns the early prototype carried.

`stalled` is currently emitted only by LBFGS. LM and Gauss–Newton emit
`converged` or `maxiter`; in particular, bounds-limited LM progress is not
misreported as convergence and normally exhausts the budget as `maxiter`.

## Linear solvers

```python
class LinearSolver(Protocol):
    def solve(
        self,
        A: Tensor,
        b: Tensor,
        ridge: Tensor | float | None = None,
    ) -> Tensor: ...

class Cholesky(LinearSolver): ...        # dense, SPD
class LSTSQ(LinearSolver): ...           # rank-deficient safe
```

Source: `src/better_robot/optim/solvers/`.

Both solvers accept `A` with shape `(B..., n, n)` and `b` with shape
`(B..., n)`. A tensor `ridge` is scalar or broadcastable to `B...`; the solver
forms `A + ridge[..., None, None] * I` without modifying `A`. Omitting
`ridge` preserves the legacy two-argument call. `Cholesky` is the default for
dense SPD systems and retains the standalone least-squares fallback;
capture-ready LM owns its stricter `cholesky_ex` info-mask/zero-step policy.
`LSTSQ` handles rank-deficient dense systems. Iterative and structured linear
solvers are intentionally deferred to M5 instead of shipping placeholders.

## Robust kernels

```python
class RobustKernel(Protocol):
    def rho(self, squared_norm: Tensor) -> Tensor: ...
    def weight(self, squared_norm: Tensor) -> Tensor: ...

class L2(RobustKernel):    ...   # trivial identity
class Huber(RobustKernel): ...
class Cauchy(RobustKernel): ...
class Tukey(RobustKernel): ...
```

Source: `src/better_robot/optim/kernels/`.

The selected kernel is applied to each residual row after
`CostItem.weight` scaling, so thresholds such as Huber's `delta` are in
weighted residual units. Built-ins use the normalized IRLS convention
`weight(s) = 2·ρ'(s)`: residual and Jacobian rows are multiplied by
`sqrt(weight(r²))` before the linear solve. LM accepts trials using the
matching robust objective `Σ ρ(r_i²)`; without a kernel, it uses
`0.5·‖r‖²`. `SolverState.residual_norm` remains raw `0.5·‖r‖²` in both
cases.

## Damping strategies

```python
class DampingStrategy(Protocol):
    def init(self, problem) -> float: ...
    def accept(self, lam: float) -> float: ...
    def reject(self, lam: float) -> float: ...

class Constant(DampingStrategy):    ...
class Adaptive(DampingStrategy):    ...   # double on reject, halve on accept
```

Source: `src/better_robot/optim/strategies/`.

`Adaptive` is the default for LM. It starts at `1e-4`, doubles on
reject, and halves on accept. `Constant` keeps lambda fixed. The former
unimplemented trust-region placeholder was removed; the bounded second-order
algorithm is implemented as part of the new M2b solver rather than a legacy
damping strategy.

## `OptimizerConfig` — every knob is wired

`OptimizerConfig` is the user-facing dial. Every declared knob is
honoured — there are no decorative fields.

```python
@dataclass
class OptimizerConfig:
    optimizer: Literal["lm", "gn", "adam", "lbfgs", "lm_then_lbfgs"] = "lm"
    max_iter: int = 100
    jacobian_strategy: JacobianStrategy = JacobianStrategy.AUTO

    linear_solver: Literal["cholesky", "lstsq"] = "cholesky"
    kernel: Literal["l2", "huber", "cauchy", "tukey"] = "l2"
    damping: Literal["constant", "adaptive"] = "adaptive"
    tol: float = 1e-6
    refine_disabled_items: tuple[str, ...] = ()
```

`solve_ik` builds the optimiser, linear solver, robust kernel, and
damping strategy explicitly:

```python
optimizer = LevenbergMarquardt(tol=cfg.tol)  # for cfg.optimizer == "lm"
linear    = _make_linear_solver(cfg.linear_solver)
kernel    = _make_robust_kernel(cfg.kernel)
damping   = _make_damping_strategy(cfg.damping)
state = optimizer.minimize(problem,
                           linear_solver=linear,
                           kernel=kernel,
                           strategy=damping,
                           max_iter=cfg.max_iter)
```

If a knob is set but ignored by the chosen optimiser (Adam does not
take a linear solver), the build step warns rather than silently
swallowing it. Focused tests check the factories and exercise both
supported linear solvers through LM.

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
    state = SolverState.from_problem(problem)
    state.damping = strategy.init(problem)
    cost = _robust_cost(state.residual, kernel)

    for step in range(max_iter):
        J = problem.jacobian(state.x)
        r_weighted, J_weighted = _apply_kernel(state.residual, J, kernel)
        JtJ = J_weighted.mT @ J_weighted
        Jtr = J_weighted.mT @ r_weighted

        A = JtJ + state.damping * torch.eye(problem._nv, dtype=J.dtype, device=J.device)
        delta = linear_solver.solve(A, -Jtr)
        x_new = problem.step(state.x, delta)
        if problem.lower is not None:
            x_new = x_new.clamp(min=problem.lower, max=problem.upper)
        r_new = problem.residual(x_new)
        cost_new = _robust_cost(r_new, kernel)

        if cost_new < cost:
            state.x, state.residual = x_new, r_new
            state.residual_norm = 0.5 * r_new.square().sum()  # always raw L2
            state.damping = strategy.accept(state.damping)
            cost = cost_new
            if Jtr.norm() < self.tol:
                state.status = "converged"
                return state
        else:
            state.damping = strategy.reject(state.damping)

    state.status = "maxiter"
    return state
```

Source: `src/better_robot/optim/optimizers/levenberg_marquardt.py`.

Within the legacy stack, this is one loop. The four optimizers' worth of code
that the early prototype carried (our LM, PyPose LM, fixed-base autodiff LM,
floating-base analytic LM) all collapse into this with different components
plugged in.

`_robust_cost(r, kernel)` is `Σ kernel.rho(r_i²)`, or `0.5·‖r‖²`
when no kernel is selected. The actual implementation also records the
gain ratio and returns `status="converged"` when its gradient tolerance
is met.

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
from robot_descriptions import panda_description
from better_robot.residuals.pose          import PoseResidual
from better_robot.residuals.limits        import JointPositionLimit
from better_robot.residuals.regularization import RestResidual
from better_robot.costs                    import CostStack
from better_robot.optim                    import LeastSquaresProblem
from better_robot.optim.optimizers         import LevenbergMarquardt
from better_robot.optim.strategies         import Adaptive
from better_robot.optim.solvers            import Cholesky
from better_robot.optim.kernels            import Huber

model = br.load(panda_description.URDF_PATH)
hand_id = model.frame_id("body_panda_hand")

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
`jacobian_fn` argument. This remains the single legacy path from a
`CostStack` to `result.x`; the named-block `Problem` evaluation contract above
is separate until M2c.

In normal use you would never write that loop; `solve_ik` does it
internally. The example exists to show that the four pluggable axes
compose cleanly.

## Sharp edges

- **Adam and L-BFGS read `problem.gradient(x)`, not
  `problem.jacobian(x)`.** They never materialise the dense
  Jacobian. Long-horizon trajopt depends on this for memory.
- **Legacy LM bounds are projection-only.** Every trial point is projected onto
  `[lower, upper]` *before* its residual is evaluated, and acceptance is a
  bare objective comparison. There is no active set, projected-gradient
  test, or KKT termination, so a run pinned at active bounds can stall with
  residual error and exhaust its budget as `status="maxiter"`. Initial
  `x0` is **not** projected. This describes only
  `optim.optimizers.LevenbergMarquardt`; the named-block solver above owns the
  projected active set and KKT statuses.
- **`MultiStageOptimizer` restores weights / active flags via
  try/finally.** Stage-wise overrides do not leak even if a stage
  raises.
- **`OptimizerConfig` exposes only supported choices.** Dense Cholesky and
  LSTSQ are the only linear-solver choices; unimplemented iterative, sparse,
  and trust-region placeholders are not importable.

## Where to look next

- {doc}`tasks` — `solve_ik` and `solve_trajopt`, which assemble a
  `CostStack`, build a `LeastSquaresProblem`, and call an
  `Optimizer`.
- {doc}`/conventions/extension` §3, §4, §5, §6 — recipes for adding
  a custom optimiser, damping strategy, linear solver, or robust
  kernel.
- {doc}`/guides/custom_residuals` — author a residual for the named-block
  evaluation contract.
- {doc}`/conventions/performance` §2.7 — matrix-free trajopt and the
  memory wins it brings.
