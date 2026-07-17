# Named-Block Evaluation and the Legacy Solver Stack

BetterRobot exposes the named-block construction API while retaining a legacy
flat contract for direct compatibility callers:

- **Named-block optimization** is the new public construction and batched
  first-/second-order API. A `Problem` owns named `VarSpec` blocks, structural
  residuals and scalar objective terms, and a lazy provider DAG. It evaluates
  residuals, objectives, tangent gradients, and Jacobian blocks;
  matrix-free `Adam` consumes its tangent gradient, while
  `LevenbergMarquardt` and `GaussNewton` solve residual-vector problems with
  independent tensor state for every batch element.
- **The legacy solver stack** under `better_robot.optim.optimizers` composes
  `CostStack`, `LeastSquaresProblem`, and `Optimizer`. No shipped task depends
  on it; compatibility callers instantiate an optimizer and call `minimize`
  directly.

## Named-block evaluation

The canonical imports live under `better_robot.optim`; the top-level
`better_robot` namespace remains compact:

```python
from better_robot.optim import (
    Adam,
    AdamState,
    AdamStatus,
    BlockBandedMatrix,
    Bounds,
    Euclidean,
    GaussNewton,
    LevenbergMarquardt,
    LMState,
    LMStatus,
    LinearizationDecision,
    LinearizationReason,
    NormalOperator,
    ObjectiveItem,
    Phase,
    Problem,
    ResidualItem,
    RobotConfig,
    RobotStateProvider,
    SE3Manifold,
    SO3Manifold,
    TemporalPattern,
    Values,
    VarSpec,
    detach_values,
    run_phases,
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
- `structured_normal(values)`, for a directly eligible temporal problem;
- `retract(values, steps)`, which applies manifold-aware feasible steps.

Callers that need a dense normal matrix form it explicitly as
`J.mT @ J` from `dense_jacobian(values)`.

`ResidualItem.kernel` and `group_size` define robust-loss groups. Direct
`residual()` remains the weighted raw vector; `objective()` sums each group's
`rho`, and `gradient()` differentiates that same objective using the normalized
`weight(s) = 2*rho'(s)` convention. Scalar `ObjectiveItem`s keep their linear
objective weights and participate in `objective()`/`gradient()` and Adam, but
LM/GN reject them via `Problem.require_least_squares()` instead of silently
changing their mathematical meaning.

### Temporal structure and route selection

`VarSpec(..., time_axis=0)` marks the first event axis as time while preserving
the existing value, retraction, and knot-major dense-column contracts. A
residual can refine its block-level `reads` declaration with
`TemporalPattern(rows, row_width, row_origin, offsets)`. For row group `r`,
offset `o` refers to knot `r + row_origin + o`. Numeric
`temporal_jacobian_blocks` use the same offsets and return exact reduced
per-knot blocks; item weights and robust row scales are applied centrally.

`Problem` caches a static `TemporalAnalysis`. Direct eligibility requires one
free time variable, a time-separable mask, complete patterns, and complete
numeric temporal blocks. Operator eligibility permits a declaration without
numeric blocks and supplies an explicit autograd JVP/VJP fallback. Multiple
optimized variables, mixed temporal/shared dependencies, undeclared temporal
residuals, or a non-separable mask are ineligible in v1. A zero weight does
not erase structure.

`LevenbergMarquardt(linearization=...)` uses the following policy:

| Request | Representation and default solver | Ineligible behavior |
|---|---|---|
| `"dense"` | dense normal / `Cholesky` | always available |
| `"structured"` | `BlockBandedMatrix` / `BandedCholesky` | raises with cached reason/detail |
| `"matrix_free"` | `NormalOperator` / `NormalCG` | raises when operator-ineligible |
| `"auto"` | banded when directly eligible, otherwise dense | records the stable fallback reason/detail |

An explicit solver must advertise a compatible system kind. Automatic mode
does not select the more expensive autograd operator fallback. There is no
numerical-zero inference or mixed band-plus-dense container. Schur elimination
for temporal plus shared/nuisance variables remains deferred until a second
production caller supplies evidence.

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
always detached. For an implicit gradient of a converged optimum with respect
to explicitly declared external parameters, opt in separately:

```python
values, state = solver.solve(
    values,
    problem,
    differentiate="implicit",
)
loss = values["q"].square().sum()
loss.backward()
```

The forward iterations remain detached; there is no gradient to the initial
guess, warm-start state, bounds/masks, or solver hyperparameters. Backward
recomputes the exact robust tangent optimality system, maps ambient output
cotangents through each manifold retraction, eliminates stable active-bound
coordinates, and uses undamped Cholesky or a verified full-rank least-squares
solve. Any invalid element, unstable active set, Huber kink, nonfinite system,
terminal-manifold quaternion representative at the absolute-pi principal-log
cut, or singular system raises `ImplicitDifferentiationError` for the whole batch. The same tensor object
cannot be both an optimized input and an external parameter. The custom
backward is first-order only.

This initial implementation has an explicit dense-size cap (512 tangent
coordinates by default). Matrix-free backward is rejected. A small banded
forward may opt into the same capped dense correctness oracle with
`ImplicitDiffConfig(allow_banded_dense_backward=True)`; this is not a true
structured backward and long trajectories are never silently densified.
Declared context parameters are the stable supported input path. Direct
ModelValues reconstruction, named item-weight/kernel-scale binding, generic
custom-kernel kink declarations, and returned per-element gradient-quality
metadata remain follow-up work. Item/kernel object identity is not a binding;
a declared differentiable parameter disconnected from terminal optimality is
rejected instead of receiving a silent zero gradient.

The terminal-representative check cannot inspect arbitrary residual/provider
code. A relative-rotation `log` can therefore have its own principal branch
point even when the optimized quaternion is near identity; custom residual
authors must reject or avoid such points until residual-level smoothness
declarations exist.

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

`block_step_limits=(("translation", 0.2), ...)` optionally caps the physical
reduced-tangent L2 norm of individual free blocks before both the LM and
projected-gradient retractions. Gain prediction uses the actual post-
retraction tangent step. This is a solver trust-region knob, not a state bound.

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

## Named-block matrix-free Adam

`Adam` uses the same `init_state(values, problem)`, pure
`update(values, state, problem)`, and detached `run(values, problem,
state=None)` lifecycle. Its moments end in each block's mask-reduced tangent
dimension, and every step goes through the block manifold's feasible
retraction. Arbitrary common leading batch axes produce independent step,
cost, gradient-norm, convergence, and status tensors.

The update calls a prevalidated tangent VJP of the robust `Problem.objective`.
It does not call `jacobian_blocks` or `dense_jacobian`.
Warm-start moments and bias-correction counts are retained only when variable
names/order, reduced shapes, batch shape, dtype, and device match; current
target-dependent diagnostics/status are always recomputed. `run` detaches all
returned leaves and may synchronize only for its all-terminal loop check.

Named-block LBFGS is deferred rather than half-ported: correct batching needs
per-element histories and line searches plus curvature-validity/history-reset
rules. Use matrix-free `Adam` or named-block LM/GN until that dedicated
milestone lands.

## Functional phases

`Phase` combines a solver instance and iteration budget with absolute item
weight overrides, static per-block tangent masks, and an optional `on_start`
hook. `run_phases(problem, values, phases)` carries accepted values forward but
creates fresh solver state for every phase, so Adam momentum cannot leak across
a DOF transition.

Problems and variable specs are immutable, so phase application is a cheap
functional rebuild rather than in-place snapshot/restore. The caller's
`Problem` is unchanged even when a phase raises. A phase mask intersects the
base mask and therefore cannot unfreeze a permanently fixed coordinate. A
Python-zero weight removes the item from evaluation, including providers read
only by that item. `on_start` runs once even when the phase has zero iterations.

## Legacy solver stack (direct compatibility backend)

A `CostStack` knows how to compute residuals; it does not know how to minimise
them. In the legacy contract, that job belongs to `LeastSquaresProblem` (which
packs the cost stack, the initial guess, and the manifold retraction into a
single self-describing problem) and to an `Optimizer` (which iterates on it).
The canonical stack definitions live in `src/better_robot/optim/cost_stack.py`;
`better_robot.costs.stack` is an identity-preserving compatibility shim.
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
ends with its loop sketch. These details remain current for legacy direct
callers; shipped tasks use the named-block surface above.

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
        """Return the per-item J(x)^T @ r(x) helper.

        Iterates the active CostStack items and accumulates each item's
        ``apply_jac_transpose(state, r_item)`` contribution when available;
        otherwise that item materialises its Jacobian. The shipped legacy
        optimisers do not call this helper.
        """

    def jacobian_blocks(self, x: Tensor) -> dict["BlockKey", Tensor]:
        """Return weighted per-item Jacobian blocks."""
```

Source: `src/better_robot/optim/problem.py`.

The `state_factory` callable is how a problem hooks into the data
model: for an IK problem, it turns the flat `x` into a `Data` object
via `model.create_data(q=x)` and a cached FK pass. This keeps
`LeastSquaresProblem` agnostic — it does not know whether it is
solving IK, trajopt, or pose-graph SLAM.

The two extras worth highlighting:

- **`gradient(x)` is only conditionally matrix-free per item.** A residual
  with `apply_jac_transpose` can avoid its dense Jacobian; every other item
  falls back to `J.T @ r`. The legacy Adam and L-BFGS implementations call
  `jacobian(x)` directly and therefore assemble dense J. Named-block
  `better_robot.optim.Adam` is the separate, shipped matrix-free path.
- **`jacobian_blocks(x)` returns weighted per-item dense Jacobians.** Current
  legacy solvers ignore this dictionary. Temporal declarations belong to the
  separate named-block `Problem` protocol and are not inferred from these
  legacy tensors.

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
| `adam.py` | `Adam` | Legacy flat solver; currently assembles dense J |
| `lbfgs.py` | `LBFGS` | Legacy flat solver; currently assembles dense J |
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
class LSTSQ(LinearSolver): ...           # dense, rank-deficient safe
class BandedCholesky(LinearSolver): ...  # BlockBandedMatrix
class NormalCG(LinearSolver): ...        # NormalOperator, warm-start capable
```

Source: `src/better_robot/optim/solvers/`.

Every solver accepts `b` with shape `(B..., n)` and the stable
`solve(A, b, ridge=None)` seam. A tensor ridge is scalar or broadcastable to
`B...` and does not modify the caller's system. `Cholesky` and `LSTSQ` accept
dense `(B..., n, n)` tensors. `BandedCholesky` consumes padded lower
`BlockBandedMatrix` storage and returns independent per-batch SPD/finite
status. `NormalCG` consumes a sized `NormalOperator`, may use its
preconditioner and a compatible warm start, and reports fixed-work convergence
and residual diagnostics. Each implementation advertises `supported_systems`;
LM rejects an explicitly incompatible solver instead of densifying silently.

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

## IK `OptimizerConfig`

`OptimizerConfig` is the user-facing dial. Solver-selection fields are
validated at the facade boundary, and a non-default method-specific setting
is either applied by the selected method or rejected; it is not silently
ignored.

```python
@dataclass
class OptimizerConfig:
    optimizer: Literal["lm", "gn", "adam", "lbfgs",
                       "lm_then_adam", "lm_then_lbfgs"] = "lm"
    max_iter: int = 100
    jacobian_strategy: JacobianStrategy = JacobianStrategy.AUTO

    linear_solver: Literal["cholesky", "lstsq"] = "cholesky"
    kernel: Literal["l2", "huber", "cauchy", "tukey"] = "l2"
    damping: Literal["constant", "adaptive"] = "adaptive"
    tol: float = 1e-6
    refine_disabled_items: tuple[str, ...] = ()
```

`solve_ik` maps these fields to the named-block solvers explicitly:

```python
solver = LevenbergMarquardt(
    max_iter=cfg.max_iter,
    gtol=cfg.tol,
    linear_solver=_make_linear_solver(cfg.linear_solver),
    kernel=_make_robust_kernel(cfg.kernel),
    jacobian_strategy=map_strategy(cfg.jacobian_strategy),
    fixed_damping=cfg.damping == "constant",
)
values, state = solver.run(values, problem)
```

Adam intentionally has no linear-solver, Jacobian-strategy, or damping
arguments. Non-default values for those fields are rejected when
`optimizer="adam"`; Gauss--Newton likewise rejects a non-default damping
selector. `refine_disabled_items` is accepted only by `lm_then_adam`.
Named-block L-BFGS is not half-ported: `"lbfgs"` and `"lm_then_lbfgs"` raise an
actionable error, while `"lm_then_adam"` uses the functional phase engine.

## Legacy multi-stage and named-block phases

The historical `LMThenLBFGS` remains a wrapper around the legacy
`MultiStageOptimizer` for `LeastSquaresProblem`. That implementation uses its
private `_cost_stack_snapshot` context manager to save the affected item
weights/active flags and restore them in `__exit__`, including when a stage
raises. `CostStack` itself has no `snapshot()` or `restore()` methods.

New named-block code uses `Phase` and `run_phases()` instead. A phase creates a
functional `Problem` view with weight and variable-mask overrides, runs a fresh
solver state, and discards the view. Because the caller's `Problem` is
immutable and never mutated, normal return and raise paths both leave it
unchanged without a mutable snapshot protocol.

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

`solve_ik` and `solve_trajopt` convert named-block tensor state to task result
objects with scalar diagnostics for unbatched calls and per-element tensors
for batches. `TrajOptResult` additionally records
`linearization_requested`, `linearization_used`, `linearization_reason`, and
`linearization_detail`; `linearization_used` is `"dense"`, `"banded"`, or
`"matrix_free"`.

## What this lets users do

```python
import better_robot as br
from robot_descriptions import panda_description
from better_robot.residuals.pose          import PoseResidual
from better_robot.residuals.limits        import JointPositionLimit
from better_robot.residuals.regularization import RestResidual
from better_robot.optim                    import CostStack, LeastSquaresProblem
from better_robot.optim.optimizers         import LevenbergMarquardt
from better_robot.optim.strategies         import Adaptive
from better_robot.optim.solvers            import Cholesky
from better_robot.optim.kernels            import Huber

model = br.load(panda_description.URDF_PATH)
hand_id = model.frame_id("body_panda_hand")

stack = CostStack()
stack.add("pose", PoseResidual(frame_id=hand_id, target=target_pose))
stack.add("limits", JointPositionLimit(model), weight=0.1)
stack.add("rest",   RestResidual(model, model.q_neutral), weight=0.01)

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
`CostStack` to `result.x`; it remains for legacy direct use only.

The example exists to show that the four legacy pluggable axes compose
cleanly. New IK and multi-block code should use the named-block API above.

## Sharp edges

- **Matrix-free is explicit.** `better_robot.optim.Adam` uses the tangent
  objective VJP, while named-block LM uses `NormalOperator`/`NormalCG` only
  when `linearization="matrix_free"` or an explicit compatible solver selects
  that route. Automatic LM prefers direct bands and otherwise falls back to
  dense. Legacy flat optimizers still assemble dense J. Batched named-block
  LBFGS is deferred.
- **Legacy LM bounds are projection-only.** Every trial point is projected onto
  `[lower, upper]` *before* its residual is evaluated, and acceptance is a
  bare objective comparison. There is no active set, projected-gradient
  test, or KKT termination, so a run pinned at active bounds can stall with
  residual error and exhaust its budget as `status="maxiter"`. Initial
  `x0` is **not** projected. This describes only
  `optim.optimizers.LevenbergMarquardt`; the named-block solver above owns the
  projected active set and KKT statuses.
- **Legacy `MultiStageOptimizer` restores weights / active flags through its
  private context manager.** Named-block `run_phases` instead uses immutable
  functional problem views; neither path calls `CostStack.snapshot()`.
- **`OptimizerConfig` exposes only supported choices.** Dense Cholesky and
  LSTSQ are the only linear-solver choices; unimplemented iterative, sparse,
  and trust-region placeholders are not importable. Incompatible
  method-specific non-defaults fail at the `solve_ik` boundary.

## Where to look next

- {doc}`tasks` — named-block `solve_ik` and temporal `solve_trajopt` presets.
- {doc}`/conventions/extension` §3, §4, §5, §6 — recipes for adding
  a custom optimiser, damping strategy, linear solver, or robust
  kernel.
- {doc}`/guides/custom_residuals` — author a residual for the named-block
  evaluation contract.
- {doc}`/conventions/performance` §2.7 — current dense, banded, matrix-free,
  and allocation boundaries.
