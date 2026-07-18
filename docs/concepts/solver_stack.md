# Named-Block Evaluation and Solvers

BetterRobot exposes one optimization surface. A `Problem` owns named `VarSpec`
blocks, structural residuals and scalar objective terms, and a lazy provider
DAG. It evaluates residuals, objectives, tangent gradients, and Jacobian
blocks; matrix-free `Adam` consumes its tangent gradient, while
`LevenbergMarquardt` and `GaussNewton` solve residual-vector problems with
independent tensor state for every batch element.

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
iteration. No public captured-execution driver is shipped. Custom residuals
and providers must satisfy the fixed-shape, sync-free eligibility rules in
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

Each kernel is attached to a `ResidualItem` and applied after item-weight
scaling, so thresholds such as Huber's `delta` are in weighted residual units.
Built-ins use the normalized IRLS convention `weight(s) = 2·ρ'(s)`: residual
and Jacobian rows are multiplied by `sqrt(weight(r²))` before the linear solve.
`Problem.objective()` evaluates the matching grouped robust objective, while
`Problem.residual()` preserves the weighted raw residual vector.

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

## Task results

`solve_ik` and `solve_trajopt` convert named-block tensor state to task result
objects with scalar diagnostics for unbatched calls and per-element tensors
for batches. `TrajOptResult` additionally records
`linearization_requested`, `linearization_used`, `linearization_reason`, and
`linearization_detail`; `linearization_used` is `"dense"`, `"banded"`, or
`"matrix_free"`.

## Sharp edges

- **Matrix-free is explicit.** `better_robot.optim.Adam` uses the tangent
  objective VJP, while named-block LM uses `NormalOperator`/`NormalCG` only
  when `linearization="matrix_free"` or an explicit compatible solver selects
  that route. Automatic LM prefers direct bands and otherwise falls back to
  dense. Batched named-block LBFGS is deferred.
- **`OptimizerConfig` exposes only supported choices.** Dense Cholesky and
  LSTSQ are the only linear-solver choices; unimplemented iterative, sparse,
  and trust-region placeholders are not importable. Incompatible
  method-specific non-defaults fail at the `solve_ik` boundary.

## Where to look next

- {doc}`tasks` — named-block `solve_ik` and temporal `solve_trajopt` presets.
- {doc}`/conventions/extension` — supported linear-solver and robust-kernel
  extension seams.
- {doc}`/guides/custom_residuals` — author a residual for the named-block
  evaluation contract.
- {doc}`/conventions/performance` §2.7 — current dense, banded, matrix-free,
  and allocation boundaries.
