# Optimization problems and solvers

BetterRobot has one optimization stack. A `Problem` owns named variable
blocks, least-squares residuals, and evaluation-local providers.
`LevenbergMarquardt` and `GaussNewton` solve dense or block-banded normal
systems; `run_first_order` adapts the same problem to any `torch.optim`
optimizer.

## The smallest complete problem

This is the complete line-fitting example. The `testcode` fence is executed by
the documentation doctest target, and the optimizer regression suite checks
that the fitted values are `(2, 1)`.

```{testcode}
import torch
from better_robot.optim import Problem, LevenbergMarquardt

x = torch.tensor([0., 1., 2., 3.]); y = torch.tensor([1., 3., 5., 7.])

def fit(ctx):
    m, c = ctx["theta"][..., 0:1], ctx["theta"][..., 1:2]
    return m * x + c - y

problem = Problem()
problem.add_variable("theta", shape=(2,))               # Euclidean by default
problem.add_residual(fit, dim=4)                        # plain callable is enough
values, state = LevenbergMarquardt().run({"theta": torch.zeros(2)}, problem)
```

`add_variable` creates the underlying `VarSpec`; `add_residual` creates the
underlying `ResidualItem`. Those records remain useful for generated or
advanced problem construction, but simple callers do not need to spell them.
A residual name defaults to its `name` attribute or callable name. A plain
callable supplies `dim=` when registering it.

The canonical public imports live under `better_robot.optim`:

```python
from better_robot.optim import (
    BandedCholesky,
    BlockBandedMatrix,
    Bounds,
    Cholesky,
    Euclidean,
    FirstOrderResult,
    GaussNewton,
    Huber,
    LevenbergMarquardt,
    LMState,
    LMStatus,
    Problem,
    ResidualItem,
    RobotConfig,
    RobotStateProvider,
    SE3Manifold,
    SO3Manifold,
    VarSpec,
    run_first_order,
)
```

`SO3Manifold` and `SE3Manifold` are optimization policies. They are distinct
from the typed Lie-group wrappers at `better_robot.lie.SO3` and
`better_robot.SE3`.

## Variables, residuals, and contexts

A variable shape is its event shape. Arbitrary leading axes in its value are
independent batch axes. A variable may specify a manifold, state-space
`Bounds`, a fixed-coordinate mask, tangent scale, and a temporal axis.
`RobotConfig(model)` handles the `nq != nv` layout and provides
`joint_bounds()` with quaternion coordinates excluded from the box.

Residuals are callables over a read-only context:

```python
class PositionError:
    name = "position"
    reads = ("x", "target")
    dim = 3

    def __call__(self, ctx):
        return ctx["x"] - ctx["target"]
```

`reads` declares Jacobian structure; it does not police runtime access. When a
problem has exactly one variable, an omitted declaration defaults to that
variable. Optional `jacobian_blocks(ctx)` and temporal hooks provide analytic
structure; otherwise `Problem` uses `jacrev`, `jacfwd`, or the explicit
finite-difference debug strategy. Jacobian strategy is a string literal:
`"auto"`, `"analytic"`, `"jacrev"`, `"jacfwd"`, or
`"finite_difference"`.

Providers declare `reads` and `outputs`, then compute a mapping. Their results
are memoized only within one evaluation context. Resolution is recursive, so
shared FK or scene queries run once per evaluation; a resolving set turns a
provider cycle into a direct error. Dependencies propagate through providers
when the optimizer determines Jacobian block structure. For a single
`RobotConfig` variable, a residual that reads `"data"` receives an automatic
`RobotStateProvider` when no explicit provider supplies it.

The main evaluation operations are:

- `residual(values)` for the weighted raw residual vector;
- `objective(values)` for the grouped robust least-squares objective;
- `gradient(values)` in reduced tangent coordinates;
- `jacobian_blocks(values)` and `dense_jacobian(values)`;
- `structured_normal(values)` for an eligible temporal problem; and
- `retract(values, steps)` for manifold-aware feasible updates.

There is no parallel scalar-objective protocol. Express an optimization term
as a residual so every solver sees the same mathematical problem.

## Dense and temporal routes

`VarSpec(..., time_axis=0)` marks time without changing knot-major value or
column ordering. A residual may refine `reads` with a `TemporalPattern` and
exact per-knot numeric Jacobian blocks. A directly eligible problem has one
free temporal variable, a separable mask, complete patterns, and complete
numeric blocks.

`LevenbergMarquardt(linearization=...)` resolves only two representations:

| Request | Representation | Default solver | Ineligible behavior |
|---|---|---|---|
| `"dense"` | dense normal matrix | `Cholesky` | always available |
| `"structured"` | `BlockBandedMatrix` | `BandedCholesky` | raises with a stable reason and detail |
| `"auto"` | banded when directly eligible, otherwise dense | matching solver | records the fallback reason and detail |

There is no numerical-zero structure inference and no mixed band-plus-dense
container. Temporal plus shared optimized variables remain dense until a
reviewed Schur-complement design has a production caller.

## LM and Gauss--Newton

LM exposes an explicit lifecycle for consumers that own the loop:

```python
solver = LevenbergMarquardt(max_iter=50, gtol=1e-6)
state = solver.init_state(values, problem)
values, state = solver.update(values, state, problem)
values, state = solver.finalize(values, state, problem)

# Or use the detached eager driver.
values, state = solver.run(values, problem)
```

`LMState` is tensor-only and preserves every batch axis. It retains state
needed for damping warm starts plus cost, residual, KKT, factorization,
convergence, status, and iteration data. `GaussNewton` is a fixed-damping
preset of the same guarded update.

Bounds use a projected active set and projected-gradient KKT termination.
`block_step_limits` optionally caps physical tangent-block norms. An initially
non-finite model is `FAILED`; a non-finite trial is rejected and can
legitimately finish `MAXITER`.

For a first-order implicit gradient of an eligible converged optimum, use:

```python
values, state = solver.solve(values, problem, differentiate="implicit")
loss = values["q"].square().sum()
loss.backward()
```

The forward solve remains detached. Backward keeps the convergence,
active-set, quaternion-cut, robust-kink, size, and routing guards documented
by `ImplicitDiffConfig`; violating a guard raises
`ImplicitDifferentiationError` instead of returning a silently wrong
gradient.

## First-order optimization through `torch.optim`

`run_first_order` holds persistent tangent buffers, lets an ordinary
`torch.optim.Optimizer` update them, retracts onto each manifold, and rebases
the buffers without discarding optimizer state:

```python
values, result = run_first_order(
    values,
    problem,
    lambda parameters: torch.optim.Adam(parameters, lr=1e-2),
    max_iter=100,
    tolerance=1e-6,
)
```

The factory may return Adam, SGD, or another compatible Torch optimizer.
`FirstOrderResult` reports detached per-element `step`, `converged`, and
`cost` tensors. The adapter deliberately does not reproduce custom warm-start
moments, per-element bias-correction counters, or atomic non-finite rollback.

`solve_ik(..., optimizer="lm_then_adam")` performs two sequential calls. It
rebuilds the refinement problem with any `refine_disabled_items` set to zero
weight; there is no general phase engine.

## Robust kernels and linear solvers

`ResidualItem.kernel` and `group_size` define contiguous robust groups. The
built-ins are `L2`, `Huber`, `Cauchy`, `Tukey`, and `GemanMcClure`; they use
the normalized IRLS convention `weight(s) = 2*rho'(s)`. Thresholds are in
weighted residual units.

Dense LM uses strict SPD `Cholesky`; temporal LM uses
`BandedCholesky`. A failed Cholesky factorization is reported to LM so damping
can increase. Rank-deficient least-squares fallback is not part of the linear
solver contract.

## Task results

`solve_ik` and `solve_trajopt` translate tensor solver state into task result
objects. Unbatched diagnostics are Python scalars; batched diagnostics retain
their leading axes. `TrajOptResult.linearization_used` is `"dense"` or
`"banded"`, with the requested route and stable reason/detail alongside it.

## Where to look next

- {doc}`tasks` — IK, trajectory optimization, and contact-force presets.
- {doc}`/guides/custom_residuals` — author a structural residual and provider.
- {doc}`/conventions/extension` — supported residual, provider, kernel, and
  linear-solver seams.
- {doc}`/conventions/performance` — allocation and capture boundaries.
