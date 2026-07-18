# Residuals, Costs, and Solvers

Optimization turns several imperfect requirements into one configuration that
balances them. A hand should reach a target, joints should stay within limits,
and a trajectory should remain smooth. BetterRobot expresses each requirement
as residual rows, combines them in a `Problem`, and solves the resulting
nonlinear least-squares problem.

The design uses one optimization surface for task helpers and direct callers.
The decisions behind that surface are {ref}`decision-residual-problems`,
{ref}`decision-variable-blocks`, and {ref}`decision-own-lm`.

## From an error to least squares

A **residual** is a signed error. If a measured position is `target` and the
model predicts `position(q)`, one residual is:

```{math}
r(q) = position(q) - target.
```

A residual can have several rows. A pose residual has translation and
rotation rows; a limit residual has rows for lower and upper violations.
Least squares makes all of them small at once:

```{math}
\min_x \frac{1}{2}\lVert r(x) \rVert^2
      = \min_x \frac{1}{2}\sum_i r_i(x)^2.
```

Squaring removes the sign and penalizes larger errors more strongly. Weights
set relative importance and convert unlike units into a useful scale. Robust
losses can reduce the influence of an outlier without changing the residual's
shape.

The [Ceres nonlinear least-squares
guide](https://ceres-solver.org/nnls_solving.html) is a useful independent
reference for residuals, parameter groups, bounds, and robust losses.

## A residual is a small callable

A residual declares a name, a fixed output width, the context entries it
reads, and a call:

```{testcode}
import torch

class PositionError:
    name = "position"
    reads = ("x", "target")
    dim = 3

    def __call__(self, ctx):
        return ctx["x"] - ctx["target"]

error = PositionError()
actual = error({"x": torch.tensor([1.0, 2.0, 3.0]), "target": torch.ones(3)})
torch.testing.assert_close(actual, torch.tensor([0.0, 1.0, 2.0]))
```

The returned tensor ends in `(dim,)` and may have leading batch axes. Fixed
width matters: Jacobian rows, robust groups, and solver state keep the same
meaning at every evaluation. Variable-size observations are padded and paired
with validity masks rather than changing the residual dimension.

An optional `jacobian_blocks(ctx)` method supplies derivatives for selected
variables. Without it, `Problem` uses automatic differentiation. Finite
differences are available only as an explicit debugging strategy. See
{doc}`kinematics_and_jacobians` for the distinction and for the documented
small-step approximations used by a few built-in blocks.

## Problems have named variables

A `Problem` owns:

- variables and their manifolds, bounds, masks, and event shapes;
- residual items with weights and robust grouping;
- constant or differentiable parameter tensors; and
- providers for shared evaluation work.

Names remove manual packing. A robot configuration can use `RobotConfig`, an
object pose can use `SE3Manifold`, and camera intrinsics can remain Euclidean.
The problem lays out tangent columns internally and returns updates under the
same names.

A plain callable is enough for a small Euclidean problem. This complete line
fit is executed by the documentation test target:

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

`add_variable` creates a variable specification. `add_residual` wraps the
callable in a residual item. Advanced construction can create `VarSpec` and
`ResidualItem` values directly.

## Providers share expensive work

Several robot residuals need the same forward kinematics. Recomputing it in
each residual would be wasteful and could produce inconsistent cache state.
A provider declares what it reads and which context values it produces.
`RobotStateProvider`, for example, produces `data` once for one problem
evaluation.

Provider outputs are cached only inside that evaluation. The next candidate
configuration gets a fresh context. Recursive resolution supports providers
that depend on other providers, while cycles raise a direct error.

When a problem has one `RobotConfig` variable and a residual reads `data`, the
usual robot-state provider is supplied automatically if no explicit provider
already produces it.

## Built-in residual families

The shipped residuals cover these roles:

| Role | Residuals |
|---|---|
| Frame targets | `PoseResidual`, `PositionResidual`, `OrientationResidual` |
| Joint preferences and limits | `JointPositionLimit`, `JointVelocityLimit`, `RestResidual`, `JointRotationPrior` |
| Trajectory structure | `ReferenceTrajectoryResidual`, `TimeIndexedResidual`, `VelocityResidual`, `AccelerationResidual` |
| Contact motion | `ContactConsistencyResidual` |
| Image observations | `ProjectionResidual` |
| Padded point sets | `MaskedChamferResidual`, `SceneSDFProvider` and its penetration, attraction, and clearance residuals |
| Spherical-joint limits | `SwingTwistLimitResidual` |

`JerkResidual` and `NullspaceResidual` are explicit placeholders that raise
`NotImplementedError`; they are not live behavior.

Residual weights belong to the surrounding `ResidualItem`. A weight of zero
as a Python number skips that item. Tensor weights may vary over the execution
batch while preserving its shape, dtype, and device.

## Robust losses

Ordinary L2 gives every squared residual its full influence. A robust kernel
keeps small errors close to L2 and reduces the influence of large ones. The
built-ins are `L2`, `Huber`, `Cauchy`, `Tukey`, and `GemanMcClure`.

`group_size` defines which contiguous residual rows form one robust unit. A
two-dimensional image observation should normally use `group_size=2`, so its
horizontal and vertical error are classified together. Grouping changes
robust weighting, not the residual layout.

Robust kernels are applied through iteratively reweighted least squares. The
problem computes an objective through `rho(squared_norm)` and scales residual
and Jacobian rows through `weight(squared_norm)` exactly once.

## Gauss--Newton in plain words

At the current value, a Jacobian replaces the nonlinear residual with a local
linear model:

```{math}
r(x + \delta) \approx r(x) + J(x)\delta.
```

Gauss--Newton chooses the tangent step that best reduces this linearized
least-squares model. It is often effective because robot fitting problems are
already written as residuals. The model is still local: a step can be poor
when the current value is far from a solution or the normal matrix is nearly
singular.

## Levenberg--Marquardt and damping

Levenberg--Marquardt, or **LM**, adds a positive damping term to the
Gauss--Newton system. Think of damping as an adjustable caution knob:

- low damping trusts a bold Gauss--Newton-like step; and
- high damping produces a smaller, more gradient-like step.

When a trial improves the objective as predicted, LM can reduce damping. When
a trial is poor or the linear solve is unhealthy, it increases damping and
tries more cautiously. Bounds use a projected active set, and manifold steps
are retracted back to valid configurations.

`GaussNewton` is a fixed, very-low-damping preset of the same guarded update.
BetterRobot owns these solvers because batched independent damping,
manifold-aware updates, robust rows, and bounds are central robotics behavior.
First-order optimizers are delegated to `torch.optim` instead.

## Owning the solver loop

LM exposes the lifecycle needed for warm starts or an application-controlled
loop:

```text
from better_robot.optim import LevenbergMarquardt

solver = LevenbergMarquardt(max_iter=50, gtol=1e-6)
state = solver.init_state(values, problem)
values, state = solver.update(values, state, problem)
values, state = solver.finalize(values, state, problem)
```

For an ordinary detached solve:

```text
values, state = solver.run(values, problem)
```

`LMState` contains tensor-valued damping, costs, convergence flags, status,
and iteration counts for every batch element. Passing a previous terminal
state to `run` preserves suitable damping information as a warm start; the
problem and values are validated again.

An initially non-finite model is reported as failed. A non-finite trial is
rejected rather than installed as the new value. Exhausting the iteration
budget returns the final candidate with a non-converged status.

## Dense and temporal linearization

Dense LM materializes the complete Jacobian or normal matrix. This is the
general correctness path and supports several variables with arbitrary
coupling.

Trajectory residuals often touch only nearby knots. A temporal variable marks
its time axis, and a residual can declare a `TemporalPattern` plus exact
per-knot numeric blocks. When every required condition is present, LM
assembles a `BlockBandedMatrix` and uses `BandedCholesky` without first
building a dense matrix.

`linearization="auto"` chooses the banded route only when the declared
structure proves it is eligible; otherwise it records a reason and uses
dense. `"structured"` requires eligibility and raises if the problem cannot
honor it. There is no inference from numerical zeros, because a value that is
zero today may be nonzero at the next iterate.

Problems with shared optimized variables outside the directly supported
temporal form remain dense. A future sparse extension should start from a real
caller and a verified elimination design, not from guessing structure.

## First-order optimization

`run_first_order` adapts the same problem to an ordinary
`torch.optim.Optimizer`:

```text
from better_robot.optim import run_first_order

values, result = run_first_order(
    values,
    problem,
    lambda parameters: torch.optim.Adam(parameters, lr=1e-2),
    max_iter=100,
    tolerance=1e-6,
)
```

The adapter keeps tangent buffers, lets the Torch optimizer update them,
retracts the result onto each manifold, and rebases the buffers without
discarding optimizer state. Adam, SGD, or another compatible optimizer can be
selected by the factory. BetterRobot does not reimplement their moment rules.

## Differentiating a solution

The default solver result is detached. Eligible converged problems can request
a guarded implicit derivative:

```text
values, state = solver.solve(values, problem, differentiate="implicit")
loss = values["q"].square().sum()
loss.backward()
```

The backward path checks convergence, active bounds, quaternion branch cuts,
robust-loss kinks, problem size, and linearization support. If those
assumptions do not hold, it raises `ImplicitDifferentiationError` instead of
returning a derivative that looks plausible but is not justified.

## Common mistakes

- Mixing quantities with different units without weights.
- Returning a residual whose last dimension changes between evaluations.
- Applying a robust kernel to scalar rows when the observation is naturally a
  vector group.
- Taking Euclidean steps in quaternion storage instead of tangent space.
- Expecting LM to guarantee a global solution to a nonlinear problem.
- Forcing the temporal route without complete structural and numeric blocks.

## Where to continue

- {doc}`kinematics_and_jacobians` explains the derivatives used here.
- {doc}`/getting_started/03_inverse_kinematics` introduces the IK facade.
- {doc}`/guides/custom_residual` builds a residual step by step.
- {doc}`/guides/own_your_optimization_loop` develops warm starts and loop
  ownership.
- {doc}`/conventions/extension` gives the exact supported extension contracts.
