# Residuals, Costs, and Optimizers

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

Squaring removes the sign and penalizes larger errors more strongly. A useful
objective must distinguish three ideas that are easy to conflate:

| Control | Purpose |
|---|---|
| `row_weight` | Whitens rows, for example by measurement standard deviation or physical units. |
| `weight` | Sets the outer importance of a term without changing its robust-loss scale. |
| `kernel` | Controls how the term treats outlying robust groups. |

After whitening, consecutive rows are grouped according to `group_size`.
BetterRobot evaluates each residual with exactly this algebra:

```text
rows = row_weight.apply(error())
cost = Σ_k active_k · w_k · ρ(‖rows_k‖²) · norm
```

Here `w_k` comes from the residual's non-negative outer `weight`. `active_k`
is a boolean group mask. The reduction factor `norm` is `1` for
`reduce="sum"`, `1 / n_groups` for `"mean"`, or
`1 / clamp(Σ_k active_k, 1)` for `"mean_active"`. The mask and active count
are detached: they gate an objective but are not differentiated.

The ordinary L2 kernel is `ρ(s) = 0.5 · s`. Its exact term is therefore
`0.5 · Σ_k active_k · w_k · ‖rows_k‖² · norm`. The `0.5` is part
of the public objective convention, not an implementation detail.

The [Ceres nonlinear least-squares
guide](https://ceres-solver.org/nnls_solving.html) is a useful independent
reference for residuals, parameter groups, bounds, and robust losses.

## A residual holds its dependencies

A residual owns references to every variable it reads, plus its name, fixed
output width, row whitening, outer weight, reduction, activity, robust kernel,
and grouping. The `@residual` adapter is enough for a small tensor function:

```{testcode}
import torch
from better_robot.optim import Variable, residual

x = Variable(torch.tensor([1.0, 2.0, 3.0]), name="x")
target = Variable(torch.ones(3), name="target", trainable=False)


@residual(x, target, dim=3, name="position")
def position_error(value, desired):
    return value - desired


actual = position_error.error()
print(actual.tolist())
```

```{testoutput}
[0.0, 1.0, 2.0]
```

The returned tensor ends in `(dim,)` and may have leading batch axes. Fixed
width matters: Jacobian rows, robust groups, and optimizer storage keep the same
meaning at every evaluation. Variable-size observations are padded and paired
with validity masks rather than changing the residual dimension.

An optional `jacobian()` method returns complete tangent blocks in
dependency order. Without it, the default `"auto"` strategy uses either
`torch.func.jacrev` or `torch.func.jacfwd` and emits one
`AutodiffFallbackWarning` per residual. The warning makes a potentially slower
but numerically correct fallback visible and names the selected transform.
Providing `jacobian()` or choosing an explicit differentiation strategy makes
that trade-off intentional and suppresses the warning. Finite differences are
available only as an explicit debugging strategy. See
{doc}`kinematics_and_jacobians` for the distinction and for the documented
small-step approximations used by a few built-in blocks.

## Problems harvest an object graph

A `Problem` freezes and lays out:

- trainable and static variables referenced by residuals or nodes;
- fixed residual rows, row weights, outer coefficients, reductions, kernels,
  and robust groups; and
- evaluation-scoped nodes for shared computation.

Each `Variable` owns its tensor and geometry. Use `RobotVariable` for a robot
configuration, `SE3Variable` for an object pose, and plain `Variable` for a
Euclidean block such as camera intrinsics. Stable names make diagnostics and
atomic `Problem.update()` calls readable without a separate packing schema.

A decorated tensor function is enough for a small Euclidean problem. This
complete line fit is executed by the documentation test target:

```{testcode}
import torch
from better_robot.optim import LevenbergMarquardt, Problem, Variable, residual

x = torch.tensor([0., 1., 2., 3.], dtype=torch.float64)
y = torch.tensor([1., 3., 5., 7.], dtype=torch.float64)
theta = Variable(torch.zeros(2, dtype=torch.float64), name="theta")
observations = Variable(y, name="observations", trainable=False)


@residual(theta, observations, dim=4)
def fit(parameters, measured):
    m, c = parameters[..., 0:1], parameters[..., 1:2]
    return m * x + c - measured


problem = Problem([fit])
info = LevenbergMarquardt(
    problem,
    max_iterations=20,
    tolerance=1e-9,
).optimize()
print("solution:", theta.tensor.round(decimals=6).tolist())
print("converged:", bool(info.converged))
```

```{testoutput}
solution: [2.0, 1.0]
converged: True
```

Subclass `Residual` when an error term needs analytic or temporal blocks, a
custom weight, or a shared node.

## Nodes share expensive work

Several robot residuals need the same forward kinematics. Recomputing it in
each residual would be wasteful and could produce inconsistent cache state.
A `Node` holds its input variables and computes a lazy value. `RobotState`,
for example, owns one `RobotVariable` reference and provides FK data.

A residual lists shared nodes in `nodes` and reads `node.value()` in
`error()` or `jacobian()`. `Problem` merges compatible nodes when it freezes,
invalidates their memos at every evaluation boundary, and therefore computes
shared graph-bearing work at most once per evaluation without carrying it to
the next candidate.

## Built-in residual families

The shipped residuals cover these roles:

| Role | Residuals |
|---|---|
| Frame targets | `PoseResidual`, `PositionResidual`, `OrientationResidual` |
| Joint preferences and limits | `JointPositionLimit`, `JointVelocityLimit`, `RestResidual`, `JointRotationPrior` |
| Trajectory structure | `ReferenceTrajectoryResidual`, `TimeIndexedResidual`, `VelocityResidual`, `AccelerationResidual` |
| Contact motion | `ContactConsistencyResidual` |
| Image observations | `ProjectionResidual` |
| Padded point sets | `MaskedChamferResidual`, `SceneSDFState` and its penetration, attraction, and clearance residuals |
| Spherical-joint limits | `SwingTwistLimitResidual` |

Every residual listed above is live. Unimplemented residual ideas are omitted
from the API until their mathematical and temporal contracts are defined.

Residual controls live on the `Residual` itself. `enabled=False` or a Python
numeric zero outer weight skips a residual without changing its reserved row
layout. Tensor outer weights remain graph-visible and may provide one
coefficient globally, per execution-batch element, or per robust group.

`Problem.error()` returns the concatenated `row_weight`-whitened rows. It does
not fold outer weights, reductions, group activity, or robust kernels into
those rows. That separation keeps the diagnostic vector meaningful when a
robust objective cannot be represented by one scaled residual vector. Use
`Problem.objective()` for the scalar objective and `Problem.term_costs()` for
one named contribution per residual; the term costs sum to the objective. The
`residual` fields returned by IK, trajectory optimization, and contact-force
tasks copy the same whitened diagnostic rows.

## Robust losses

Ordinary L2 gives every squared residual its full influence. A robust kernel
keeps small errors close to L2 and reduces the influence of large ones. The
built-ins are `L2`, `Huber`, `Cauchy`, `Tukey`, and `GemanMcClure`.

`group_size` defines which contiguous residual rows form one robust unit. A
two-dimensional image observation should normally use `group_size=2`, so its
horizontal and vertical error are classified together. Grouping changes
robust weighting, not the residual layout.

Robust kernels are applied through iteratively reweighted least squares
(IRLS). The problem computes the exact objective through `rho(squared_norm)`.
LM and GN scale a group's residual and Jacobian rows by
`sqrt(active_k · w_k · norm · kernel.weight(squared_norm))`.

This is uncorrected IRLS; it does not apply a Triggs second-order correction.
It is gradient-consistent while the active set is fixed. At an activity
threshold, the detached mask and detached `mean_active` count make the
objective non-differentiable, so threshold crossings are judged by descent
rather than a derivative equality.

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
BetterRobot owns these optimizers because batched independent damping,
manifold-aware updates, robust rows, and bounds are central robotics behavior.
First-order optimizers are delegated to `torch.optim` instead.

## Owning the optimizer loop

Every optimizer owns one problem. `step()` advances referenced variables once;
`optimize()` runs the complete eager loop:

```{testcode}
problem.update({"theta": torch.zeros_like(theta.tensor)})
optimizer = LevenbergMarquardt(
    problem,
    max_iterations=50,
    tolerance=1e-9,
)
initial_cost = problem.objective()
step_info = optimizer.step()
print("one step reduced cost:", bool((step_info.cost < initial_cost).all()))
```

```{testoutput}
one step reduced cost: True
```

For an ordinary detached solve:

```{testcode}
info = optimizer.optimize()
solution = theta.tensor
print("detached solution:", solution.round(decimals=6).tolist())
print("converged:", bool(info.converged))
```

```{testoutput}
detached solution: [2.0, 1.0]
converged: True
```

`OptimizerInfo` contains only per-element status, iterations, cost, and a
derived convergence flag. Detailed LM state is private. `Problem.update()`
changes current variable tensors atomically; LM refreshes the changed graph
while retaining compatible damping. `optimizer.reset()` clears optimizer
state but deliberately retains current variable values.

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
dense. A residual that declares temporal structure without providing temporal
Jacobian blocks emits one `AutodiffFallbackWarning` when this forces automatic
dense routing. Providing the blocks or explicitly choosing dense linearization
makes the cost visible and suppresses the warning. `"structured"` requires
eligibility and raises if the problem cannot honor it. There is no inference
from numerical zeros, because a value that is zero today may be nonzero at the
next iterate.

Problems with shared optimized variables outside the directly supported
temporal form remain dense. A future sparse extension should start from a real
caller and a verified elimination design, not from guessing structure.

## First-order optimization

`TorchOptimizer` adapts the same problem to an ordinary
`torch.optim.Optimizer` class or factory:

```{testcode}
from better_robot.optim import TorchOptimizer

adam_value = Variable(torch.tensor([0.0], dtype=torch.float64), name="adam_value")
adam_target = Variable(
    torch.tensor([3.0], dtype=torch.float64),
    name="adam_target",
    trainable=False,
)


@residual(adam_value, adam_target, dim=1)
def adam_error(value, desired):
    return value - desired


adam_problem = Problem([adam_error])
adam = TorchOptimizer(
    adam_problem,
    torch.optim.Adam,
    lr=0.1,
    max_iterations=250,
    tolerance=1e-6,
)
adam_info = adam.optimize()
near_target = torch.allclose(adam_value.tensor, adam_target.tensor, atol=1e-3, rtol=0.0)
print("Adam reached target:", bool(near_target))
print("Adam converged:", bool(adam_info.converged))
```

```{testoutput}
Adam reached target: True
Adam converged: True
```

The adapter keeps tangent buffers, lets the Torch optimizer update them,
retracts through each variable's geometry, and rebases the buffers without
discarding optimizer state. Adam, SGD, or another compatible optimizer can be
selected by the factory. BetterRobot does not reimplement their moment rules.

## Differentiating a solution

The default optimizer result is detached. Eligible converged problems can
request a guarded implicit derivative:

```{testcode}
target_tensor = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
q = Variable(torch.zeros(1, dtype=torch.float64), name="q")
target = Variable(target_tensor, name="implicit_target", trainable=False)


@residual(q, target, dim=1)
def target_error(value, desired):
    return value - desired


implicit_problem = Problem([target_error])
lm = LevenbergMarquardt(implicit_problem, max_iterations=20, tolerance=1e-10)
implicit_info = lm.optimize(differentiate="implicit")
loss = q.tensor.square().sum()
loss.backward()
print("implicit solution:", q.tensor.detach().round(decimals=6).tolist())
print("target gradient:", target_tensor.grad.round(decimals=6).tolist())
print("implicit converged:", bool(implicit_info.converged))
```

```{testoutput}
implicit solution: [2.0]
target gradient: [4.0]
implicit converged: True
```

Graph-carrying static variables are the differentiable inputs. The backward
path checks convergence, active bounds, quaternion branch cuts, robust-loss
kinks, problem size, and linearization support. If those assumptions do not
hold, it raises `ImplicitDifferentiationError` instead of returning a
derivative that looks plausible but is not justified.

Implicit differentiation is unavailable when any residual uses
`reduce="mean_active"` or overrides `active_groups()`. Those features depend
on detached state-dependent activity, so there is no consistent implicit
derivative. The optimize request raises a `ValueError` naming the residual
that makes the problem ineligible.

## Common mistakes

- Using outer `weight` to convert units instead of `row_weight`.
- Reading `Problem.error()` as an objective-scaled residual vector.
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
