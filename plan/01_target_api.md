# Target API — optimization v2

This is the design order 01 implements. It is written as the API the user
sees, then the object model behind it, then what dies. Theseus
(`references/design/theseus.md`) is the reference for the variable/residual/
weight layer; jaxopt-style value-typed internals and our batching, bounds,
banded, and implicit machinery are the parts of today's stack that stay.

Vocabulary, fixed up front: an **Optimizer** iterates to minimize
(Levenberg–Marquardt, Gauss–Newton, wrapped `torch.optim`). A **Solver**
solves a linear system exactly (Cholesky, LU, banded Cholesky) and appears
only as an optimizer ingredient. Nothing iterative is called a solver
anywhere — names, docs, file names, test names.

## 1. What the user writes

Fitting a line — smallest possible problem:

```python
import torch
from better_robot.optim import Variable, Problem, LevenbergMarquardt, residual

x = torch.tensor([0.0, 1.0, 2.0, 3.0])
y = torch.tensor([1.0, 3.0, 5.0, 7.0])
theta = Variable(torch.zeros(2), name="theta")

@residual(theta, dim=4)
def line_fit(theta):
    m, c = theta[..., 0:1], theta[..., 1:2]
    return m * x + c - y

problem = Problem([line_fit])
info = LevenbergMarquardt(problem, max_iterations=20).optimize()
print(theta.tensor)   # tensor([2., 1.])
```

IK, no facade:

```python
from better_robot.optim import RobotVariable, Problem, LevenbergMarquardt
from better_robot.residuals import PoseResidual

q = RobotVariable(model, q0, bounds=True)          # bounds=True → joint limits
reach = PoseResidual(q, frame="body_panda_hand", target=T_goal)
info = LevenbergMarquardt(Problem([reach])).optimize()
print(q.tensor)                                     # solved configuration
print(info.converged)                               # per-element, derived from status
```

A trajectory — time is part of the event, before the feature axis:

```python
q_traj = RobotVariable(model, initial_q_traj, time_axis=0)   # (..., T, nq)
smooth = VelocityResidual(q_traj, dt=dt, weight=0.1)
reach_end = PoseResidual(q_traj, frame="body_panda_hand", target=T_goal, knot=-1)
info = LevenbergMarquardt(Problem([smooth, reach_end])).optimize()
```

A custom residual with an analytic Jacobian — subclass and override:

```python
from better_robot.residuals import Residual     # also re-exported from optim

class PointDistance(Residual):
    def __init__(self, q, point, *, weight=1.0):
        super().__init__(q, dim=1, weight=weight)
        self.q, self.point = q, point

    def error(self):                       # unweighted, reads variable tensors
        p = fk_point(self.q.tensor)
        return (p - self.point).norm(dim=-1, keepdim=True)

    def jacobian(self):                    # optional; omit → autodiff
        return (d_dist_dq(self.q.tensor, self.point),)
```

Owning the loop, torch-style:

```python
from better_robot.optim import TorchOptimizer

opt = TorchOptimizer(problem, torch.optim.Adam, lr=0.05)
for _ in range(200):
    info = opt.step()
print(info.cost)
```

Batched re-solves feed new data by name, without rebuilding anything:

```python
problem.update({"q": q_batch, "target": new_targets})
info = optimizer.optimize()
```

`update` refreshes every value-derived piece of optimizer state (status,
iterations, costs, cached residuals and node outputs). A same-shaped update
may keep compatible damping state; a change of batch shape, dtype, or device
rebuilds all batch-shaped state. Structural validation of the fed tensors
happens here — `update` is a public boundary.

## 2. Variables

`optim/variables.py`. A `Variable` is a standalone object that owns its
tensor. It lives outside the `Problem` — residuals hold references to it, the
`Problem` harvests it, and optimizers write solved values back into it. Torch
users already know this object: it behaves like a parameter.

```python
class Variable:
    """A tensor with a name, an optional geometry, and a trainable flag."""
    def __init__(self, tensor, *, name=None, trainable=True,
                 bounds=None, mask=None, scale=None,
                 batch_ndim=0, time_axis=None): ...
    tensor: torch.Tensor          # current value; leading axes are batch
    name: str                     # auto-generated if omitted; unique per Problem
    trainable: bool               # False → constant (like requires_grad=False)
    # geometry hooks — Euclidean here; subclasses override:
    def tangent_dim(self) -> int ...
    def retract(self, delta) -> Tensor ...
    def difference(self, other) -> Tensor ...
```

Specialized variables carry their geometry in the type, so retraction is
never a user concern:

- `Variable` — any event shape, Euclidean. Plain data. Its constructor
  tensor's full shape is the event unless `batch_ndim` declares leading batch
  axes.
- `SO3Variable` — quaternion storage `(..., 4)`, tangent width 3.
- `SE3Variable` — pose storage `(..., 7)`, tangent width 6.
- `RobotVariable(model, tensor=None, *, bounds=None, time_axis=None, ...)` —
  storage `(..., nq)`, tangent `nv`; handles mixed revolute/spherical/
  free-flyer coordinates; `bounds=True` installs the model's joint limits
  (this replaces `RobotConfig.joint_bounds()` and the manifold class itself).

`time_axis=0` declares a leading *event* axis of knots — `(..., T, nq)` for a
trajectory — exactly today's `VarSpec.time_axis` semantics (knot-major,
separable masks, temporal tangent width). Typed variables infer their feature
width from the type; batch axes are whatever precedes the declared event.

`bounds`, `mask` (frozen tangent coordinates), and `scale` move from
`VarSpec` onto the variable — they are properties of the thing being
optimized, not of a registration record. `trainable=False` is the static
variable: residuals read it, `Problem.update` feeds it, no optimizer moves
it. Static variables have a second, load-bearing job: they are the declared
route for gradients *through* a solve. Implicit differentiation receives
exactly the graph-carrying static variables as its custom-backward inputs
(today's explicit-parameter registration, `implicit.py:442,462`) — `solve_ik`
wraps target poses and `q_rest` as `Variable(trainable=False)` for this
reason. Plain tensors are also accepted anywhere a residual wants constant
data that never needs a name or a gradient.

Extension contract: subclass `Variable`, override the three geometry hooks
(plus `project` if bounds interact with the geometry). One page of docs, no
registry.

The `Manifold` protocol and its classes (`Euclidean`, `SO3Manifold`,
`SE3Manifold`, `RobotConfig`) disappear into the variable subclasses — their
math moves, it does not shrink. `Bounds` stays as the value type.

## 3. Residuals and weights

`Residual` becomes an abstract class in `residuals/base.py` — the layer
contract ranks `residuals` below `optim` (`test_layer_dependencies.py:23`),
so the ABC lives in the lower layer and `better_robot.optim` re-exports it.
`residuals/base.py` does not import optimizer types: it holds variable
references structurally (a local protocol — has `.tensor`, `.name`,
`.trainable`) and stores `kernel`/`weight` values it does not interpret.

```python
class Residual(ABC):
    def __init__(self, *variables, dim, weight=1.0, kernel=None,
                 group_size=1, name=None): ...
    @abstractmethod
    def error(self) -> Tensor           # (..., dim), unweighted
    def jacobian(self) -> tuple[Tensor, ...] | None
        # one block per trainable variable, reduced tangent coords;
        # default None → autodiff (jacrev over the tangent)
    def weighted_error(self) -> Tensor  # weight applied; what optimizers consume
```

Rules:

- `error()` reads `self.<var>.tensor` and must be a pure function of those
  tensors. Optimizers evaluate trial points by swapping candidate tensors
  into the variables for the duration of one evaluation. An **evaluation
  epoch is one exact assignment of tensors**: node memos (below) are
  invalidated on every assignment — every LM candidate, every AD closure
  call, every finite-difference sign — and restoration runs on every exit
  path including exceptions. (Theseus solves this with copied variables; we
  swap-and-restore instead, so the epoch rule carries the correctness.)
  Residual authors never see any of this.
- `group_size` survives from `ResidualItem`: robust kernels apply per group
  of rows (contact forces use groups of 6 and 3, `contact_forces.py:334`),
  with the same positive-divisor validation.
- **Weights follow Theseus's one non-negotiable rule**: whatever multiplies
  the error multiplies the Jacobian rows by the same factor
  (square-root-information convention), in one place. A `Weight` hierarchy
  (in `residuals/base.py`, re-exported from optim) replaces today's five
  scattered `_broadcast_weight` call sites: `ScaleWeight` (scalar or batch),
  `DiagonalWeight` (per-row). Bare floats and tensors auto-wrap.
  `weighted_error`/`error` are both public — report the raw residual,
  optimize the weighted one. `problem.error()` returns the concatenated
  *weighted* residual (what optimizers consume; matches today's task
  outputs); raw values come from the residual objects.
- Weights are mutable (`reach.weight = 0.0` skips an item on the next
  evaluation), but staged presets snapshot-and-restore around each stage
  (`try/finally`) so later evaluations and final diagnostics see the original
  weights — see order 01 T5 for `lm_then_adam`.
- **Kernels stay ours**: `kernel=` at construction, default `None` meaning
  L2; LM consumes it through the existing grouped IRLS. Robust behavior is
  per-residual, not a solver mode.
- A plain function becomes a residual with the `@residual(*variables, dim=)`
  decorator (or `residual(fn, ...)` call form) — one adapter, autodiff
  Jacobian, replaces both `_CallableResidual` and trajopt's
  `_ResidualAdapter`.
- Generic residuals ship alongside the robot ones: `Difference(var, target)`
  (manifold-aware — uses `var.difference`, so it is a pose prior for
  `SE3Variable` and a plain subtraction for `Variable`), joining the existing
  regularization/limit residuals converted to the class form.

Shared computation (today's providers) becomes an explicit node object:

```python
state = RobotState(q)          # shared FK bundle for one robot variable
r1 = PoseResidual(state, frame=..., target=...)
r2 = FrameDistanceResidual(state, ...)
```

`RobotState` computes what today's `RobotStateProvider` computes — FK with
frames (`providers.py:89`); Jacobian-bearing quantities stay lazy, computed
by the residuals that need them, exactly as now — eager bundles would change
cost-only evaluation. Memoization is per evaluation epoch; graph-bearing
outputs never survive an epoch. **Merging by identity is automatic only for
`RobotState`** (key: variable identity + model identity); residual
constructors that take a bare `RobotVariable` create a private `RobotState`
merged at freeze, so the single-residual user never learns the concept.
Multi-input nodes (`SceneSDFState`, the contact-dynamics node — six-plus
inputs each, `scene_sdf.py:29`, `contact_forces.py:78`) share only when the
caller passes the same node object; no key can honestly summarize them. This
deletes `providers.py`, the `EvaluationContext` mapping, `reads`
declarations, and the auto-wiring block in `Problem.__init__` — the
variable-reference graph *is* the structure the Jacobian assembly needs.

## 4. Problem

```python
problem = Problem([r1, r2, ...])       # residuals; variables are harvested
problem.add_residual(r3)               # allowed before freeze
problem.variables                      # name → Variable (trainables + named statics)
problem.update({"q": tensor, ...})     # name-keyed data feed (see §1 semantics)
problem.error() / problem.objective()  # weighted, at current variable values
```

The problem freezes at first use by an optimizer (layout, temporal analysis,
node merging — computed once). `add_variable` is gone; a variable exists by
being referenced. The O(n²) `_rebuild`-on-every-add constructor dies with it.
`Values` (the bare dict alias) and `VarSpec` are deleted; `detach_values`
becomes unnecessary once optimizers own the write-back.

Temporal problems keep the banded mathematics unchanged, but the plumbing is
an interface rewrite, not a rename: temporal analysis and structured assembly
(`temporal.py:195,334`) consume `VarSpec.time_axis`, item reads, and residual
hooks, and `TimeIndexedResidual` (`residuals/temporal.py:26,73`) wraps the
evaluation context and slices cached `Data`. All of that re-targets the
variable/node objects. Order 01 T5 scopes it honestly.

## 5. Optimizers and solvers

```python
class Optimizer(ABC):                       # optimizers.py
    def __init__(self, problem, *, max_iterations=50, tolerance=1e-8): ...
    def step(self) -> OptimizerInfo         # one iteration; updates variables
    def optimize(self, *, verbose=False) -> OptimizerInfo
    def reset(self) -> None                 # clear internal state

class LevenbergMarquardt(Optimizer):        # lm.py
    def __init__(self, problem, *, solver="auto", damping=...,
                 max_iterations=50, tolerance=1e-8, ...): ...
    # solver: "auto" | Cholesky() | BandedCholesky() | LU()
    def optimize(self, *, verbose=False, differentiate=None) -> OptimizerInfo

class GaussNewton(LevenbergMarquardt): ...

class TorchOptimizer(Optimizer):            # optimizers.py
    def __init__(self, problem, optimizer_cls, *, max_iterations=100,
                 tolerance=0.0, **optimizer_kwargs): ...
    # TorchOptimizer(problem, torch.optim.Adam, lr=1e-2)
```

`TorchOptimizer` hosts the persistent tangent-buffer + retract + rebase
mechanics of `run_first_order` (they are correct; they move, not shrink).
Closure-free optimizers (Adam, SGD, AdamW, RMSprop, …) are the supported
family. `torch.optim.LBFGS` requires a closure and re-evaluates it several
times per step (`lbfgs.py:325,379`): `TorchOptimizer` detects
closure-requiring optimizers and supplies a closure over the summed
objective, with the documented caveat that the line search couples batch
elements (torch keeps one global history). If the solution-quality
regressions cannot be met that way, the honest error stays — order 01 T5
decides on evidence, not hope.

`OptimizerInfo` is the one result type: per-element `status`, `iterations`,
`cost`, and a derived `converged` property computed from `status` — nothing
stored that nothing reads. Solved values live in the variables; there is no
`(values, state)` pair to keep in sync. `step()` exists so users can own the
loop like a torch training loop; `optimize()` is the batteries-included
driver and the only place `verbose` printing happens (one line per
iteration: iteration, cost, converged count).

Task facades take an **optimizer factory** where they took a pre-built
optimizer: the facade constructs the `Problem`, then calls
`optimizer(problem)` — default `LevenbergMarquardt`. A pre-constructed
optimizer cannot exist before the problem does.

Two internals contracts survive unchanged beneath this surface, deliberately:

- The per-iteration LM core stays a **pure, fixed-shape, sync-free tensor
  step** (private), with per-element damping/acceptance/status exactly as
  today. `step()`/`optimize()` are thin drivers over it. This is what keeps
  `torch.compile`, warm starts, and a future Warp lane possible — Theseus's
  mutate-everything execution is the documented counterexample.
- Implicit differentiation keeps its guards and enters through
  `optimize(differentiate="implicit")`; the graph-carrying inputs it
  differentiates toward are the static variables (§2). `solve_ik(
  differentiable=True)` rewires mechanically. `TorchOptimizer` raises an
  honest error for `differentiate`.

`solvers.py` gains dense `LU` (torch `linalg.lu_factor`) beside `Cholesky`
and `BandedCholesky`; the `LinearSolver` protocol is the abstract parent.
This file is already correctly named — solvers solve exactly.

## 6. What dies (ledger rows in MIGRATION.md for each)

| Today | Fate |
|---|---|
| `VarSpec`, `Values` alias, `detach_values` | → `Variable` hierarchy (incl. `time_axis`) |
| `Manifold`, `Euclidean`, `SO3Manifold`, `SE3Manifold`, `RobotConfig` | → variable subclasses (math relocates) |
| `Problem.add_variable`, `_rebuild` constructor churn | → harvest from residuals |
| `EvaluationContext`, `providers.py`, `reads`, auto-provider block | → node objects (`RobotState`, …) |
| `ResidualItem` (incl. `group_size`) | → attributes on `Residual` (weight/kernel/group_size/name/dim) |
| `_CallableResidual`, trajopt `_ResidualAdapter` | → one `@residual` adapter |
| `LMState` (22 fields), `run/update/finalize` public lifecycle | → `OptimizerInfo` + `step()/optimize()`; the pure step and the fields it reads go private |
| `run_first_order`, `FirstOrderResult`, `OptimizerFactory` | → `TorchOptimizer` (mechanics move) |
| `autograd.tangent_grad`, `perturb_values`, `Problem.external_parameters` | public wrappers deleted (test-only callers); `_tangent_value_and_grad` stays private |
| scattered `_broadcast_weight` ×5 | → `Weight.apply_*`, one home |

Consumers (`solve_ik`, `solve_trajopt`, `solve_contact_forces`) rewrite onto
the new surface and should get shorter — IK's residual assembly becomes
variable references instead of spec/item bookkeeping, and the
`_state_iterations` hasattr ladder dies because there is one info type.
`solve_trajopt`'s new shape: the caller (or the facade, when given a bare
initial trajectory) constructs the trajectory `RobotVariable`, residuals
reference it, and the horizon derives from it.

## 7. What this is explicitly not

- Not a differentiable-optimization `nn.Module` layer (Theseus's
  `TheseusLayer`). The structure is ready for it — detached solves, implicit
  backward, variables as objects — but it is a `for_future.md` design task.
- Not an in-place mutation engine internally. Variables are the *interface*;
  the math stays value-typed underneath.
- Not a port of Theseus's sparse extlib, matrix storage, 2-D batching, or
  vectorization machinery. Our batching (arbitrary leading axes, per-element
  everything) and banded route are strictly more capable for our workloads.
