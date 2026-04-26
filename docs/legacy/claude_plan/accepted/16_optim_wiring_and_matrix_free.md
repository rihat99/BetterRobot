# 16 · Optimization wiring, residual specs, and matrix-free trajopt

★★ **Structural.** Closes a real gap: `solve_ik`'s config knobs are
declared but not wired through; trajectory optimisation needs
sparse / matrix-free paths from day one. Lands the
`ResidualSpec`, `TrajectoryParameterization`, and
`MultiStageOptimizer` concepts that the gpt-plan review surfaced
and that this folder did not previously cover end-to-end.

## Problem

The optimisation stack has good bones —
[07 Residuals/Costs/Solvers](../../design/07_RESIDUALS_COSTS_SOLVERS.md)
already specifies the `Residual` Protocol with optional analytic
`.jacobian()`, the `CostStack` with named/weighted/activatable
items, and the `LeastSquaresProblem` shape. What is missing falls
into three categories:

### 16.1 Config knobs declared but not wired

`tasks/ik.py` defines:

```python
@dataclass(frozen=True)
class OptimizerConfig:
    optimizer: str = "lm"
    max_iter: int = 100
    jacobian_strategy: JacobianStrategy = JacobianStrategy.AUTO
    linear_solver: str = "cholesky"     # accepted, NOT WIRED
    kernel: str | None = None           # accepted, NOT WIRED
    damping: float | str = 1e-4         # accepted as float, dispatch missing
    ...
```

`solve_ik` reads `optimizer_cfg.optimizer` and a handful of
others, but `linear_solver`, `kernel`, and the dispatch on
`damping` are dropped on the floor. A user who sets
`OptimizerConfig(linear_solver="qr", kernel="huber")` gets the same
solve as the default — silently. The right answer is to wire them
through (or remove them until they are real).

### 16.2 Robust kernels are decoupled from the solver

`optim/kernels/` defines `Identity`, `Huber`, `Cauchy`, `Tukey`. The
LM and GN loops never read them. `CostStack` applies a scalar
weight per item; the robust kernel is an additional scalar applied
to the squared residual at solver time, and the weights are not the
same thing. The two are conflated today.

### 16.3 Dense Jacobians don't scale to long-horizon trajopt

`solve_trajopt` flattens `(B, T, nq)` into `(B, T*nq)` and feeds
it to the existing IK solver. For a 200-knot Panda trajectory the
flattened Jacobian is `(dim, T*nv) = (≥1000, 1400)` — dense. For a
G1 humanoid at the same horizon this is many GB of fp32. We
materialise it because the solver needs `J` and `J^T J`. Most of
those entries are *zero* (each pose-keyframe residual hits one
knot; each smoothness residual hits a 5-knot stencil), but our
solver doesn't know that.

The right path here is well-known from Crocoddyl, Drake, and the
gpt-plan review:

- residuals declare a `ResidualSpec` describing block / temporal
  structure;
- `LeastSquaresProblem` exposes a matrix-free `gradient(x)` that
  returns `J^T r` directly;
- Adam and LBFGS read `gradient(x)`, not `J(x)`;
- LM uses block-sparse Cholesky for assembled trajopt;
- the trajectory variable can be parameterised — knot points are
  one parameterisation; B-spline control points are another.

### 16.4 Multi-stage solves are a one-off, not a primitive

`LMThenLBFGS` exists. It is the only multi-stage thing in the
codebase. Real optimisation pipelines often want **per-stage cost
weights** (coarse pose first, smoothness second, collision third)
or **per-stage active-set changes**. The gpt-plan review proposed
generalising `LMThenLBFGS` into a `MultiStageOptimizer` with
explicit stages — a small refactor that removes the special case.

## Goal

A self-contained optimisation surface that:

1. Has every config knob wired or removed.
2. Distinguishes residual weights, robust kernels, and active /
   inactive items as three independent concepts.
3. Supports matrix-free `J^T r` evaluation as a first-class path.
4. Exposes a `ResidualSpec` so solvers can build sparse blocks.
5. Has a `TrajectoryParameterization` Protocol with `KnotTrajectory`
   and `BSplineTrajectory` implementations.
6. Generalises `LMThenLBFGS` into `MultiStageOptimizer`.

This is the gpt-plan-driven addition to the proposal set; it sits
*between* trajectory locking ([Proposal 08](08_trajectory_lock_in.md))
and the dynamics milestone plan
([Proposal 14](14_dynamics_milestone_plan.md)).

## The proposal

### 16.A Wire the IK optimizer config

`solve_ik` extends to honour every documented field:

```python
def solve_ik(
    model: Model,
    targets: dict[str, "SE3 | Float[Tensor, '*B 7']"],
    *,
    initial_q: ConfigTensor | None = None,
    cost_cfg: IKCostConfig = IKCostConfig(),
    optimizer_cfg: OptimizerConfig = OptimizerConfig(),
) -> IKResult:
    cost_stack = _assemble_cost_stack(model, targets, cost_cfg)
    problem    = _build_least_squares_problem(model, cost_stack, initial_q)

    optimizer  = _make_optimizer(optimizer_cfg)        # honours .optimizer
    linear     = _make_linear_solver(optimizer_cfg)    # honours .linear_solver
    kernel     = _make_robust_kernel(optimizer_cfg)    # honours .kernel
    damping    = _make_damping_strategy(optimizer_cfg) # honours .damping

    state = optimizer.run(
        problem,
        linear_solver=linear,
        robust_kernel=kernel,
        damping=damping,
        max_iter=optimizer_cfg.max_iter,
    )
    return IKResult.from_state(state, model=model, targets=targets)
```

`linear_solver` accepts `"cholesky" | "qr" | "lsqr" | "block_cholesky"`.
`kernel` accepts `"identity" | "huber" | "cauchy" | "tukey"` plus
optional kernel-specific params (`huber_delta=`, `tukey_c=`).
`damping` accepts `float | "adaptive" | DampingStrategy`.

If a knob is *known* but *not yet implemented* on a particular
optimizer (e.g. Adam doesn't take a linear solver), the field is
either ignored with a `UserWarning` ("Adam does not use
`linear_solver=qr`; ignored") or rejected at config-build time
depending on which is louder for the user. We default to warning,
not silent.

### 16.B Three concepts that look similar but are not

```text
                cost item active?         (structural inclusion)
                       │
                       ▼
   raw residual ─→ × weight ─→ × robust kernel ─→ accumulate
       │                              │
       └ Residual.__call__            └ optim/kernels/*.py
       └ analytic .jacobian()         └ acts on (residual, jacobian)
                                        in the LM/GN normal eqn
```

| Concept | Layer | What it does | Stable across iterations? |
|---------|-------|--------------|---------------------------|
| Active flag | `CostStack` | Include / exclude an item structurally. | **Yes** — toggling re-validates problem dim. |
| Weight | `CostStack` | Scalar multiplier; allows soft-priority. | Yes — but `weight = 0` is *not* the same as inactive. |
| Robust kernel | `optim/kernels/` | Reweights the squared-error contribution at solver time. | Yes per kernel; some have hyperparameters. |

The critical rule: **`weight = 0` is *not* a valid way to disable
an item**. A zero-weight item still occupies a slot in the
preallocated residual / Jacobian; the active flag is what
structurally removes it. This is the gpt-plan-flagged subtlety —
sparse trajopt depends on it being right.

### 16.C `ResidualSpec` — the structural metadata

[07 §7](../../design/07_RESIDUALS_COSTS_SOLVERS.md) already names
`ResidualSpec` for `time_coupling`. This proposal pins the full
shape:

```python
@dataclass(frozen=True)
class ResidualSpec:
    """Structural metadata about a residual.

    Returned by ``Residual.spec(state)``. Used by solvers to plan
    Jacobian assembly, sparse-block storage, and active-set updates.
    """
    output_dim: int
    tangent_dim: int

    # Block / temporal structure.
    structure: Literal[
        "dense",          # full (output_dim, tangent_dim) Jacobian.
        "block",          # one or more (out, in) blocks at named indices.
        "banded",         # banded along a temporal axis.
        "matrix_free",    # only J^T r is available; J is never built.
    ]

    # Time coupling for trajopt residuals.
    time_coupling: Literal["single", "5-point", "custom"] | None = None
    affected_knots: tuple[int, ...] | None = None     # for "block" trajopt residuals

    # Spatial / kinematic locality.
    affected_joints: tuple[int, ...] | None = None
    affected_frames: tuple[int, ...] | None = None

    # Whether output_dim depends on the input (e.g. active-pair collision).
    # If True, a stable upper-bound dimension still exists; the *active*
    # subset varies. Solvers preallocate to the upper bound and zero out
    # inactive rows.
    dynamic_dim: bool = False
```

The `Residual` Protocol grows the optional method:

```python
class Residual(Protocol):
    name: str
    dim: int

    def __call__(self, state: ResidualState) -> torch.Tensor: ...
    def jacobian(self, state: ResidualState) -> torch.Tensor | None: ...

    def spec(self, state: ResidualState) -> ResidualSpec: ...
    def apply_jac_transpose(
        self,
        state: ResidualState,
        vec: torch.Tensor,
    ) -> torch.Tensor:
        """Compute ``J(x)^T @ vec`` without materialising ``J``.
        Default implementation: build ``J = self.jacobian(state)`` and
        return ``J.T @ vec``. Override for memory-efficient paths."""
```

`spec` lands with a default implementation (`structure="dense"`,
`output_dim=dim`, etc.) so existing residuals do not need to
change.

### 16.D Matrix-free `LeastSquaresProblem.gradient(x)`

```python
class LeastSquaresProblem:
    # existing residual / jacobian / state_factory …

    def gradient(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``J(x)^T @ r(x)`` using each residual's
        ``apply_jac_transpose`` if available, else falling back to
        ``J.T @ r`` via the dense path."""
        state = self.state_factory(x)
        grad  = torch.zeros_like(x)
        for item in self.cost_stack.active_items():
            r = item(state)                      # weighted residual
            grad = grad + item.apply_jac_transpose(state, r)
        return grad
```

`Adam` and `LBFGS` then read `problem.gradient(x)` rather than
materialising a full Jacobian — which is the existing optim
behaviour, except it now goes through a typed contract instead of a
side-channel. The same path becomes the **default for trajopt**
unless the user explicitly chooses an LM-with-block-Cholesky path.

`LM` and `GN` continue to need `J^T J + λ I` assembly. For dense
Jacobians they call `problem.jacobian(x)`. For block-structured
trajopt they call a new
`problem.jacobian_blocks(x) -> dict[BlockKey, Tensor]` and the
linear solver is `block_cholesky`. The block solver is one of the
linear-solver options exposed by 16.A.

### 16.E Trajectory parameterisations

`Trajectory` ([Proposal 08](08_trajectory_lock_in.md)) is the
sample-at-knots representation. Optimisation often works in a
*lower-dimensional* parameter space — control points of a B-spline,
coefficients of a basis. The Protocol:

```python
class TrajectoryParameterization(Protocol):
    """Maps a flat optimization variable z to a Trajectory.

    The solver works in z-space; residuals consume the unpacked
    Trajectory. The mapping is differentiable: ``J_traj_wrt_z`` is
    used by the chain rule when residuals' Jacobians w.r.t. q are
    composed back to z.
    """
    @property
    def tangent_dim(self) -> int: ...
    def unpack(self, z: torch.Tensor) -> "Trajectory": ...
    def retract(self, z: torch.Tensor, dz: torch.Tensor) -> torch.Tensor: ...
    def pack_initial(self, traj: "Trajectory") -> torch.Tensor: ...
```

Two concrete implementations land in v1:

```python
class KnotTrajectory(TrajectoryParameterization):
    """The variable IS the trajectory — z and traj.q identify after
    a reshape. Right for dense exact-per-timestep constraints."""

class BSplineTrajectory(TrajectoryParameterization):
    """The variable is a (B..., K, nq) control-point grid; the
    trajectory is reconstructed via a fixed B-spline basis at given
    knot times. Right for smooth motion generation; way fewer
    optimization variables for the same horizon."""
```

`solve_trajopt` accepts a `parameterization=` kwarg defaulting to
`KnotTrajectory`. The cuRobo lesson: B-spline parameterisation with
ghost / control points at the endpoints is the right default for
smooth motion. We ship it but do not force it.

The parameterisation also defines its own retraction so manifold
bits of `z` (e.g. SO(3) blocks of a control point) are retracted
correctly; the LM linear solver works in `dz`-space and the
parameterisation handles the manifold step.

### 16.F `MultiStageOptimizer` generalises `LMThenLBFGS`

```python
@dataclass(frozen=True)
class OptimizerStage:
    optimizer: Optimizer
    max_iter: int
    active_items: tuple[str, ...] | None = None      # if not None, replace active set
    disabled_items: tuple[str, ...] = ()
    weight_overrides: dict[str, float] | None = None  # e.g. {"smooth": 0.0}
    tol: float | None = None

@dataclass(frozen=True)
class MultiStageOptimizerConfig:
    stages: tuple[OptimizerStage, ...]

class MultiStageOptimizer(Optimizer):
    """Run a fixed sequence of solver stages. Each stage may toggle
    cost-stack active flags or override item weights; stage exit is
    ``max_iter`` or ``tol`` whichever first.

    Existing ``LMThenLBFGS`` is implemented in terms of this:
        MultiStageOptimizer(stages=(
            OptimizerStage(LM(), max_iter=50, tol=1e-6),
            OptimizerStage(LBFGS(), max_iter=100, tol=1e-8),
        ))
    """
```

Important detail (gpt-plan flag): the multi-stage optimizer must
**restore active-flag and weight-override state** even if a stage
raises. A `try/finally` around the `cost_stack.snapshot()` /
`.restore()` pattern keeps the user's `CostStack` intact regardless
of outcome.

### 16.G Solver-component contracts

Define and test the protocols already named in
[07_RESIDUALS_COSTS_SOLVERS.md](../../design/07_RESIDUALS_COSTS_SOLVERS.md):

```python
class LinearSolver(Protocol):
    """Solve ``(J^T W J + D) δ = − J^T W r`` with normal-equation
    structure provided by the caller."""
    def solve(
        self,
        problem: LeastSquaresProblem,
        x: torch.Tensor,
        damping: torch.Tensor | float,
    ) -> torch.Tensor: ...
    capabilities: frozenset[str]                      # e.g. {"dense", "block_sparse"}

class DampingStrategy(Protocol):
    def initial(self) -> float: ...
    def on_accepted(self, current: float) -> float: ...
    def on_rejected(self, current: float) -> float: ...

class RobustKernel(Protocol):
    def rho(self, r2: torch.Tensor) -> torch.Tensor: ...
    def weight(self, r2: torch.Tensor) -> torch.Tensor: ...
```

`LeastSquaresProblem` carries no solver state; `SolverState` carries
per-iteration scalars (cost, damping, accepted-step boolean,
gradient-norm). Tests verify each field is updated as documented.

### 16.H Collision residuals: stable dim, dynamic active-set

The collision residual today exposes a per-pair output but mixes
"all candidate pairs" with "active pairs above the margin". Per
gpt-plan, the contract is:

- `dim = number_of_candidate_pairs` (stable across iterations).
- Pairs outside the safety margin contribute zero — but the slot
  exists, so the Jacobian has a corresponding row of zeros.
- Active-pair compaction is a *kernel-internal* optimisation —
  collision SDFs and matrix-free `J^T r` paths can avoid the work
  on inactive rows without changing the public residual dim.

This makes line-search and damping stable: the dim is fixed, so LM
preallocates once and reuses.

## Files that change

```
tasks/ik.py                          wire OptimizerConfig knobs through to solver
optim/optimizers/levenberg_marquardt.py  accept linear_solver, robust_kernel, damping
optim/optimizers/gauss_newton.py     same
optim/optimizers/adam.py             call problem.gradient(x); document choice
optim/optimizers/lbfgs.py            same
optim/optimizers/lm_then_lbfgs.py    rewritten as MultiStageOptimizer instance
optim/optimizers/multi_stage.py      new — MultiStageOptimizer + OptimizerStage
optim/state.py                       SolverState carries per-iter fields per spec
optim/kernels/__init__.py            kernels match RobustKernel Protocol
optim/linear_solvers/                new package — cholesky, qr, lsqr, block_cholesky
optim/damping.py                     DampingStrategy implementations
optim/problem.py                     LeastSquaresProblem.gradient(); jacobian_blocks()
residuals/base.py                    Residual.spec() default impl; apply_jac_transpose() default
residuals/temporal.py                override apply_jac_transpose for banded structure
residuals/collision.py               stable dim contract; sparse spec
tasks/trajopt.py                     parameterization=; defaults to KnotTrajectory
tasks/parameterization.py            new — TrajectoryParameterization, Knot/BSpline impls
tasks/retarget.py                    accept parameterization
tests/optim/test_config_wiring.py    new — every OptimizerConfig field affects state
tests/optim/test_robust_kernels.py   new — kernels affect normal eqn
tests/optim/test_multi_stage.py      new — restore-on-error, weight overrides
tests/optim/test_matrix_free.py      new — gradient(x) == J.T @ r for dense residuals
tests/tasks/test_trajopt_param.py    new — Knot vs BSpline; same final cost on a small problem
```

## Tradeoffs

| For | Against |
|-----|---------|
| `OptimizerConfig` knobs become real, not decorative. | One-time porting cost for the linear-solver and kernel dispatch. |
| Matrix-free path makes long-horizon trajopt feasible without giant Jacobians. | Two solver paths to maintain (dense vs matrix-free). Mitigation: matrix-free is the default for trajopt; LM-dense remains the default for IK. |
| `ResidualSpec` makes block-sparse assembly a normal piece of the solver, not a special case. | New residuals must declare `spec`. Mitigation: a default impl returns dense-with-stated-dim; existing residuals work unchanged. |
| `TrajectoryParameterization` separates the optimisation variable from the trajectory representation. | Conceptually one extra layer for users to learn. Mitigation: default `KnotTrajectory` makes the simple case identity. |
| `MultiStageOptimizer` removes the `LMThenLBFGS` special case. | None significant — `LMThenLBFGS` becomes a thin wrapper for backward compatibility. |

## Acceptance criteria

- A failing test exists today (or is added in the PR that lands
  16.A) demonstrating that `OptimizerConfig(linear_solver="qr")`
  produces a *different* numerical trajectory than the default. The
  PR makes that test pass.
- Robust kernels, when set, change the LM normal-equation
  weighting; a unit test asserts the residual contribution is
  reweighted by `kernel.weight(r²)`.
- `weight=0` does **not** structurally remove a residual. Setting
  `active=False` does. A unit test exercises both.
- `LeastSquaresProblem.gradient(x)` matches `(J(x).T @ r(x))` to
  fp64 ulp for dense residuals.
- Adam and LBFGS, when run on a non-trivial Panda IK problem,
  complete via `problem.gradient(x)` without ever calling
  `problem.jacobian(x)`.
- `MultiStageOptimizer` with stage-wise weight overrides correctly
  restores the original `CostStack` weights after the run, even if
  a stage raises.
- `solve_trajopt(parameterization=BSplineTrajectory(...))` returns
  a `Trajectory` whose final IK-target residual is within
  tolerance, *and* the optimisation variable's tangent dim is
  smaller than the equivalent `KnotTrajectory` problem.
- Collision residual dim is stable across an LM run; the active-pair
  count varies internally without changing the public `dim`.

## Cross-references

- [07_RESIDUALS_COSTS_SOLVERS.md](../../design/07_RESIDUALS_COSTS_SOLVERS.md) —
  the spec these proposals operationalise; `ResidualSpec.time_coupling`
  is already pinned there.
- [Proposal 03 §"What gets retired"](03_replace_pypose.md) —
  `apply_jac_transpose` and analytic Jacobians remain first-class
  even after autograd becomes safe; this proposal explains why.
- [Proposal 08](08_trajectory_lock_in.md) — `Trajectory` is the
  return type; this proposal adds parameterisations on top.
- [Proposal 11](11_quality_gates_ci.md) — adds the new
  optimisation contract tests to the CI bundle.
- [Proposal 14](14_dynamics_milestone_plan.md) — D7 (three-layer
  action model) builds on the optimisation primitives pinned here.
