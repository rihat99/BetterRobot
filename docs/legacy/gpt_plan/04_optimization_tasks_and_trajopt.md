# Optimization, Tasks, And Trajectory Optimization

## Current State

The optimization stack has the right bones:

- `Residual` protocol with optional analytic `.jacobian`.
- `ResidualState` carrying `model`, `data`, and `variables`.
- `CostStack` with named, weighted, activatable items.
- `LeastSquaresProblem` with `state_factory`, `x0`, bounds, tangent dimension, and retraction.
- LM, GN, Adam, LBFGS, and `LMThenLBFGS`.
- `solve_ik` as a thin task facade.
- `solve_trajopt` flattening `(T, nq)` into the existing solver interface.

The main issue is that several extension knobs are described but not fully wired, and dense Jacobians will not scale to serious trajectory or humanoid workloads.

## Immediate Fixes

1. Wire optimizer config fields.

   `OptimizerConfig.linear_solver`, `kernel`, and `damping` are accepted by `solve_ik` but not used to construct the solver components. Either wire them now or remove the fields until they are real.

2. Separate robust kernels from residual weights.

   `CostStack` has weights, and `optim/kernels` has robust kernels. The LM/GN loops currently ignore robust kernels. Add a clear rule:

   - residual class produces raw residual,
   - cost item applies scalar weight,
   - robust kernel transforms cost/Jacobian weights at solver time.

3. Make enabled/disabled different from zero weight.

   Keep `active` as structural inclusion and `weight` as numeric scaling. Do not use weight zero to mean disabled, because sparse specs and preallocation need stable active sets.

4. Define batch semantics for solvers.

   Most optimizers currently assume unbatched vectors when computing `r @ r`, `J.mT @ J`, and scalar `float(...)`. Decide whether v1 solvers are unbatched or truly batched. If unbatched, validate and fail early.

## Residual API Evolution

Add an optional spec method:

```python
class Residual(Protocol):
    name: str
    dim: int

    def __call__(self, state: ResidualState) -> torch.Tensor: ...
    def jacobian(self, state: ResidualState) -> torch.Tensor | None: ...
    def spec(self, state: ResidualState) -> ResidualSpec: ...
    def apply_jac_transpose(self, state: ResidualState, vec: torch.Tensor) -> torch.Tensor: ...
```

`ResidualSpec` should describe:

- output dimension,
- tangent dimension,
- dense/block/sparse/matrix-free structure,
- affected timesteps,
- affected joints or frames,
- whether dimension is fixed or input-dependent.

This matters for collision and temporal smoothness.

## Dynamic Residual Dimensions

Collision docs say "one entry per active pair", while the class initializes `dim` to all self pairs. Dynamic active-pair dimensions make solver preallocation and line search harder.

Recommended policy:

- Public residual dimension should be stable for a fixed problem.
- Collision residual returns one value per candidate pair, with zero outside margin, at least for dense solvers.
- A later sparse/matrix-free collision path can compact active pairs internally without changing the public residual dimension.

## Matrix-Free Trajectory Optimization

Dense `(dim, T * nv)` Jacobians are acceptable only for small tests. For long horizon trajectories:

- Residuals should implement `apply_jac_transpose`.
- Solvers should have a first-order or matrix-free path that needs `J^T r`, not full `J`.
- Temporal residuals should exploit banded structure.
- Pose/keyframe residuals should scatter into specific timestep blocks.
- Collision residuals should produce sparse block updates.

Add a `GradientProblem` or extend `LeastSquaresProblem`:

```python
class LeastSquaresProblem:
    def residual(self, x): ...
    def jacobian(self, x): ...
    def gradient(self, x): ...  # J.T @ r without materializing J if possible
```

Then Adam and LBFGS can use `problem.gradient(x)` directly.

## B-Spline Trajectory Parameterization

Do not make `solve_trajopt` only optimize sample points. Add a B-spline parameterization before the API freezes.

Suggested design:

```python
class TrajectoryParameterization(Protocol):
    def unpack(self, z: torch.Tensor) -> Trajectory: ...
    def retract(self, z: torch.Tensor, dz: torch.Tensor) -> torch.Tensor: ...
    @property
    def tangent_dim(self) -> int: ...

class KnotTrajectory(TrajectoryParameterization): ...
class BSplineTrajectory(TrajectoryParameterization): ...
```

For v1:

- default to B-spline for IK-like smooth motion generation,
- keep knot trajectory for debugging and exact per-timestep constraints,
- use ghost/control points as cuRobo does when endpoint behavior matters.

## Multi-Stage Optimization

Keep `LMThenLBFGS`, but generalize the concept:

```python
@dataclass
class OptimizerStage:
    optimizer: Optimizer
    max_iter: int
    active_items: tuple[str, ...] | None = None
    disabled_items: tuple[str, ...] = ()
    tol: float | None = None

class MultiStageOptimizer:
    stages: tuple[OptimizerStage, ...]
```

This allows:

- coarse pose solve,
- collision clearance,
- smoothness refinement,
- final limit cleanup,
- different residual weights per stage later.

## Task Facades

Task APIs should stay thin, but the configs should be real and typed.

### `solve_ik`

Good current shape. Improvements:

- Accept `target: SE3 | torch.Tensor`.
- Return a result containing `SolverState` or at least status/gain/history.
- Wire linear solver, robust kernel, damping strategy.
- Validate batched vs unbatched inputs.
- Add optional collision residual when `robot_collision` is supplied.

### `solve_trajopt`

Should be a facade over a trajectory parameterization:

- accepts sample or B-spline initial guess,
- accepts user-built `CostStack`,
- exposes sparse/matrix-free solver options,
- returns `Trajectory` with optional diagnostics.

### `retarget`

Should reduce to `solve_trajopt`, but needs first-class concepts:

- frame map,
- marker map,
- optional scale/offset calibration,
- source and target frame conventions,
- anatomical constraints for human models.

## Optimizer Component Contracts

Define and test these protocols:

- `Optimizer`
- `LinearSolver`
- `DampingStrategy`
- `RobustKernel`
- `StopScheduler`
- `LineSearch`

The existing files are a good start. The next step is making every constructor path use the same contracts.

## Tests To Add

- `solve_ik` config fields change actual solver components.
- Batched inputs either work or fail with `ShapeError`.
- `CostStack.gradient` equals dense `J.T @ r`.
- Temporal residual `apply_jac_transpose` equals dense Jacobian transpose.
- Collision residual dimension is stable across changing distances.
- `LMThenLBFGS` restores cost item active flags even if stage 2 raises.
- `solve_trajopt` works with both sample and B-spline parameterizations.
- Robust kernels affect LM normal equations.
