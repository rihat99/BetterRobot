# tasks/ — High-Level Task Facades

## Design Rule

Tasks are thin facades. No Jacobian code, no solver loops, no branching for fixed vs floating base. They assemble a `CostStack` + `LeastSquaresProblem` and call the optimizer.

## Implementation Status

| Task | Status |
|------|--------|
| `solve_ik` | Implemented |
| `solve_trajopt` | Implemented (with `KnotTrajectory` + `BSplineTrajectory` parameterisations) |
| `retarget` | Stub (raises `NotImplementedError`) |
| `Trajectory` | Implemented (`with_batch_dims`, `slice`, `resample(linear|sclerp)`, `downsample`, `to_data`) |

## solve_ik

Assembles: `PoseResidual` per target + `JointPositionLimit` + `RestResidual` into a `CostStack`, wraps in `LeastSquaresProblem`, calls optimizer. Returns `IKResult` with `.q`, `.converged`, `.fk()`, `.frame_pose(name)`. Honours every documented `OptimizerConfig` knob (`linear_solver`, `kernel`, `damping`, `optimizer`).

**Single code path** — floating-base is transparent. First 7 DOF of q are base pose for free-flyer models. Solver doesn't need to know.

## solve_trajopt

Flattens a `(T, nq)` trajectory into a `LeastSquaresProblem` vector, installs a per-knot `Model.integrate` retraction (for `KnotTrajectory`) or Euclidean update on control points (for `BSplineTrajectory`, with chain-rule Jacobian `J_q @ B_block`). The user supplies the `CostStack`; targets/keyframes are expressed via `TimeIndexedResidual(...)`.

## Trajectory

`Trajectory(t, q, v=None, a=None, tau=None)` accepts unbatched `(T, nq)` and batched `(*B, T, nq)`. Methods:

- `with_batch_dims(n)` — view with `n` leading singleton dims
- `slice(t_start, t_end)` — sub-trajectory by time range
- `resample(new_t, kind="linear"|"sclerp")` — manifold-aware quaternion resampling on indices `[3:7]` when `kind="sclerp"`
- `downsample(factor)` — every Nth sample along the time axis
- `to_data(model)` — batched `Data` with FK populated, batch dim = T·B
