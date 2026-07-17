# tasks/ — High-Level Task Facades

## Design Rule

Tasks are thin facades. No Jacobian code, no solver loops, no branching for fixed vs floating base. IK assembles a named-block `Problem`; trajopt temporarily retains the legacy flat stack until M5.

## Implementation Status

| Task | Status |
|------|--------|
| `solve_ik` | Implemented |
| `solve_trajopt` | Implemented with `KnotTrajectory`; non-knot robot parameterisations are gated until M5 |
| `Trajectory` | Implemented (`with_batch_dims`, `slice`, `resample(linear|sclerp)`, `downsample`, `to_data`) |

## solve_ik

Assembles one bounded `RobotConfig` block, `PoseResidual` items, optional limit/rest items, and a lazy `RobotStateProvider`. Pose targets are declared differentiable `Problem.parameters`. Named-block LM/GN/Adam and `lm_then_adam` are supported; the L-BFGS spellings fail honestly. Arbitrary common leading batch axes return per-element diagnostics.

**Single code path** — floating-base is transparent. First 7 DOF of q are base pose for free-flyer models. Solver doesn't need to know.

## solve_trajopt

Flattens a `(T, nq)` knot trajectory into a `LeastSquaresProblem` vector and installs a per-knot `Model.integrate` retraction. The user supplies the `CostStack`; targets/keyframes are expressed via `TimeIndexedResidual(...)`. `BSplineTrajectory` remains a Euclidean numerical basis utility, but `solve_trajopt` rejects it: manifold-safe interpolation/retraction, bounds, and multi-stage replacement are deferred to M5.

## Trajectory

`Trajectory(t, q, v=None, a=None, tau=None)` accepts unbatched `(T, nq)` and batched `(*B, T, nq)`. Methods:

- `with_batch_dims(n)` — view with `n` leading singleton dims
- `slice(t_start, t_end)` — sub-trajectory by time range
- `resample(new_t, kind="linear"|"sclerp")` — manifold-aware quaternion resampling on indices `[3:7]` when `kind="sclerp"`
- `downsample(factor)` — every Nth sample along the time axis
- `to_data(model)` — batched `Data` with FK populated, batch dim = T·B
