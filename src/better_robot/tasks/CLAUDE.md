# tasks/ — High-Level Task Facades

## Design Rule

Tasks are thin facades. No Jacobian code, no solver loops, no branching for fixed vs floating base. They assemble a `CostStack` + `LeastSquaresProblem` and call the optimizer.

## Implementation Status

| Task | Status |
|------|--------|
| `solve_ik` | Fully implemented |
| `solve_trajopt` | Stub (raises NotImplementedError) |
| `retarget` | Stub (raises NotImplementedError) |
| `Trajectory` | Dataclass only (methods are stubs) |

## solve_ik

Assembles: `PoseResidual` per target + `JointPositionLimit` + `RestResidual` into a `CostStack`, wraps in `LeastSquaresProblem`, calls optimizer. Returns `IKResult` with `.q`, `.converged`, `.fk()`, `.frame_pose(name)`.

**Single code path** — floating-base is transparent. First 7 DOF of q are base pose for free-flyer models. Solver doesn't need to know.

## Trajectory

`Trajectory(t, q, v, a, u)` with shape `(B, T, nq)`. `.slice()`, `.resample()`, `.to_data()` are stubs.
