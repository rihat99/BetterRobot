# tasks/ — High-Level Task Facades

## Design Rule

Tasks are thin facades. No Jacobian code, no solver loops, no branching for fixed vs floating base. IK and knot trajopt assemble named-block `Problem` instances.

## Implementation Status

| Task | Status |
|------|--------|
| `solve_ik` | Implemented |
| `solve_trajopt` | Named-block `RobotConfig` trajectory with automatic dense/banded routing; non-knot robot parameterisations remain gated |
| `solve_contact_forces` | Implemented for batched floating-base clips through one named force block and shared RNEA provider |
| `Trajectory` | Implemented (`with_batch_dims`, `slice`, `resample(linear|sclerp)`, `downsample`, `to_data`) |
| `smooth_trajectory` | Implemented for batched quaternion and SE3 pose trajectories with explicit kernels |

## solve_ik

Assembles one bounded `RobotConfig` block, `PoseResidual` items, optional limit/rest items, and a lazy `RobotStateProvider`. Pose targets are declared differentiable `Problem.parameters`. Named-block LM/GN/Adam and `lm_then_adam` are supported; the L-BFGS spellings fail honestly. Arbitrary common leading batch axes return per-element diagnostics.

**Single code path** — floating-base is transparent. First 7 DOF of q are base pose for free-flyer models. Solver doesn't need to know.

## solve_trajopt

Adapts active soft `CostStack` items into one `VarSpec("q", (T, nq), RobotConfig(model), time_axis=0)` with a lazy `RobotStateProvider`. Route-aware named-block LM chooses the banded path when every residual declares temporal blocks; forced dense remains the parity oracle and explicit `matrix_free` uses the normal-operator route. `TrajOptResult` exposes `linearization_requested`, `linearization_used`, `linearization_reason`, and `linearization_detail`. Arbitrary leading batch axes return per-element iterations, convergence, and status. Legacy optimizer objects and constraint-kind items fail actionably. `BSplineTrajectory` remains a Euclidean numerical basis utility and is rejected until a separately reviewed manifold-safe mapping exists; M5 does not promise to enable it.

## solve_contact_forces

Fits world-frame point forces for a frozen `(*B, T, nq)` trajectory. Contacts
are named by joint id and gated by a broadcastable `(B..., T, C)` active mask.
The task central-differences velocity/acceleration, freezes world-to-local
rotations, scatters `[force, torque=0]` external wrenches, and runs `rnea_raw`
once per evaluation through a provider. The public term weights are base
wrench, force magnitude, force smoothness, and actuated-torque smoothness.
Gravity is a task argument; do not mutate or replace the caller's model.

## Trajectory

`Trajectory(t, q, v=None, a=None, tau=None)` accepts unbatched `(T, nq)` and batched `(*B, T, nq)`. Methods:

- `with_batch_dims(n)` — view with `n` leading singleton dims
- `slice(t_start, t_end)` — sub-trajectory by time range
- `resample(new_t, kind="linear"|"sclerp")` — manifold-aware quaternion resampling on indices `[3:7]` when `kind="sclerp"`
- `downsample(factor)` — every Nth sample along the time axis
- `to_data(model)` — batched `Data` with FK populated, batch dim = T·B
- `smooth_trajectory(traj, kernel, kind="auto"|"so3"|"se3")` — batched manifold kernel mean via SLERP / ScLERP
