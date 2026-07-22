# tasks/ — High-Level Task Facades

## Design Rule

Tasks are thin facades. No Jacobian code, no private optimizer loops, no
branching for fixed vs floating base. IK and knot trajopt assemble
object-referenced `Problem` graphs and call public optimizers.

## Implementation Status

| Task | Status |
|------|--------|
| `solve_ik` | Implemented |
| `solve_trajopt` | `RobotVariable(..., time_axis=0)` with automatic dense/banded routing; non-knot robot parameterisations remain absent |
| `solve_contact_forces` | Implemented for batched floating-base clips through one force `Variable` and shared RNEA `Node` |
| `Trajectory` | Implemented (`with_batch_dims`, `slice`, `resample(linear|sclerp)`, `downsample`, `to_data`) |
| `smooth_trajectory` | Implemented for batched quaternion and SE3 pose trajectories with explicit kernels |

## solve_ik

Assembles one bounded `RobotVariable`, `PoseResidual` objects, optional
limit/rest residuals, and evaluation-local `RobotState` nodes. Pose and rest
targets are graph-carrying static `Variable` objects.
`differentiable=True` selects LM's guarded implicit backward; other optimizer
choices reject that flag. Object-owned LM/GN, `TorchOptimizer`, and sequential
`lm_then_adam` are supported. Arbitrary common leading batch axes return
per-element diagnostics.

`IKCostConfig` keeps row-scale-style tuning for compatibility: `pose_weight`,
`limit_weight`, and `rest_weight` are squared into residual outer coefficients;
`pos_weight` and `ori_weight` whiten the corresponding pose rows directly.
Refinement toggles residual `enabled` state and restores it without mutating
configured weights.
`IKResult.residual` contains the final whitened rows, including pose
position/orientation row scales but excluding outer pose, limit, and rest
coefficients, reductions, and robust-kernel scaling.

**Single code path** — floating-base is transparent. The first 7 DOF of `q`
are the base pose for free-flyer models; the optimizer does not need to know.

## solve_trajopt

Accepts a caller-owned `RobotVariable(model, q, time_axis=0)` with concrete
residuals, or a trajectory tensor with `q -> Residual` factories. The facade
harvests one `Problem` and invokes an optional `problem -> Optimizer` factory.
Route-aware LM chooses the banded path when every residual declares numeric
temporal blocks; forced dense remains the parity oracle. `TrajOptResult`
exposes `linearization_requested`, `linearization_used` (`"dense"` or
`"banded"`), `linearization_reason`, and `linearization_detail`. Arbitrary
leading batch axes return per-element iterations, convergence, and status.
Callers omit residuals they do not want. B-spline parameterization remains
deferred until a separately reviewed manifold-safe mapping exists.

`TrajOptResult.residual` is the final concatenated whitened row vector. Outer
coefficients, reductions, activity factors, and robust-kernel scaling are not
folded into it.

## solve_contact_forces

Fits world-frame point forces for a frozen `(*B, T, nq)` trajectory. Contacts
are named by joint id and gated by a broadcastable `(B..., T, C)` active mask.
The task central-differences velocity/acceleration, freezes world-to-local
rotations, scatters `[force, torque=0]` external wrenches, and runs `rnea_raw`
once per evaluation through a shared node. The public term weights are base
wrench, force magnitude, force smoothness, and actuated-torque smoothness.
Gravity is a task argument; do not mutate or replace the caller's model.
Final diagnostics preserve available graphs to gravity and active-mask inputs;
the force optimizer and frozen trajectory internals remain detached.
`ContactForceWeights` values are outer objective coefficients and pass through
without square-root conversion. `ContactForceResult.residual` contains final
whitened rows, not coefficient-scaled rows.

## Trajectory

`Trajectory(t, q, v=None, a=None, tau=None)` accepts unbatched `(T, nq)` and batched `(*B, T, nq)`. Methods:

- `with_batch_dims(n)` — view with `n` leading singleton dims
- `slice(t_start, t_end)` — sub-trajectory by time range
- `resample(new_t, kind="linear"|"sclerp")` — quaternion SLERP only on the fixed `[3:7]` block when `kind="sclerp"`; all other coordinates stay linear
- `downsample(factor)` — every Nth sample along the time axis
- `to_data(model)` — batched `Data` with FK populated, batch dim = T·B
- `smooth_trajectory(traj, kernel, kind="auto"|"so3"|"se3")` — batched manifold kernel mean via SLERP / ScLERP
