# costs/ — Residual Functions

Each function maps `(cfg, ...)` → residual vector. Residuals are minimized in least-squares sense by the solvers.

## Signature Contract

All residual functions passed to `CostTerm.residual_fn` must have signature `(x: Tensor) -> Tensor`. Use `functools.partial` to bind extra args (robot, target, etc.) before passing.

## Files

### `_pose.py` — `pose_residual`
Computes `log(T_target^{-1} @ T_actual(cfg))` in se3 tangent space.

Returns shape `(*batch, 6)` = `[pos_err*pos_weight, ori_err*ori_weight]`.

**Weight defaults**: `pos_weight=1.0`, `ori_weight=0.1`. The low ori_weight is intentional — the Panda's default config has ~173° orientation from identity, which puts the SO3 log map near its singularity. Increasing ori_weight causes the optimizer to oscillate.

**`base_pose` arg**: optional `(*batch, 7)` SE3 passthrough to `forward_kinematics`.
Default `None` (fixed base, all existing callers unchanged). Always pass as keyword
argument (`base_pose=...`) when constructing `functools.partial` bindings.

### `_limits.py` — `limit_residual`
Returns `torch.clamp(lo - cfg, min=0)` and `torch.clamp(cfg - hi, min=0)` concatenated.

**Critical**: the clamp is not optional. Without it, the residuals are nonzero even within limits, acting as a quadratic centering force that overwhelms the pose cost (18 limit dims vs 6 pose dims for a 9-DOF robot).

Joint limits are also enforced by hard projection in the LM solver (via `Problem.lower_bounds/upper_bounds`). The soft cost here provides gradients near the boundary.

### `_regularization.py` — `rest_residual`
Returns `(cfg - rest_pose) * weight`. Used to keep the robot near a comfortable pose and resolve null-space redundancy.

### `_collision.py`, `_manipulability.py`
Stubs — not implemented.

### `_jacobian.py` — Analytical Jacobian Functions

**`pose_jacobian(cfg, robot, target_link_index, target_pose, pos_weight, ori_weight, base_pose=None) → (6, n)`**

Geometric (analytical) Jacobian of `pose_residual` wrt `cfg`. Body-frame convention (matches `log(T_target^{-1} @ T_actual)` right log error). Uses cross-product formula:
- Revolute/continuous joint j: `J[:,j] = R_ee^T @ [ω_j × (p_ee - p_j); ω_j]` with weights applied
- Prismatic joint j: `J[:,j] = R_ee^T @ [d_j; 0]`

`robot.get_chain(target_link_index)` determines which joints contribute.

**jlog approximation**: `jlog(T_err) ≈ I` (identity). Valid for errors < ~30°. For large errors the analytical J is an approximation; the solver still converges but may take more iterations.

**`limit_jacobian(cfg, robot) → (2n, n)`**
Diagonal: `-1` where `cfg < lo`, `+1` where `cfg > hi`, `0` inside limits. Matches `limit_residual`.

**`rest_jacobian(cfg, rest_pose) → (n, n)`**
Identity matrix. Matches `rest_residual = cfg - rest_pose`.

## Adding a New Cost

1. Write `my_cost(cfg: Tensor, ...) -> Tensor` returning a 1D residual
2. Use `functools.partial(my_cost, ...)` to bind non-cfg args
3. Wrap in `CostTerm(residual_fn=..., weight=...)` and add to `Problem.costs`
