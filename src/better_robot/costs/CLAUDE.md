# costs/ — Residual Functions

Each function maps `(cfg, ...)` → residual vector. Residuals are minimized in least-squares sense by the solvers.

## Signature Contract

All residual functions passed to `CostTerm.residual_fn` must have signature `(x: Tensor) -> Tensor`. Use `functools.partial` to bind extra args (robot, target, etc.) before passing.

## Files

### `_pose.py` — `pose_residual`
Computes `log(T_target^{-1} @ T_actual(cfg))` in se3 tangent space.

Returns shape `(*batch, 6)` = `[pos_err*pos_weight, ori_err*ori_weight]`.

**Weight defaults**: `pos_weight=1.0`, `ori_weight=0.1`. The low ori_weight is intentional — the Panda's default config has ~173° orientation from identity, which puts the SO3 log map near its singularity. Increasing ori_weight causes the optimizer to oscillate.

### `_limits.py` — `limit_residual`
Returns `torch.clamp(lo - cfg, min=0)` and `torch.clamp(cfg - hi, min=0)` concatenated.

**Critical**: the clamp is not optional. Without it, the residuals are nonzero even within limits, acting as a quadratic centering force that overwhelms the pose cost (18 limit dims vs 6 pose dims for a 9-DOF robot).

Joint limits are also enforced by hard projection in the LM solver (via `Problem.lower_bounds/upper_bounds`). The soft cost here provides gradients near the boundary.

### `_regularization.py` — `rest_residual`
Returns `(cfg - rest_pose) * weight`. Used to keep the robot near a comfortable pose and resolve null-space redundancy.

### `_collision.py`, `_manipulability.py`
Stubs — not implemented.

## Adding a New Cost

1. Write `my_cost(cfg: Tensor, ...) -> Tensor` returning a 1D residual
2. Use `functools.partial(my_cost, ...)` to bind non-cfg args
3. Wrap in `CostTerm(residual_fn=..., weight=...)` and add to `Problem.costs`
