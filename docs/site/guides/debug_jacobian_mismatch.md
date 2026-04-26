# Debug an analytic-Jacobian mismatch

If a residual's analytic Jacobian disagrees with the LM-solver's
update direction, the fastest way to localise the bug is to compare
the analytic Jacobian against central finite differences:

```python
from better_robot.kinematics.jacobian import residual_jacobian
from better_robot.kinematics.jacobian_strategy import JacobianStrategy

J_analytic = residual_jacobian(residual, state, strategy=JacobianStrategy.ANALYTIC)
J_fd       = residual_jacobian(residual, state, strategy=JacobianStrategy.FINITE_DIFF)
print((J_analytic - J_fd).abs().max())
```

Use fp64 for both. A discrepancy larger than ~1e-5 typically points at
a sign error in the analytic block or a missing `Jr_inv(log_err)` for
the orientation portion of a pose residual.

`JacobianStrategy.AUTO` (the default) prefers analytic and falls back
to central FD per-residual when no analytic implementation is
available.
