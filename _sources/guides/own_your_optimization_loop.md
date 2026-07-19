# Own your optimization loop

{py:class}`better_robot.optim.LevenbergMarquardt` offers both a complete eager
driver and three smaller lifecycle methods. Use `run` for an ordinary solve.
Use `init_state`, `update`, and `finalize` when your application must schedule
one solver step at a time, inspect state between steps, or interleave other
work.

The complete example below fits the line `y = m x + c` while owning the loop.

```{testcode}
import torch
from better_robot.optim import LMStatus, LevenbergMarquardt, Problem

x = torch.tensor([0.0, 1.0, 2.0, 3.0])
y = torch.tensor([1.0, 3.0, 5.0, 7.0])


def line_error(ctx):
    m = ctx["theta"][..., 0:1]
    c = ctx["theta"][..., 1:2]
    return m * x + c - ctx["observations"]


problem = Problem(parameters={"observations": y})
problem.add_variable("theta", shape=(2,))
problem.add_residual(line_error, dim=4)

solver = LevenbergMarquardt(max_iter=20)
values = {"theta": torch.zeros(2)}
state = solver.init_state(values, problem)

for _ in range(solver.max_iter):
    if bool((state.status != LMStatus.RUNNING).all()):
        break
    values, state = solver.update(values, state, problem)

values, state = solver.finalize(values, state, problem)
torch.testing.assert_close(
    values["theta"],
    torch.tensor([2.0, 1.0]),
    atol=2e-4,
    rtol=2e-4,
)

# Reuse the previous damping state after the observations change.
shifted = Problem(
    vars=problem.vars,
    residuals=problem.residuals,
    parameters={"observations": y + 1.0},
)
values, state = solver.run(values, shifted, state=state)
torch.testing.assert_close(
    values["theta"],
    torch.tensor([2.0, 2.0]),
    atol=2e-4,
    rtol=2e-4,
)
```

`LMState` stores one tensor value per batch element, including damping,
iterations, cost, convergence, and status. The `bool(...)` check above is an
intentional eager synchronization point owned by this Python loop. Keep the
condition in tensors if a compiled caller owns the stepping schedule.

Passing `state=` to `run` or `solve` refreshes the residual artifacts for the
new problem while retaining compatible damping information. Pass the previous
`values` separately when you also want to warm-start the solution itself.

Always call `finalize` after a manual loop so cost, residual, gradient, and
status describe the final values rather than the point before the last update.
