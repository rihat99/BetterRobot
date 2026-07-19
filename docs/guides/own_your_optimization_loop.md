# Own your optimization loop

{py:class}`better_robot.optim.LevenbergMarquardt` owns one
{py:class}`better_robot.optim.Problem`. Call `optimize()` for a complete eager
solve or `step()` when your application must schedule one iteration at a time,
inspect public diagnostics, or interleave other work. Solved tensors live in
the variables; `OptimizerInfo` reports per-element status, iterations, cost,
and convergence.

The complete example below fits the line `y = m x + c` while owning the first
solve's loop, then warm-starts after the observations change.

```{testcode}
import torch
from better_robot.optim import (
    LevenbergMarquardt,
    OptimizerStatus,
    Problem,
    Variable,
    residual,
)

x = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64)
y = torch.tensor([1.0, 3.0, 5.0, 7.0], dtype=torch.float64)
theta = Variable(torch.zeros(2, dtype=torch.float64), name="theta")
observations = Variable(y, name="observations", trainable=False)


@residual(theta, observations, dim=4, name="line_error")
def line_error(parameters, measured):
    m = parameters[..., 0:1]
    c = parameters[..., 1:2]
    return m * x + c - measured


problem = Problem([line_error])
optimizer = LevenbergMarquardt(
    problem,
    max_iterations=20,
    tolerance=1e-9,
)

for _ in range(optimizer.max_iterations):
    info = optimizer.step()
    if bool((info.status != OptimizerStatus.RUNNING).all()):
        break

torch.testing.assert_close(
    theta.tensor,
    torch.tensor([2.0, 1.0], dtype=torch.float64),
    atol=2e-8,
    rtol=2e-8,
)

# Updating a referenced static variable refreshes the problem while LM keeps
# compatible damping state and starts from theta's current value.
problem.update({"observations": y + 1.0})
info = optimizer.optimize()
torch.testing.assert_close(
    theta.tensor,
    torch.tensor([2.0, 2.0], dtype=torch.float64),
    atol=2e-8,
    rtol=2e-8,
)
assert bool(info.converged)
```

The `bool(...)` condition is an intentional eager synchronization point owned
by this Python loop. Keep per-element decisions in tensors inside compiled
math. LM's private iteration remains fixed-shape and synchronization-free.

`Problem.update()` atomically replaces current variable tensors and invalidates
shared node memos. The next LM call refreshes terminal artifacts and retains
compatible damping information. Call `optimizer.reset()` when you instead
want fresh optimizer state; it deliberately keeps the variables' current
values.

`step()` returns diagnostics for that iteration. `optimize()` is the canonical
complete driver: it stops when every batch element is terminal or the budget
is exhausted and performs the final-point refresh before returning.
