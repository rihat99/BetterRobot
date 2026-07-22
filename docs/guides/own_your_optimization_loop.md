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

print([round(value, 6) for value in theta.tensor.tolist()])

# Updating a referenced static variable refreshes the problem while LM keeps
# compatible damping state and starts from theta's current value.
problem.update({"observations": y + 1.0})
info = optimizer.optimize()
print([round(value, 6) for value in theta.tensor.tolist()])
print(bool(info.converged))
```

```{testoutput}
[2.0, 1.0]
[2.0, 2.0]
True
```

The `bool(...)` condition is an intentional eager synchronization point owned
by this Python loop. Keep per-element decisions in tensors inside compiled
math. LM's private iteration remains fixed-shape and synchronization-free.

`Problem.update()` atomically replaces current variable tensors and invalidates
shared node memos. The next LM call refreshes terminal artifacts and retains
compatible damping information. After MAXITER or an `enabled`/weight phase
change without an input update, call `optimizer.resume()`; LM keeps values and
cumulative iteration counts but rebuilds phase-specific damping and acceptance
state. Call `optimizer.reset()` when you instead want all algorithm state and
counts cleared; it deliberately keeps the variables' current values.

Updates address named Variables harvested from the complete residual and node
graph. Use `trainable=False` for observations, masks, or labels that should
change without becoming optimization coordinates; those static Variables may
use boolean or integer tensors. A bare tensor passed to a constructor is a
construction-time constant and cannot be named in `Problem.update()`.

`step()` returns diagnostics for that iteration. `optimize()` is the canonical
complete driver: it stops when every batch element is terminal or the budget
is exhausted and performs the final-point refresh before returning.

## Solve robot groups in phases

The complete {doc}`staged_fit` guide combines this frozen-layout
handoff with preserved Adam moments, a scheduler, input swaps, term logging,
and L-BFGS polish. The shorter pattern below focuses only on the immutable
robot-group boundary.

`RobotVariable` can expose topology-derived tangent groups and exclude whole
groups from an optimization phase. A common floating-base warm-up first
places the root while holding articulated joints fixed, then creates a fresh
unfrozen problem that continues from that result:

```text
from better_robot.optim import LevenbergMarquardt, Problem, RobotVariable
from better_robot.residuals import SmoothnessResidual

# make_residuals is your application-owned residual factory.
root_phase_q = RobotVariable(
    model,
    initial_q,
    name="q",
    time_axis=0,
    frozen_groups=("joints",),
)
root_rows = root_phase_q.tangent_weight(
    {"root_lin": 0.5, "root_ang": 2.0, "joints": 0.1}
)
root_problem = Problem(make_residuals(root_phase_q, root_rows))
LevenbergMarquardt(root_problem).optimize()

# The frozen set is immutable. Transfer the solved value into a new variable.
full_q = RobotVariable(
    model,
    root_phase_q.tensor.detach().clone(),
    name="q",
    time_axis=0,
)
order = 3
full_residuals = make_residuals(full_q, full_q.tangent_weight({"joints": 1.0}))
if full_q.time_length > order:
    full_residuals.append(
        SmoothnessResidual(
            full_q,
            order=order,
            dt=dt,
            coordinate_weight=full_q.tangent_weight(
                {"root_lin": 0.5, "root_ang": 2.0, "joints": 1.0}
            ),
        )
    )
full_problem = Problem(full_residuals)
LevenbergMarquardt(full_problem).optimize()
```

The group weights above are square-root-information row multipliers; they are
not objective coefficients and should not be square-rooted again. A frozen
coordinate stays unchanged exactly and any state bound on that coordinate is
inactive for the phase. Individual joint names may be used wherever a group
name is accepted. Fixed-base models expose `joints` but no `root`,
`root_lin`, or `root_ang` groups.
