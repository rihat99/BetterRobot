# Own the optimization loop

A `Problem` does not force its solvers on you. This guide shows three ways to fit it into your own code:
minimize `Problem.loss` with any `torch.optim` optimizer, hand the optimizer to `solve` through
`TorchOptim`, and differentiate through a converged solve with `solve_layer`.

## `Problem.loss` is an ordinary tensor

`problem.loss(x)` returns a differentiable `(B,)` loss for the variables `x`. Put it in a loop like any
other loss: sum over the batch, call `backward`, step.

```python
import torch
from better_robot.optim import LevenbergMarquardt, Problem, TorchOptim, solve, solve_layer

torch.manual_seed(0)
target = torch.randn(4, 2)
problem = Problem({"x": torch.zeros(4, 2)}, data={"target": target})
problem.residual("fit", lambda x, target: x - target)
problem.scalar("small", lambda x: (x**2).sum(-1), weight=1e-2)

x = problem.variables["x"].clone().requires_grad_(True)
optimizer = torch.optim.Adam([x], lr=0.05)
for _ in range(300):
    optimizer.zero_grad()
    problem.loss({"x": x}).sum().backward()
    optimizer.step()
print(problem.losses({"x": x.detach()}))  # per-term (4,) losses
```
```output
{'fit': tensor([0.0005, 0.0010, 0.0006, 0.0002]), 'small': tensor([0.0237, 0.0487, 0.0301, 0.0083])}
```

The loop steps the ambient tensor. For a variable on a manifold (a robot configuration with a free-flyer
quaternion) or with `bounds`, call `problem.project(x)` after each step to map it back. Because the loss is
a plain tensor, you can also add it to a network's training loss.

## `TorchOptim` runs the same loop for you

`TorchOptim(optimizer_cls, scheduler=None, **kwargs)` wraps any `torch.optim` optimizer as a `Solver`, so it
works with `solve`, `run` and `Phase`. It projects onto manifolds and bounds after every step. Keyword
arguments go to the optimizer; `scheduler(optimizer)` may return a learning-rate scheduler that is stepped
once per solver step.

```python
solver = TorchOptim(
    torch.optim.Adam, lr=0.05,
    scheduler=lambda opt: torch.optim.lr_scheduler.StepLR(opt, step_size=100, gamma=0.5),
)
result = solve(problem, solver, iters=300)
print(result.loss)
```
```output
tensor([0.0241, 0.0497, 0.0307, 0.0085])
```

Adam-like optimizers keep batch elements independent. `torch.optim.LBFGS` couples them through its line
search and treats the batch as one problem; use the batched `LBFGS` solver from `better_robot.optim` when
the problems must stay independent.

## Differentiate through a solve with `solve_layer`

`solve_layer(problem, solver, iters=50, x=None, damping=1e-6)` solves without a graph, then attaches the
solution to the autograd graph of the problem's `data`. The value is exactly the solver's `x*`; its
derivative is that of one Newton step `-H⁻¹ g(x*, data)` with a constant Gauss-Newton `H`, which at a
converged solution is the implicit-function derivative of `x*` with respect to the data.

```python
target = torch.randn(4, 2, requires_grad=True)
problem = Problem({"x": torch.zeros(4, 2)}, data={"target": target})
problem.residual("fit", lambda x, target: x - target)

result = solve_layer(problem, LevenbergMarquardt(), iters=20)
result.x["x"].sum().backward()
print(target.grad)  # ones: x* = target, so dx*/dtarget = I
assert torch.allclose(target.grad, torch.ones(4, 2), atol=1e-4)
```
```output
tensor([[1.0000, 1.0000],
        [1.0000, 1.0000],
        [1.0000, 1.0000],
        [1.0000, 1.0000]])
```

!!! note "When the derivative is valid"
    The solve must have converged (check `result.done`). Coordinates on an active bound get zero
    sensitivity. With large residuals at the optimum the Gauss-Newton `H` is an approximation, so the
    derivative is too.

`examples/04_ik_layer.py` uses `solve_layer` as an IK layer: a small network maps a task code to a hand
target, `terms.ik_problem` and `solve_layer` turn the target into a Panda configuration, and a loss on that
configuration trains the network through the solve.

```bash
uv run python examples/04_ik_layer.py
```

## Where next

- [Optimization](../concepts/optimization.md): `loss` versus `linearize`, and the `init`/`step` protocol
  if you want to drive a built-in solver step by step.
- [Differentiate](differentiate.md): gradients through kinematics and dynamics.
