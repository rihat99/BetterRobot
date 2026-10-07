# Write a custom term

A term is a plain function; this guide makes one fast and trustworthy. You will write a per-knot term and
a velocity term for a trajectory, give both an analytic Jacobian and a time `window`, and prove the
Jacobians right with `check_jacobian`. Background: [Optimization](../concepts/optimization.md) and
[Time and structure](../concepts/time-and-structure.md).

## Start with the function

The parameter names are what the term reads: variables, `data` tensors or `cached` values. Here a 2-D
path `x (B, T, 2)` must stay on a circle of radius `radius`.

```python
import torch
from better_robot.optim import LevenbergMarquardt, Problem, TermFn, check_jacobian, solve

torch.manual_seed(0)
B, T = 3, 20
start = torch.randn(B, T, 2, dtype=torch.float64)

def on_circle(x, radius):
    return x.norm(dim=-1, keepdim=True) - radius  # (B, T, 1)

problem = Problem({"x": start}, data={"radius": torch.tensor(2.0, dtype=torch.float64)}, time=("x",))
problem.residual("circle", on_circle, window=1)
```

This already works: without a `jac` the Jacobian comes from autodiff. `window=1` says that row `t` reads
only knot `t`, which keeps the normal equations banded and lets autodiff differentiate all rows at once.

## Add an analytic Jacobian

For a windowed term, `jac` takes the same arguments as the function and returns one block per variable it
differentiates:

| Variable | Block shape |
|---|---|
| time variable | `(B, T - k + 1, m, k, n)`: row, residual entry, knot in the window, tangent coordinate |
| variable without a time axis | `(B, T - k + 1, m, n)` |
| any variable, no `window` | `(B, m, n)` with the residual flattened to `m` |

Wrap the function and its Jacobian in a `TermFn` together with what it reads:

```python
def on_circle_jac(x, radius):
    direction = x / x.norm(dim=-1, keepdim=True)  # (B, T, 2)
    return {"x": direction[:, :, None, None, :]}  # (B, T, m=1, k=1, n=2)

circle = TermFn(on_circle, reads=("x", "radius"), jac=on_circle_jac, window=1)
```

A velocity term reads two knots, so `window=2` and the block holds `-I` for knot `t` and `+I` for knot
`t + 1`:

```python
def velocity(x):
    return x[:, 1:] - x[:, :-1]  # (B, T - 1, 2)

def velocity_jac(x):
    eye = torch.eye(2, dtype=x.dtype)
    block = torch.stack([-eye, eye], dim=-2)  # (m=2, k=2, n=2)
    return {"x": block.expand(x.shape[0], x.shape[1] - 1, 2, 2, 2)}

smooth = TermFn(velocity, reads=("x",), jac=velocity_jac, window=2)
```

`Problem.residual` picks up `jac` and `window` from a `TermFn`, so you register it like any function:

```python
problem = Problem({"x": start}, data={"radius": torch.tensor(2.0, dtype=torch.float64)}, time=("x",))
problem.residual("circle", circle)
problem.residual("smooth", smooth, weight=0.1)
```

## Verify with `check_jacobian`

`check_jacobian(problem)` differentiates every term that has a `jac` or a `window` row by row with plain
autodiff and reports the largest error on `H` and `g`, relative to the largest entry of `H`. Put this in
your test suite: a wrong `jac` or a row that reads outside its `window` gives a wrong step with no error.

```python
errors = check_jacobian(problem)
print(errors)  # {'circle': ~1e-16, 'smooth': ~1e-16}
assert max(errors.values()) < 1e-9

result = solve(problem, LevenbergMarquardt(), iters=50)
radius_error = (result.x["x"].norm(dim=-1) - 2.0).abs().max()
print(f"largest distance from the circle: {float(radius_error):.1e}")
```
```output
{'circle': 0.0, 'smooth': 0.0}
largest distance from the circle: 2.0e-13
```

Use float64 when you check: in float32 the reference itself has errors around `1e-6`.

## Share work with `cached`

When several terms need the same intermediate, such as a kinematic state, register it once with
`problem.cached(fn, name=...)` and read it by name. It is computed once per evaluation. If a term reads a
cached value, its `jac` still returns blocks for the *variables*, with the chain rule through the cached
value already applied. The robot terms in `better_robot.terms` (`terms/robot.py`, `terms/collision.py`) are
built this way: factories that return `TermFn`s reading the cached `state`.

## Checklist

- Parameter names match variables, data or cached names, or pass `reads=`.
- Output `(B, ..., m)`; with `window=k`, `(B, T - k + 1, m)`.
- `jac` blocks in tangent coordinates, shapes from the table above.
- `check_jacobian` below `1e-9` in float64 in a test.

## Where next

- [Fit a clip](fit-a-clip.md): terms with `window` inside a phased trajectory fit.
- [Enforce hard constraints](hard-constraints.md): the same term as an exact constraint.
- [Extending BetterRobot](../contributing/extending.md): ship a term inside the library.
