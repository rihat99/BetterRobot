# {py:mod}`better_robot.optim2.constrained`

```{py:module} better_robot.optim2.constrained
```

```{autodoc2-docstring} better_robot.optim2.constrained
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`solve_constrained <better_robot.optim2.constrained.solve_constrained>`
  - ```{autodoc2-docstring} better_robot.optim2.constrained.solve_constrained
    :summary:
    ```
````

### API

````{py:function} solve_constrained(problem: better_robot.optim2.problem.Problem, solver: better_robot.optim2.solvers.Solver, *, equalities: collections.abc.Sequence[str] = (), inequalities: collections.abc.Sequence[str] = (), outer: int = 10, iters: int = 30, penalty: float = 10.0, growth: float = 10.0, tolerance: float = 1e-06, x: collections.abc.Mapping[str, torch.Tensor] | None = None) -> better_robot.optim2.phases.Result
:canonical: better_robot.optim2.constrained.solve_constrained

```{autodoc2-docstring} better_robot.optim2.constrained.solve_constrained
```
````
