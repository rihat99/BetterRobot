# {py:mod}`better_robot.optim`

```{py:module} better_robot.optim
```

```{autodoc2-docstring} better_robot.optim
:allowtitles:
```

## Subpackages

```{toctree}
:titlesonly:
:maxdepth: 3

better_robot.optim.strategies
better_robot.optim.optimizers
better_robot.optim.kernels
better_robot.optim.solvers
```

## Submodules

```{toctree}
:titlesonly:
:maxdepth: 1

better_robot.optim.state
better_robot.optim.problem
better_robot.optim.jacobian_spec
```

## Package Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`solve <better_robot.optim.solve>`
  - ```{autodoc2-docstring} better_robot.optim.solve
    :summary:
    ```
````

### API

````{py:function} solve(problem: better_robot.optim.problem.LeastSquaresProblem, *, optimizer: better_robot.optim.optimizers.base.Optimizer | None = None, max_iter: int = 50, linear_solver=None, kernel=None, strategy=None, scheduler=None) -> better_robot.optim.state.SolverState
:canonical: better_robot.optim.solve

```{autodoc2-docstring} better_robot.optim.solve
```
````
