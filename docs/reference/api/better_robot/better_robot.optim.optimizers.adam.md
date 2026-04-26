# {py:mod}`better_robot.optim.optimizers.adam`

```{py:module} better_robot.optim.optimizers.adam
```

```{autodoc2-docstring} better_robot.optim.optimizers.adam
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Adam <better_robot.optim.optimizers.adam.Adam>`
  - ```{autodoc2-docstring} better_robot.optim.optimizers.adam.Adam
    :summary:
    ```
````

### API

`````{py:class} Adam(*, lr: float = 0.01, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-08, tol: float = 1e-06)
:canonical: better_robot.optim.optimizers.adam.Adam

```{autodoc2-docstring} better_robot.optim.optimizers.adam.Adam
```

````{py:method} minimize(problem: better_robot.optim.problem.LeastSquaresProblem, *, max_iter: int = 200, linear_solver=None, kernel=None, strategy=None, scheduler=None) -> better_robot.optim.state.SolverState
:canonical: better_robot.optim.optimizers.adam.Adam.minimize

```{autodoc2-docstring} better_robot.optim.optimizers.adam.Adam.minimize
```

````

`````
