# {py:mod}`better_robot.optim.optimizers.gauss_newton`

```{py:module} better_robot.optim.optimizers.gauss_newton
```

```{autodoc2-docstring} better_robot.optim.optimizers.gauss_newton
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`GaussNewton <better_robot.optim.optimizers.gauss_newton.GaussNewton>`
  - ```{autodoc2-docstring} better_robot.optim.optimizers.gauss_newton.GaussNewton
    :summary:
    ```
````

### API

`````{py:class} GaussNewton(*, tol: float = 1e-06, eps: float = 1e-08)
:canonical: better_robot.optim.optimizers.gauss_newton.GaussNewton

```{autodoc2-docstring} better_robot.optim.optimizers.gauss_newton.GaussNewton
```

````{py:method} minimize(problem: better_robot.optim.problem.LeastSquaresProblem, *, max_iter: int = 50, linear_solver=None, kernel=None, strategy=None, scheduler=None) -> better_robot.optim.state.SolverState
:canonical: better_robot.optim.optimizers.gauss_newton.GaussNewton.minimize

```{autodoc2-docstring} better_robot.optim.optimizers.gauss_newton.GaussNewton.minimize
```

````

`````
