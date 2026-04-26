# {py:mod}`better_robot.optim.optimizers.levenberg_marquardt`

```{py:module} better_robot.optim.optimizers.levenberg_marquardt
```

```{autodoc2-docstring} better_robot.optim.optimizers.levenberg_marquardt
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`LevenbergMarquardt <better_robot.optim.optimizers.levenberg_marquardt.LevenbergMarquardt>`
  - ```{autodoc2-docstring} better_robot.optim.optimizers.levenberg_marquardt.LevenbergMarquardt
    :summary:
    ```
````

### API

`````{py:class} LevenbergMarquardt(*, lam0: float = 0.0001, factor: float = 2.0, tol: float = 1e-06)
:canonical: better_robot.optim.optimizers.levenberg_marquardt.LevenbergMarquardt

```{autodoc2-docstring} better_robot.optim.optimizers.levenberg_marquardt.LevenbergMarquardt
```

````{py:method} minimize(problem: better_robot.optim.problem.LeastSquaresProblem, *, max_iter: int = 50, linear_solver=None, kernel=None, strategy=None, scheduler=None) -> better_robot.optim.state.SolverState
:canonical: better_robot.optim.optimizers.levenberg_marquardt.LevenbergMarquardt.minimize

```{autodoc2-docstring} better_robot.optim.optimizers.levenberg_marquardt.LevenbergMarquardt.minimize
```

````

`````
