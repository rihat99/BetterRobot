# {py:mod}`better_robot.optim.optimizers.lbfgs`

```{py:module} better_robot.optim.optimizers.lbfgs
```

```{autodoc2-docstring} better_robot.optim.optimizers.lbfgs
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`LBFGS <better_robot.optim.optimizers.lbfgs.LBFGS>`
  - ```{autodoc2-docstring} better_robot.optim.optimizers.lbfgs.LBFGS
    :summary:
    ```
````

### API

`````{py:class} LBFGS(*, history_size: int = 10, lr: float = 1.0, tol: float = 1e-06, c1: float = 0.0001, max_ls: int = 20)
:canonical: better_robot.optim.optimizers.lbfgs.LBFGS

```{autodoc2-docstring} better_robot.optim.optimizers.lbfgs.LBFGS
```

````{py:method} minimize(problem: better_robot.optim.problem.LeastSquaresProblem, *, max_iter: int = 50, linear_solver=None, kernel=None, strategy=None, scheduler=None) -> better_robot.optim.state.SolverState
:canonical: better_robot.optim.optimizers.lbfgs.LBFGS.minimize

```{autodoc2-docstring} better_robot.optim.optimizers.lbfgs.LBFGS.minimize
```

````

`````
