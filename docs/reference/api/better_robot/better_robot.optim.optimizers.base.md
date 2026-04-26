# {py:mod}`better_robot.optim.optimizers.base`

```{py:module} better_robot.optim.optimizers.base
```

```{autodoc2-docstring} better_robot.optim.optimizers.base
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Optimizer <better_robot.optim.optimizers.base.Optimizer>`
  - ```{autodoc2-docstring} better_robot.optim.optimizers.base.Optimizer
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`OptimizationResult <better_robot.optim.optimizers.base.OptimizationResult>`
  - ```{autodoc2-docstring} better_robot.optim.optimizers.base.OptimizationResult
    :summary:
    ```
````

### API

````{py:data} OptimizationResult
:canonical: better_robot.optim.optimizers.base.OptimizationResult
:value: >
   None

```{autodoc2-docstring} better_robot.optim.optimizers.base.OptimizationResult
```

````

`````{py:class} Optimizer
:canonical: better_robot.optim.optimizers.base.Optimizer

Bases: {py:obj}`typing.Protocol`

```{autodoc2-docstring} better_robot.optim.optimizers.base.Optimizer
```

````{py:method} minimize(problem: better_robot.optim.problem.LeastSquaresProblem, *, max_iter: int, linear_solver, kernel, strategy, scheduler=None) -> better_robot.optim.state.SolverState
:canonical: better_robot.optim.optimizers.base.Optimizer.minimize

```{autodoc2-docstring} better_robot.optim.optimizers.base.Optimizer.minimize
```

````

`````
