# {py:mod}`better_robot.optim.optimizers.multi_stage`

```{py:module} better_robot.optim.optimizers.multi_stage
```

```{autodoc2-docstring} better_robot.optim.optimizers.multi_stage
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`OptimizerStage <better_robot.optim.optimizers.multi_stage.OptimizerStage>`
  - ```{autodoc2-docstring} better_robot.optim.optimizers.multi_stage.OptimizerStage
    :summary:
    ```
* - {py:obj}`MultiStageOptimizer <better_robot.optim.optimizers.multi_stage.MultiStageOptimizer>`
  - ```{autodoc2-docstring} better_robot.optim.optimizers.multi_stage.MultiStageOptimizer
    :summary:
    ```
````

### API

`````{py:class} OptimizerStage
:canonical: better_robot.optim.optimizers.multi_stage.OptimizerStage

```{autodoc2-docstring} better_robot.optim.optimizers.multi_stage.OptimizerStage
```

````{py:attribute} optimizer
:canonical: better_robot.optim.optimizers.multi_stage.OptimizerStage.optimizer
:type: typing.Any
:value: >
   None

```{autodoc2-docstring} better_robot.optim.optimizers.multi_stage.OptimizerStage.optimizer
```

````

````{py:attribute} max_iter
:canonical: better_robot.optim.optimizers.multi_stage.OptimizerStage.max_iter
:type: int
:value: >
   50

```{autodoc2-docstring} better_robot.optim.optimizers.multi_stage.OptimizerStage.max_iter
```

````

````{py:attribute} disabled_items
:canonical: better_robot.optim.optimizers.multi_stage.OptimizerStage.disabled_items
:type: tuple[str, ...]
:value: >
   ()

```{autodoc2-docstring} better_robot.optim.optimizers.multi_stage.OptimizerStage.disabled_items
```

````

````{py:attribute} weight_overrides
:canonical: better_robot.optim.optimizers.multi_stage.OptimizerStage.weight_overrides
:type: dict[str, float]
:value: >
   'field(...)'

```{autodoc2-docstring} better_robot.optim.optimizers.multi_stage.OptimizerStage.weight_overrides
```

````

````{py:attribute} linear_solver
:canonical: better_robot.optim.optimizers.multi_stage.OptimizerStage.linear_solver
:type: typing.Any | None
:value: >
   None

```{autodoc2-docstring} better_robot.optim.optimizers.multi_stage.OptimizerStage.linear_solver
```

````

````{py:attribute} kernel
:canonical: better_robot.optim.optimizers.multi_stage.OptimizerStage.kernel
:type: typing.Any | None
:value: >
   None

```{autodoc2-docstring} better_robot.optim.optimizers.multi_stage.OptimizerStage.kernel
```

````

````{py:attribute} strategy
:canonical: better_robot.optim.optimizers.multi_stage.OptimizerStage.strategy
:type: typing.Any | None
:value: >
   None

```{autodoc2-docstring} better_robot.optim.optimizers.multi_stage.OptimizerStage.strategy
```

````

`````

`````{py:class} MultiStageOptimizer(*, stages: collections.abc.Sequence[better_robot.optim.optimizers.multi_stage.OptimizerStage])
:canonical: better_robot.optim.optimizers.multi_stage.MultiStageOptimizer

```{autodoc2-docstring} better_robot.optim.optimizers.multi_stage.MultiStageOptimizer
```

````{py:method} minimize(problem: better_robot.optim.problem.LeastSquaresProblem, *, max_iter: int | None = None, linear_solver=None, kernel=None, strategy=None, scheduler=None) -> better_robot.optim.state.SolverState
:canonical: better_robot.optim.optimizers.multi_stage.MultiStageOptimizer.minimize

```{autodoc2-docstring} better_robot.optim.optimizers.multi_stage.MultiStageOptimizer.minimize
```

````

`````
