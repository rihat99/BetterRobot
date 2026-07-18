# {py:mod}`better_robot.optim.first_order`

```{py:module} better_robot.optim.first_order
```

```{autodoc2-docstring} better_robot.optim.first_order
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`FirstOrderResult <better_robot.optim.first_order.FirstOrderResult>`
  - ```{autodoc2-docstring} better_robot.optim.first_order.FirstOrderResult
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`run_first_order <better_robot.optim.first_order.run_first_order>`
  - ```{autodoc2-docstring} better_robot.optim.first_order.run_first_order
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`OptimizerFactory <better_robot.optim.first_order.OptimizerFactory>`
  - ```{autodoc2-docstring} better_robot.optim.first_order.OptimizerFactory
    :summary:
    ```
````

### API

````{py:data} OptimizerFactory
:canonical: better_robot.optim.first_order.OptimizerFactory
:type: typing.TypeAlias
:value: >
   None

```{autodoc2-docstring} better_robot.optim.first_order.OptimizerFactory
```

````

`````{py:class} FirstOrderResult
:canonical: better_robot.optim.first_order.FirstOrderResult

Bases: {py:obj}`typing.NamedTuple`

```{autodoc2-docstring} better_robot.optim.first_order.FirstOrderResult
```

````{py:attribute} step
:canonical: better_robot.optim.first_order.FirstOrderResult.step
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.optim.first_order.FirstOrderResult.step
```

````

````{py:attribute} converged
:canonical: better_robot.optim.first_order.FirstOrderResult.converged
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.optim.first_order.FirstOrderResult.converged
```

````

````{py:attribute} cost
:canonical: better_robot.optim.first_order.FirstOrderResult.cost
:type: torch.Tensor
:value: >
   None

```{autodoc2-docstring} better_robot.optim.first_order.FirstOrderResult.cost
```

````

`````

````{py:function} run_first_order(values: collections.abc.Mapping[str, torch.Tensor], problem: better_robot.optim.problem.Problem, optimizer_factory: better_robot.optim.first_order.OptimizerFactory, *, max_iter: int, tolerance: float, weights: collections.abc.Mapping[str, better_robot.optim.problem.Weight] | None = None) -> tuple[better_robot.optim.variables.Values, better_robot.optim.first_order.FirstOrderResult]
:canonical: better_robot.optim.first_order.run_first_order

```{autodoc2-docstring} better_robot.optim.first_order.run_first_order
```
````
